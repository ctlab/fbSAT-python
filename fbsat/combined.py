__all__ = ('Instance', )

import os
import time
import regex
import shutil
import pickle
import tempfile
import subprocess
from io import StringIO
from collections import deque, namedtuple

from .efsm import *
from .utils import *
from .printers import *

VARIABLES = 'color transition trans_event output_event algorithm_0 algorithm_1 nodetype terminal child_left child_right parent value child_value_left child_value_right first_fired not_fired'


class Instance:

    Reduction = namedtuple('Reduction', VARIABLES + ' totalizer')
    Assignment = namedtuple('Assignment', VARIABLES + ' C K P N')

    def __init__(self, *, scenario_tree, C, K, P=None, N=0, is_minimize=False, is_incremental=False, sat_solver, sat_isolver=None, filename_prefix='', write_strategy='StringIO', is_reuse=False):
        assert write_strategy in ('direct', 'tempfile', 'StringIO', 'pysat')

        if is_incremental:
            assert sat_isolver is not None, "You need to specify incremental SAT solver using `--sat-isolver` option"

        if not is_minimize and is_incremental:
            log_warn('Not minimizing -> ignoring incremental')
            is_incremental = False

        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.P = P
        self.N = None
        self.N_start = N
        self.is_minimize = is_minimize
        self.is_incremental = is_incremental
        self.sat_solver = sat_solver
        self.sat_isolver = sat_isolver
        self.filename_prefix = filename_prefix
        self.write_strategy = write_strategy
        self.is_reuse = is_reuse
        self.stream = None
        self.best = None

        self.is_pysat = self.write_strategy == 'pysat'
        if self.is_pysat:
            from pysat.solvers import Glucose4
            self.oracle = Glucose4()
            self.N_defined = None

    def run(self):
        if self.P is None:
            for P in [1, 3, 5, 10, 15]:
                log_info(f'Trying P = {P}')
                self.P = P
                # TODO: refactor solve method by moving incremental part into manual-callable solve_isolver, it will eliminate next dirty is_inc patch
                _is_inc = self.is_incremental
                self.is_incremental = False
                assignment = self.run_once()
                self.is_incremental = _is_inc

                if assignment:
                    log_success(f'Found P = {P}')
                    log_br()

                    if self.is_minimize:
                        self.run_minimize(_reuse_base=True)
                    else:
                        # Reuse assignment found above
                        log_success(f'Solution with C={self.C}, K={self.K}, P={self.P} has N={assignment.N}')
                        log_br()

                    self.best = assignment
                    break
            else:
                log_warn(f'I\'m tired searching for P, are you sure it is SAT with given C and K (C={self.C}, K={self.K})?')
                log_br()

        else:
            if self.is_minimize:
                self.run_minimize()
            else:
                assignment = self.run_once()
                if assignment:
                    log_success(f'Solution with C={self.C}, K={self.K}, P={self.P} has N={assignment.N}')
                    self.best = assignment
                else:
                    if self.N is None:
                        log_error(f'No solution with C={self.C}, K={self.K}, P={self.P}')
                    else:
                        log_error(f'No solution with C={self.C}, K={self.K}, P={self.P}, N={self.N}')
                log_br()

        if self.best:
            efsm = self.build_efsm(self.best)

            filename_automaton = f'{self.filename_prefix}_automaton'
            with open(filename_automaton, 'wb') as f:
                pickle.dump(efsm, f, pickle.HIGHEST_PROTOCOL)

            filename_gv = f'{self.filename_prefix}_C{self.best.C}_K{self.best.K}_P{self.best.P}_N{self.best.N}_efsm.gv'
            os.makedirs(os.path.dirname(filename_gv), exist_ok=True)
            efsm.write_gv(filename_gv)

            output_format = 'svg'
            cmd = f'dot -T{output_format} {filename_gv} -O'
            log_debug(cmd, symbol='$')
            os.system(cmd)

            efsm.verify(self.scenario_tree)

    def run_minimize(self, *, _reuse_base=False):
        self.number_of_variables = 0
        self.number_of_clauses = 0
        if _reuse_base:
            _is_reuse = self.is_reuse
            self.is_reuse = True
            self.generate_base_reduction()
            self.is_reuse = _is_reuse
        else:
            self.generate_base_reduction()

        if self.is_incremental:
            self.isolver_process = subprocess.Popen(self.sat_isolver, shell=True, universal_newlines=True,
                                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            self.feed_isolver(self.get_filename_base())

        # Declare totalizer+comparator for given N
        if self.N_start != 0:
            self.N = self.N_start
            self.generate_totalizer()
            _nv = self.number_of_variables  # base+totalizer variables
            _nc = self.number_of_clauses  # base+totalizer clauses

            if self.is_incremental:
                self.feed_isolver(self.get_filename_totalizer())
                self.generate_comparator_isolver()
            else:
                self.generate_comparator()

        assignment = self.solve()
        if assignment:
            log_debug(f'Initial estimation of number_of_nodes = {assignment.N}')
        log_br()

        if self.is_incremental:
            self.N_defined = None

        # Generate totalizer if it wasn't created
        if assignment and self.N_start == 0:
            self.generate_totalizer()
            _nv = self.number_of_variables  # base+totalizer variables
            _nc = self.number_of_clauses  # base+totalizer clauses

            if self.is_incremental:
                self.feed_isolver(self.get_filename_totalizer())

        # Try to minimize number of nodes
        while assignment is not None:
            self.best = assignment
            self.N = assignment.N - 1
            log_info(f'Trying N = {self.N}')

            if self.is_incremental:
                self.generate_comparator_isolver()
            else:
                self.number_of_variables = _nv
                self.number_of_clauses = _nc
                self.generate_comparator()

            assignment = self.solve()
            log_br()

        if self.best:
            log_success(f'Best solution with C={self.best.C}, K={self.best.K}, P={self.best.P} has N={self.best.N}')
        else:
            log_error('Completely UNSAT :c')
        log_br()

        if self.is_incremental:
            self.isolver_process.kill()

    def run_once(self):
        self.number_of_variables = 0
        self.number_of_clauses = 0
        self.generate_base_reduction()

        if self.N_start != 0:
            self.N = self.N_start
            self.generate_totalizer()
            self.generate_comparator()

        assignment = self.solve()
        log_br()

        return assignment

    def maybe_new_stream(self, filename):
        assert self.stream is None, "Please, be careful creating new stream without closing previous one"
        if self.is_reuse and os.path.exists(filename):
            log_debug(f'Reusing <{filename}>')
            self.stream = None
        elif self.write_strategy == 'direct':
            self.stream = open(filename, 'w')
        elif self.write_strategy == 'tempfile':
            self.stream = tempfile.NamedTemporaryFile('w', delete=False)
        elif self.write_strategy == 'StringIO':
            self.stream = StringIO()
        elif self.is_pysat:
            self.stream = None

    def maybe_close_stream(self, filename):
        if self.stream is not None:
            if self.write_strategy == 'direct':
                self.stream.close()
            elif self.write_strategy == 'tempfile':
                self.stream.close()
                shutil.move(self.stream.name, filename)
            elif self.write_strategy == 'StringIO':
                with open(filename, 'w') as f:
                    self.stream.seek(0)
                    shutil.copyfileobj(self.stream, f)
                self.stream.close()
            self.stream = None

    def feed_isolver(self, filename):
        assert self.is_incremental, "Do not feed isolver when not solving incrementaly"
        log_debug(f'Feeding isolver from <{filename}>...')
        with open(filename) as f:
            shutil.copyfileobj(f, self.isolver_process.stdin)

    def new_variable(self):
        self.number_of_variables += 1
        return self.number_of_variables

    def add_clause(self, *vs):
        self.number_of_clauses += 1
        if self.stream is not None:
            self.stream.write(' '.join(map(str, vs)) + ' 0\n')
        elif self.is_pysat:
            self.oracle.add_clause(vs)

    def declare_array(self, *dims, with_zero=False):
        def last():
            if with_zero:
                return [self.new_variable() for _ in closed_range(0, dims[-1])]
            else:
                return [None] + [self.new_variable() for _ in closed_range(1, dims[-1])]
        n = len(dims)
        if n == 1:
            return last()
        elif n == 2:
            return [None] + [last()
                             for _ in closed_range(1, dims[0])]
        elif n == 3:
            return [None] + [[None] + [last()
                                       for _ in closed_range(1, dims[1])]
                             for _ in closed_range(1, dims[0])]
        elif n == 4:
            return [None] + [[None] + [[None] + [last()
                                                 for _ in closed_range(1, dims[2])]
                                       for _ in closed_range(1, dims[1])]
                             for _ in closed_range(1, dims[0])]
        else:
            raise ValueError(f'unsupported number of dimensions ({n})')

    def generate_base_reduction(self):
        C = self.C
        K = self.K
        P = self.P
        assert self.number_of_variables == 0
        assert self.number_of_clauses == 0

        log_debug(f'Generating base reduction for C={C}, K={K}, P={P}...')
        time_start_base = time.time()
        filename = self.get_filename_base()
        self.maybe_new_stream(filename)

        # =-=-=-=-=-=
        #  CONSTANTS
        # =-=-=-=-=-=

        tree = self.scenario_tree
        V = tree.V
        E = tree.E
        O = tree.O
        X = tree.X
        Z = tree.Z
        U = tree.U
        # Y = tree.Y

        # =-=-=-=-=-=
        #  VARIABLES
        # =-=-=-=-=-=

        # automaton variables
        color = self.declare_array(V, C)
        transition = self.declare_array(C, K, C, with_zero=True)
        trans_event = self.declare_array(C, K, E, with_zero=True)
        output_event = self.declare_array(C, O)
        algorithm_0 = self.declare_array(C, Z)
        algorithm_1 = self.declare_array(C, Z)
        # guards variables
        nodetype = self.declare_array(C, K, P, 4, with_zero=True)
        terminal = self.declare_array(C, K, P, X, with_zero=True)
        child_left = self.declare_array(C, K, P, P, with_zero=True)
        child_right = self.declare_array(C, K, P, P, with_zero=True)
        parent = self.declare_array(C, K, P, P, with_zero=True)
        value = self.declare_array(C, K, P, U)
        child_value_left = self.declare_array(C, K, P, U)
        child_value_right = self.declare_array(C, K, P, U)
        first_fired = self.declare_array(C, U, K)
        not_fired = self.declare_array(C, U, K)
        # bfs variables
        use_bfs = True
        if use_bfs:
            bfs_transition = self.declare_array(C, C)
            bfs_parent = self.declare_array(C, C)

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

        def ALO(data):
            lower = 1 if data[0] is None else 0
            self.add_clause(*data[lower:])

        def AMO(data):
            lower = 1 if data[0] is None else 0
            upper = len(data) - 1
            for a in range(lower, upper):
                for b in closed_range(a + 1, upper):
                    self.add_clause(-data[a], -data[b])

        def imply(lhs, rhs):
            """lhs => rhs"""
            self.add_clause(-lhs, rhs)

        def iff(lhs, rhs):
            """lhs <=> rhs"""
            imply(lhs, rhs)
            imply(rhs, lhs)

        def iff_and(lhs, rhs):
            """lhs <=> AND(rhs)"""
            rhs = tuple(rhs)
            for x in rhs:
                self.add_clause(x, -lhs)
            self.add_clause(lhs, *(-x for x in rhs))

        def iff_or(lhs, rhs):
            """lhs <=> OR(rhs)"""
            rhs = tuple(rhs)
            for x in rhs:
                self.add_clause(-x, lhs)
            self.add_clause(-lhs, *rhs)

        so_far_state = [self.number_of_clauses]

        def so_far():
            now = self.number_of_clauses
            ans = now - so_far_state[0]
            so_far_state[0] = now
            return ans

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # 1. Color constraints
        # 1.0 ALO/AMO(color)
        for v in closed_range(1, V):
            ALO(color[v])
            AMO(color[v])

        # 1.1. Start vertex corresponds to start state
        #   constraint color[1] = 1;
        self.add_clause(color[1][1])

        # 1.2. Color definition
        for v in closed_range(2, V):
            for c in closed_range(1, C):
                if tree.output_event[v] == 0:  # IF toe[v]=0 THEN color[v,c] <=> color[parent[v],c]
                    iff(color[v][c], color[tree.parent[v]][c])
                else:  # IF toe[v]!=0 THEN not (color[v,c] and color[parent[v],c])
                    self.add_clause(-color[v][c], -color[tree.parent[v]][c])

        log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

        # 2. Transition constraints
        # 2.0a ALO/AMO(transition)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                ALO(transition[c][k])
                AMO(transition[c][k])

        # 2.0b ALO/AMO(trans_event)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                ALO(trans_event[c][k])
                AMO(trans_event[c][k])

        # 2.1. (transition + trans_event + first_fired definitions)
        #   constraint forall (v in 2..V where tree_output_event[v] != O+1) (
        #       exists (k in 1..K) (
        #           y[color[tree_previous_active[v]], k] = color[v]
        #           /\ w[color[tree_previous_active[v]], k] = tree_input_event[v]
        #           /\ first_fired[color[tree_previous_active[v]], k, input_nums[v]]
        #       )
        #   );
        for v in closed_range(2, V):
            p = tree.parent[v]
            if tree.output_event[v] != 0:
                for i in closed_range(1, C):  # p's color
                    for j in closed_range(1, C):  # v's color
                        # if i == j:  # FIXME: we could allow loops...
                        #     continue
                        # (color[v,j] & color[p,i]) => OR_k(y & w & ff)
                        constraint = [-color[v][j], -color[p][i]]
                        for k in closed_range(1, K):
                            # aux <-> y[i,k,j] /\ w[i,k,tie[v]] /\ first_fired[i,tin[v],k]
                            aux = self.new_variable()
                            x1 = transition[i][k][j]
                            x2 = trans_event[i][k][tree.input_event[v]]
                            x3 = first_fired[i][tree.input_number[v]][k]
                            iff_and(aux, (x1, x2, x3))
                            constraint.append(aux)
                        self.add_clause(*constraint)

        # for c in closed_range(1, C):
        #     for k in closed_range(1, K):
        #         self.add_clause(-transition[c][k][c])

        for c in closed_range(1, C):
            for k in closed_range(1, K - 1):
                imply(transition[c][k][0], transition[c][k + 1][0])

        for c in closed_range(1, C):
            for k in closed_range(1, K):
                iff(transition[c][k][0], trans_event[c][k][0])

        log_debug(f'2. Clauses: {so_far()}', symbol='STAT')

        # 3. Output event constraints
        # 3.0. ALO/AMO(output_event)
        for c in closed_range(1, C):
            ALO(output_event[c])
            AMO(output_event[c])

        # 3.1. Start state does INITO (root's output event)
        self.add_clause(output_event[1][tree.output_event[1]])

        # 3.2. Output event is the same as in the tree
        for v in closed_range(2, V):
            if tree.output_event[v] != 0:
                for c in closed_range(1, C):
                    imply(color[v][c], output_event[c][tree.output_event[v]])

        log_debug(f'3. Clauses: {so_far()}', symbol='STAT')

        # 4. Algorithm constraints
        # 4.1. Start state does nothing
        for z in closed_range(1, Z):
            self.add_clause(-algorithm_0[1][z])
            self.add_clause(algorithm_1[1][z])

        # 4.2. Algorithms definition
        for v in closed_range(2, V):
            if tree.output_event[v] != 0:
                for z in closed_range(1, Z):
                    old = tree.output_value[tree.parent[v]][z]  # parent/tpa, no difference
                    new = tree.output_value[v][z]
                    for c in closed_range(1, C):
                        if (old, new) == (False, False):
                            imply(color[v][c], -algorithm_0[c][z])
                        elif (old, new) == (False, True):
                            imply(color[v][c], algorithm_0[c][z])
                        elif (old, new) == (True, False):
                            imply(color[v][c], -algorithm_1[c][z])
                        elif (old, new) == (True, True):
                            imply(color[v][c], algorithm_1[c][z])

        log_debug(f'4. Clauses: {so_far()}', symbol='STAT')

        # 5. Firing constraints
        # 5.0. AMO(first_fired)
        for c in closed_range(1, C):
            for u in closed_range(1, U):
                AMO(first_fired[c][u])

        # 5.1. (not_fired definition)
        for v in closed_range(2, V):
            if tree.output_event[v] == 0:
                for c in closed_range(1, C):
                    imply(color[v][c],  # passive: color[v] == color[parent[v]] == color[tpa[v]]
                          not_fired[c][tree.input_number[v]][K])

        # 5.2. not fired
        for c in closed_range(1, C):
            for u in closed_range(1, U):
                for k in closed_range(2, K):
                    imply(not_fired[c][u][k], not_fired[c][u][k - 1])
                for k in closed_range(1, K - 1):
                    imply(-not_fired[c][u][k], -not_fired[c][u][k + 1])

        # 5.3. first_fired
        for c in closed_range(1, C):
            for u in closed_range(1, U):
                for k in closed_range(1, K):
                    # ~(ff & nf)
                    self.add_clause(-first_fired[c][u][k], -not_fired[c][u][k])
                for k in closed_range(2, K):
                    # ff_k => nf_{k-1}
                    imply(first_fired[c][u][k], not_fired[c][u][k - 1])

        # 5.4. Value from fired
        for c in closed_range(1, C):
            for u in closed_range(1, U):
                for k in closed_range(1, K):
                    # nf => ~value
                    imply(not_fired[c][u][k], -value[c][k][1][u])
                    # ff => value
                    imply(first_fired[c][u][k], value[c][k][1][u])
                    # else => unconstrained

        log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        # 6. Guard conditions constraints
        # 6.1. ALO/AMO(nodetype)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(nodetype[c][k][p])
                    AMO(nodetype[c][k][p])

        # 6.2. ALO/AMO(terminal)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(terminal[c][k][p])
                    AMO(terminal[c][k][p])

        # 6.3. ALO/AMO(child_left)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(child_left[c][k][p])
                    AMO(child_left[c][k][p])

        # 6.4. ALO/AMO(child_right)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(child_right[c][k][p])
                    AMO(child_right[c][k][p])

        # 6.5. ALO/AMO(parent)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(parent[c][k][p])
                    AMO(parent[c][k][p])

        log_debug(f'6. Clauses: {so_far()}', symbol='STAT')

        # 7. Extra guard conditions constraints
        # 7.1. Root has no parent
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                self.add_clause(parent[c][k][1][0])

        # 7.2. BFS: typed nodes (except root) have parent with lesser number
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(2, P):
                    # nodetype[p] != 4  =>  OR_par(parent[p] = par)
                    self.add_clause(nodetype[c][k][p][4],
                                    *[parent[c][k][p][par] for par in range(1, p)])

        # 7.3. parent<->child relation
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for ch in closed_range(p + 1, P):
                        # parent[ch]=p => (child_left[p]=ch | child_right[p]=ch)
                        self.add_clause(-parent[c][k][ch][p], child_left[c][k][p][ch], child_right[c][k][p][ch])

        log_debug(f'7. Clauses: {so_far()}', symbol='STAT')

        # 8. None-type nodes constraints
        # 8.1. None-type nodes have largest numbers
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    imply(nodetype[c][k][p][4], nodetype[c][k][p + 1][4])

        # 8.2. None-type nodes have no parent
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    imply(nodetype[c][k][p][4], parent[c][k][p][0])

        # 8.3. None-type nodes have no children
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    imply(nodetype[c][k][p][4], child_left[c][k][p][0])
                    imply(nodetype[c][k][p][4], child_right[c][k][p][0])

        # 8.4. None-type nodes have False value and child_values
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    for u in closed_range(1, U):
                        imply(nodetype[c][k][p][4], -value[c][k][p][u])
                        imply(nodetype[c][k][p][4], -child_value_left[c][k][p][u])
                        imply(nodetype[c][k][p][4], -child_value_right[c][k][p][u])

        log_debug(f'8. Clauses: {so_far()}', symbol='STAT')

        # 9. Terminals constraints
        # 9.1. Only terminals have associated terminal variables
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    iff(nodetype[c][k][p][0], -terminal[c][k][p][0])

        # 9.2. Terminals have no children
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    imply(nodetype[c][k][p][0], child_left[c][k][p][0])
                    imply(nodetype[c][k][p][0], child_right[c][k][p][0])

        # 9.3. Terminals have value from associated input variable
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    for x in closed_range(1, X):
                        for u in closed_range(1, U):
                            if tree.unique_input[u][x]:
                                imply(terminal[c][k][p][x], value[c][k][p][u])
                            else:
                                imply(terminal[c][k][p][x], -value[c][k][p][u])

        log_debug(f'9. Clauses: {so_far()}', symbol='STAT')

        # 10. AND/OR nodes constraints
        # 10.0. AND/OR nodes cannot have numbers P-1 or P
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                if P >= 1:
                    self.add_clause(-nodetype[c][k][P][1])
                    self.add_clause(-nodetype[c][k][P][2])
                if P >= 2:
                    self.add_clause(-nodetype[c][k][P - 1][1])
                    self.add_clause(-nodetype[c][k][P - 1][2])

        # 10.1. AND/OR: left child has greater number
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    _cons = [child_left[c][k][p][ch] for ch in closed_range(p + 1, P - 1)]
                    self.add_clause(-nodetype[c][k][p][1], *_cons)
                    self.add_clause(-nodetype[c][k][p][2], *_cons)

        # 10.2. AND/OR: right child is adjacent (+1) to left
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 1, P - 1):
                        for nt in [1, 2]:
                            # (nodetype[p]=nt & child_left[p]=ch) => child_right[p]=ch+1
                            self.add_clause(-nodetype[c][k][p][nt],
                                            -child_left[c][k][p][ch],
                                            child_right[c][k][p][ch + 1])

        # 10.3. AND/OR: children`s parents
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 1, P - 1):
                        for nt in [1, 2]:
                            # (nodetype[p]=nt & child_left[p]=ch) => (parent[ch]=p & parent[ch+1]=p)
                            _cons = [-nodetype[c][k][p][nt], -child_left[c][k][p][ch]]
                            self.add_clause(*_cons, parent[c][k][ch][p])
                            self.add_clause(*_cons, parent[c][k][ch + 1][p])

        # 10.4a AND/OR: child_value_left is a value of left child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 1, P - 1):
                        for u in closed_range(1, U):
                            for nt in [1, 2]:
                                # (nodetype[p]=nt & child_left[p]=ch) => (child_value_left[p,u] <=> value[ch,u])
                                x1 = nodetype[c][k][p][nt]
                                x2 = child_left[c][k][p][ch]
                                x3 = child_value_left[c][k][p][u]
                                x4 = value[c][k][ch][u]
                                self.add_clause(-x1, -x2, -x3, x4)
                                self.add_clause(-x1, -x2, x3, -x4)

        # 10.4b AND/OR: child_value_right is a value of right child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 2, P):
                        for u in closed_range(1, U):
                            for nt in [1, 2]:
                                # (nodetype[p]=nt & child_right[p]=ch) => (child_value_right[p,u] <=> value[ch,u])
                                x1 = nodetype[c][k][p][nt]
                                x2 = child_right[c][k][p][ch]
                                x3 = child_value_right[c][k][p][u]
                                x4 = value[c][k][ch][u]
                                self.add_clause(-x1, -x2, -x3, x4)
                                self.add_clause(-x1, -x2, x3, -x4)

        # 10.5a AND: value is calculated as a conjunction of children
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for u in closed_range(1, U):
                        # nodetype[p]=1 => (value[p,u] <=> cvl[p,u] & cvr[p,u])
                        x1 = nodetype[c][k][p][1]
                        x2 = value[c][k][p][u]
                        x3 = child_value_left[c][k][p][u]
                        x4 = child_value_right[c][k][p][u]
                        self.add_clause(-x1, x2, -x3, -x4)
                        self.add_clause(-x1, -x2, x3)
                        self.add_clause(-x1, -x2, x4)

        # 10.5b OR: value is calculated as a disjunction of children
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for u in closed_range(1, U):
                        # nodetype[p]=2 => (value[p,u] <=> cvl[p,u] | cvr[p,u])
                        x1 = nodetype[c][k][p][2]
                        x2 = value[c][k][p][u]
                        x3 = child_value_left[c][k][p][u]
                        x4 = child_value_right[c][k][p][u]
                        self.add_clause(-x1, -x2, x3, x4)
                        self.add_clause(-x1, x2, -x3)
                        self.add_clause(-x1, x2, -x4)

        log_debug(f'10. Clauses: {so_far()}', symbol='STAT')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # 11. NOT nodes constraints
        # 11.0. NOT nodes cannot have number P
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                self.add_clause(-nodetype[c][k][P][3])

        # 11.1. NOT: left child has greater number
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    # nodetype[p]=3 => OR_ch(child_left[p]=ch)
                    _cons = [child_left[c][k][p][ch] for ch in closed_range(p + 1, P)]
                    self.add_clause(-nodetype[c][k][p][3], *_cons)

        # 11.2. NOT: no right child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    imply(nodetype[c][k][p][3], child_right[c][k][p][0])

        # 11.3. NOT: child`s parents
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for ch in closed_range(p + 1, P):
                        # (nodetype[p]=3 & child_left[p]=ch) => parent[ch] = p
                        self.add_clause(-nodetype[c][k][p][3],
                                        -child_left[c][k][p][ch],
                                        parent[c][k][ch][p])

        # 11.4a NOT: child_value_left is a value of left child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for ch in closed_range(p + 1, P):
                        for u in closed_range(1, U):
                            # (nodetype[p]=3 & child_left[p]=ch) => (value[ch,u] <=> cvl[p,u])
                            x1 = nodetype[c][k][p][3]
                            x2 = child_left[c][k][p][ch]
                            x3 = value[c][k][ch][u]
                            x4 = child_value_left[c][k][p][u]
                            self.add_clause(-x1, -x2, -x3, x4)
                            self.add_clause(-x1, -x2, x3, -x4)

        # 11.4b NOT: child_value_right is False
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for u in closed_range(1, U):
                        imply(nodetype[c][k][p][3], -child_value_right[c][k][p][u])

        # 11.5. NOT: value is calculated as a negation of child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for u in closed_range(1, U):
                        # nodetype[p]=3 => (value[p,u] <=> ~cvl[p,u])
                        x1 = nodetype[c][k][p][3]
                        x2 = value[c][k][p][u]
                        x3 = child_value_left[c][k][p][u]
                        self.add_clause(-x1, -x2, -x3)
                        self.add_clause(-x1, x2, x3)

        log_debug(f'11. Clauses: {so_far()}', symbol='STAT')

        if use_bfs:
            # 12. BFS constraints
            # 12.1. F_t
            for i in closed_range(1, C):
                for j in closed_range(1, C):
                    if i == j:
                        continue
                    # t_ij <=> OR_k(transition_ikj)
                    iff_or(bfs_transition[i][j],
                           [transition[i][k][j] for k in closed_range(1, K)])
                self.add_clause(-bfs_transition[i][i])

            # 12.2. F_p
            for i in closed_range(1, C):
                for j in closed_range(1, i):
                    self.add_clause(-bfs_parent[j][i])
                for j in closed_range(i + 1, C):
                    # p_ji <=> t_ij & AND_[k<i](~t_kj)
                    iff_and(bfs_parent[j][i],
                            [bfs_transition[i][j]] +
                            [-bfs_transition[k][j] for k in closed_range(1, i - 1)])

            # 12.3. F_ALO(p)
            for j in closed_range(2, C):
                self.add_clause(*[bfs_parent[j][i] for i in closed_range(1, j - 1)])

            # 12.4. F_BFS(p)
            for k in closed_range(1, C):
                for i in closed_range(k + 1, C):
                    for j in closed_range(i + 1, C - 1):
                        # p_ji => ~p_{j+1,k}
                        imply(bfs_parent[j][i], -bfs_parent[j + 1][k])

            log_debug(f'12. Clauses: {so_far()}', symbol='STAT')

        # Declare any ad-hoc you like
        # AD-HOCs

        # adhoc-1
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                iff(transition[c][k][0], nodetype[c][k][1][4])

        # adhoc-2
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for u in closed_range(1, U):
                    # (t!=0 & nf) => nodetype[1]!=4
                    self.add_clause(transition[c][k][0], -not_fired[c][u][k], -nodetype[c][k][1][4])
                    # ff => nodetype[1]!=4
                    imply(first_fired[c][u][k], -nodetype[c][k][1][4])

        log_debug(f'A. Clauses: {so_far()}', symbol='STAT')

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.maybe_close_stream(filename)
        self.reduction = self.Reduction(
            color=color,
            transition=transition,
            trans_event=trans_event,
            output_event=output_event,
            algorithm_0=algorithm_0,
            algorithm_1=algorithm_1,
            nodetype=nodetype,
            terminal=terminal,
            child_left=child_left,
            child_right=child_right,
            parent=parent,
            value=value,
            child_value_left=child_value_left,
            child_value_right=child_value_right,
            first_fired=first_fired,
            not_fired=not_fired,
            totalizer=None
        )

        log_debug(f'Done generating base reduction ({self.number_of_variables} variables, {self.number_of_clauses} clauses) in {time.time() - time_start_base:.2f} s')

    def generate_totalizer(self):
        log_debug('Generating totalizer...')
        time_start_totalizer = time.time()
        _nv = self.number_of_variables
        _nc = self.number_of_clauses
        filename = self.get_filename_totalizer()
        self.maybe_new_stream(filename)

        _E = [-self.reduction.nodetype[c][k][p][4]
              for c in closed_range(1, self.C)
              for k in closed_range(1, self.K)
              for p in closed_range(1, self.P)]  # set of input variables
        _L = []  # set of linking variables

        q = deque([e] for e in _E)
        while len(q) != 1:
            a = q.popleft()  # 0-based
            b = q.popleft()  # 0-based

            m1 = len(a)
            m2 = len(b)
            m = m1 + m2

            r = [self.new_variable() for _ in range(m)]  # 0-based

            if len(q) != 0:
                _L.extend(r)

            for alpha in closed_range(m1):
                for beta in closed_range(m2):
                    sigma = alpha + beta

                    if sigma == 0:
                        C1 = None
                    elif alpha == 0:
                        C1 = [-b[beta - 1], r[sigma - 1]]
                    elif beta == 0:
                        C1 = [-a[alpha - 1], r[sigma - 1]]
                    else:
                        C1 = [-a[alpha - 1], -b[beta - 1], r[sigma - 1]]

                    if sigma == m:
                        C2 = None
                    elif alpha == m1:
                        C2 = [b[beta], -r[sigma]]
                    elif beta == m2:
                        C2 = [a[alpha], -r[sigma]]
                    else:
                        C2 = [a[alpha], b[beta], -r[sigma]]

                    if C1 is not None:
                        self.add_clause(*C1)
                    if C2 is not None:
                        self.add_clause(*C2)

            q.append(r)

        _S = q.pop()  # set of output variables
        assert len(_E) == len(_S)

        self.maybe_close_stream(filename)
        self.reduction = self.reduction._replace(totalizer=_S)  # Note: totalizer is 0-based!

        log_debug(f'Done generating totalizer ({self.number_of_variables-_nv} variables, {self.number_of_clauses-_nc} clauses) in {time.time() - time_start_totalizer:.2f} s')

    def generate_comparator(self):
        log_debug(f'Generating comparator for N={self.N}...')
        time_start_comparator = time.time()
        _nc = self.number_of_clauses
        filename = self.get_filename_comparator()
        self.maybe_new_stream(filename)

        if self.is_pysat and self.N_defined is not None:
            N_max = self.N_defined
        else:
            N_max = self.C * self.K * self.P

        # sum(E) <= N   <=>   sum(E) < N + 1
        for n in reversed(closed_range(self.N + 1, N_max)):
            self.add_clause(-self.reduction.totalizer[n - 1])  # Note: totalizer is 0-based!

        if self.is_pysat:
            self.N_defined = self.N

        self.maybe_close_stream(filename)

        log_debug(f'Done generating comparator ({self.number_of_clauses-_nc} clauses) in {time.time() - time_start_comparator:.2f} s')

    def generate_comparator_isolver(self):
        assert self.is_incremental

        log_debug(f'Generating comparator for N={self.N} and feeding it to isolver...')
        time_start_comparator = time.time()
        _nc = self.number_of_clauses
        self.stream = self.isolver_process.stdin

        if self.N_defined is not None:
            N_max = self.N_defined
        else:
            N_max = self.C * self.K * self.P

        for i in reversed(closed_range(self.N + 1, N_max)):
            self.add_clause(-self.reduction.totalizer[i - 1])  # Note: totalizer is 0-based!

        self.N_defined = self.N

        self.stream = None

        log_debug(f'Done feeding comparator ({self.number_of_clauses-_nc} clauses) to isolver in {time.time() - time_start_comparator:.2f} s')

    def solve(self):
        log_info('Solving...')
        time_start_solve = time.time()

        if self.is_pysat:
            is_sat = self.oracle.solve()

            if is_sat:
                log_success(f'SAT in {time.time() - time_start_solve:.2f} s')
                raw_assignment = [None] + self.oracle.get_model()
                return self.parse_raw_assignment(raw_assignment)
            else:
                log_error(f'UNSAT in {time.time() - time_start_solve:.2f} s')
        elif self.is_incremental:
            p = self.isolver_process
            p.stdin.write('solve 0\n')  # TODO: pass timeout?
            p.stdin.flush()

            answer = p.stdout.readline().rstrip()

            if answer == 'SAT':
                log_success(f'SAT in {time.time() - time_start_solve:.2f} s')
                line_assignment = p.stdout.readline().rstrip()
                if line_assignment.startswith('v '):
                    raw_assignment = [None] + list(map(int, line_assignment[2:].split()))  # 1-based
                    return self.parse_raw_assignment(raw_assignment)
                else:
                    log_error('Error reading line with assignment')
            elif answer == 'UNSAT':
                log_error(f'UNSAT in {time.time() - time_start_solve:.2f} s')
            elif answer == 'UNKNOWN':
                log_error(f'UNKNOWN in {time.time() - time_start_solve:.2f} s')
        else:
            self.write_header()
            self.write_merged()

            cmd = f'{self.sat_solver} {self.get_filename_merged()}'
            log_debug(cmd)
            p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                               universal_newlines=True)

            if p.returncode == 10:
                log_success(f'SAT in {time.time() - time_start_solve:.2f} s')

                raw_assignment = [None]  # 1-based
                for line in p.stdout.split('\n'):
                    if line.startswith('v'):
                        for value in map(int, regex.findall(r'-?\d+', line)):
                            if value == 0:
                                break
                            assert abs(value) == len(raw_assignment)
                            raw_assignment.append(value)
                return self.parse_raw_assignment(raw_assignment)
            elif p.returncode == 20:
                log_error(f'UNSAT in {time.time() - time_start_solve:.2f} s')
            else:
                log_error(f'returncode {p.returncode} in {time.time() - time_start_solve:.2f} s')

    def parse_raw_assignment(self, raw_assignment):
        log_debug('Building assignment...')
        time_start_assignment = time.time()

        def wrapper_int(data):
            if isinstance(data[1], (list, tuple)):
                return [None] + list(map(wrapper_int, data[1:]))
            else:
                for i, x in enumerate(data):
                    if x is not None and raw_assignment[x] > 0:
                        return i
                log_warn('data[...] is unknown')

        def wrapper_bool(data):
            if isinstance(data[1], (list, tuple)):
                return [None] + list(map(wrapper_bool, data[1:]))
            else:
                if data[0] is None:
                    return [NotBool] + [raw_assignment[x] > 0 for x in data[1:]]
                else:
                    return [raw_assignment[x] > 0 for x in data]

        def wrapper_algo(data):
            return [None] + [''.join('1' if raw_assignment[item] > 0 else '0'
                                     for item in subdata[1:])
                             for subdata in data[1:]]

        nodetype = wrapper_int(self.reduction.nodetype)
        assignment = self.Assignment(
            color=wrapper_int(self.reduction.color),
            transition=wrapper_int(self.reduction.transition),
            trans_event=wrapper_int(self.reduction.trans_event),
            output_event=wrapper_int(self.reduction.output_event),
            algorithm_0=wrapper_algo(self.reduction.algorithm_0),
            algorithm_1=wrapper_algo(self.reduction.algorithm_1),
            nodetype=nodetype,
            terminal=wrapper_int(self.reduction.terminal),
            child_left=wrapper_int(self.reduction.child_left),
            child_right=wrapper_int(self.reduction.child_right),
            parent=wrapper_int(self.reduction.parent),
            value=wrapper_bool(self.reduction.value),
            child_value_left=wrapper_bool(self.reduction.child_value_left),
            child_value_right=wrapper_bool(self.reduction.child_value_right),
            first_fired=wrapper_bool(self.reduction.first_fired),
            not_fired=wrapper_bool(self.reduction.not_fired),
            C=self.C,
            K=self.K,
            P=self.P,
            N=sum(nodetype[c][k][p] != 4
                  for c in closed_range(1, self.C)
                  for k in closed_range(1, self.K)
                  for p in closed_range(1, self.P)),
        )

        log_debug(f'Done building assignment in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def maybe_n(self):
        return f'_N{self.N}' if self.N else ''

    def get_filename_prefix(self):
        return f'{self.filename_prefix}_C{self.C}_K{self.K}_P{self.P}'

    def get_filename_base(self):
        return f'{self.get_filename_prefix()}_base.dimacs'

    def get_filename_totalizer(self):
        return f'{self.get_filename_prefix()}_totalizer.dimacs'

    def get_filename_comparator(self):
        return f'{self.get_filename_prefix()}{self.maybe_n()}_comparator.dimacs'

    def get_filename_header(self):
        return f'{self.get_filename_prefix()}{self.maybe_n()}_header.dimacs'

    def get_filename_merged(self):
        return f'{self.get_filename_prefix()}{self.maybe_n()}_merged.dimacs'

    def get_filenames(self):
        if self.N:
            return (self.get_filename_header(),
                    self.get_filename_base(),
                    self.get_filename_totalizer(),
                    self.get_filename_comparator())
        else:
            return (self.get_filename_header(),
                    self.get_filename_base())

    def write_header(self):
        filename = self.get_filename_header()
        if self.is_reuse and os.path.exists(filename):
            log_debug(f'Reusing header from <{filename}>')
            return
        log_debug(f'Writing header to <{filename}>...')
        with open(filename, 'w') as f:
            f.write(f'p cnf {self.number_of_variables} {self.number_of_clauses}\n')

    def write_merged(self):
        filename = self.get_filename_merged()
        if self.is_reuse and os.path.exists(filename):
            log_debug(f'Reusing merged reduction from <{filename}>')
            return
        log_debug(f'Writing merged reduction to <{filename}>...')
        cmd_cat = f'cat {" ".join(self.get_filenames())} > {filename}'
        log_debug(cmd_cat, symbol='$')
        subprocess.run(cmd_cat, shell=True)
        # Replace cat with shutil module:
        # with open(filename, 'w') as merged:
        #     for fn in self.get_filenames():
        #         with open(fn) as f:
        #             shutil.copyfileobj(f, merged)

    def build_efsm(self, assignment):
        log_debug('Building EFSM...')
        time_start_build = time.time()

        tree = self.scenario_tree
        C = assignment.C
        K = assignment.K
        # P = assignment.P
        # N = assignment.N
        input_events = tree.input_events
        output_events = tree.output_events

        efsm = EFSM()
        for c in closed_range(1, C):
            efsm.add_state(c,
                           output_events[assignment.output_event[c] - 1],
                           assignment.algorithm_0[c],
                           assignment.algorithm_1[c])
        efsm.initial_state = 1

        for c in closed_range(1, C):
            for k in closed_range(1, K):
                dest = assignment.transition[c][k]
                if dest != 0:
                    input_event = input_events[assignment.trans_event[c][k] - 1]
                    guard = Guard(assignment.nodetype[c][k],
                                  assignment.terminal[c][k],
                                  assignment.parent[c][k],
                                  assignment.child_left[c][k],
                                  assignment.child_right[c][k])
                    efsm.add_transition(c, dest, input_event, guard)
                else:
                    log_debug(f'state {c} has no {k}-th transition')

        # =======================
        efsm.pprint()
        # =======================

        if efsm.number_of_nodes != assignment.N:
            log_error(f'Inequal number of nodes: efsm has {efsm.number_of_nodes}, assignment has {assignment.N}')

        log_debug(f'Done building EFSM with {efsm.number_of_states} states, {efsm.number_of_transitions} transitions and {efsm.number_of_nodes} nodes in {time.time() - time_start_build:.2f} s')
        return efsm
