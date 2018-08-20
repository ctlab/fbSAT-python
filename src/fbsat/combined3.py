import os
import pickle
import shutil
import subprocess
import tempfile
import time
from collections import deque, namedtuple
from io import StringIO

import pymzn
import regex

from .efsm import *
from .printers import *
from .utils import *

__all__ = ['Instance']

VARIABLES = 'color transition output_event algorithm_0 algorithm_1 first_fired not_fired'


class Instance:

    Reduction = namedtuple('Reduction', VARIABLES + ' totalizer')
    Assignment = namedtuple('Assignment', VARIABLES + ' C K T')

    def __init__(self, *, scenario_tree, C, K=None, T=0, is_minimize=False, sat_solver, filename_prefix='', write_strategy='StringIO', is_reuse=False):
        assert write_strategy in ('direct', 'tempfile', 'StringIO', 'pysat')

        if K is None:
            K = C

        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.T = None
        self.T_start = T
        self.is_minimize = is_minimize
        self.sat_solver = sat_solver
        self.filename_prefix = filename_prefix
        self.write_strategy = write_strategy
        self.is_reuse = is_reuse
        self.stream = None
        self.best = None

        self.is_pysat = self.write_strategy == 'pysat'
        if self.is_pysat:
            from pysat.solvers import Glucose4
            self.oracle = Glucose4()
            self.T_defined = None

    def run(self):
        if self.is_minimize:
            self.run_minimize()
        else:
            assignment = self.run_once()
            if assignment:
                log_success(f'Solution with C={self.C}, K={self.K} has T={assignment.T}')
                self.best = assignment
            else:
                if self.T is None:
                    log_error(f'No solution with C={self.C}, K={self.K}')
                else:
                    log_error(f'No solution with C={self.C}, K={self.K}, T={self.T}')
            log_br()

        if self.best:
            efsm = self.build_efsm(self.best)

            filename_automaton = f'{self.filename_prefix}_comb3_TT_automaton'
            with open(filename_automaton, 'wb') as f:
                pickle.dump(efsm, f, pickle.HIGHEST_PROTOCOL)

            filename_gv = f'{self.filename_prefix}_C{self.best.C}_K{self.best.K}_T{self.best.T}_comb3_TT_efsm.gv'
            os.makedirs(os.path.dirname(filename_gv), exist_ok=True)
            efsm.write_gv(filename_gv)

            output_format = 'svg'
            cmd = f'dot -T{output_format} {filename_gv} -O'
            log_debug(cmd, symbol='$')
            os.system(cmd)

            efsm.verify(self.scenario_tree)

            # =================================================================
            for c in closed_range(1, self.C):
                state = efsm.states[c]
                for k, transition in enumerate(state.transitions, start=1):
                    guard = transition.guard
                    log_debug(f'Truth table: {guard}')
                    input_values = []  # [[bool]] of size (U',X)
                    root_value = []  # [bool]
                    for u in closed_range(1, self.scenario_tree.U):
                        q = guard.truth_table[u - 1]
                        iv = self.scenario_tree.unique_inputs[u - 1]
                        if q == '0':
                            input_values.append([{'0': 'false', '1': ' true'}[v] for v in iv])
                            root_value.append(False)
                        elif q == '1':
                            input_values.append([{'0': 'false', '1': ' true'}[v] for v in iv])
                            root_value.append(True)
                        else:
                            log_debug(f'Don\'t care for input {iv}')
                    U_ = len(input_values)  # maybe lesser than real tree.U
                    X = self.scenario_tree.X
                    log_debug(f'c={c}, k={k}, U\'/U={U_}/{self.scenario_tree.U}, X={X}')
                    # for input_values, root_value in data:
                    #     log_debug(f' >  {input_values} -> {root_value}', symbol=None)
                    for P in closed_range(1, 7):
                        log_debug(f'Trying P={P}...')
                        data = dict(P=P, U=U_, X=X, input_values=input_values, root_value=root_value)
                        pymzn.dict2dzn(data, fout=f'{self.filename_prefix}_C{self.C}_T{self.T}_comb3-bf_c{c}_k{k}_P{P}.dzn')
                        sols = pymzn.minizinc('minizinc/boolean-function.mzn', data=data,
                                              output_vars=['nodetype', 'terminal', 'parent', 'child_left', 'child_right'])
                        try:
                            sols._fetch_all()
                        except pymzn.MiniZincUnsatisfiableError:
                            log_error(f'UNSAT for P={P}!')
                        else:
                            log_success(f'SAT for P={P}!')
                            sol = sols[0]
                            log_debug(f'nodetype = {sol["nodetype"]}')
                            log_debug(f'terminal = {sol["terminal"]}')
                            log_debug(f'parent = {sol["parent"]}')
                            log_debug(f'child_left = {sol["child_left"]}')
                            log_debug(f'child_right = {sol["child_right"]}')
                            new_guard = ParseTreeGuard([None] + sol["nodetype"],
                                                       [None] + sol["terminal"],
                                                       [None] + sol["parent"],
                                                       [None] + sol["child_left"],
                                                       [None] + sol["child_right"])
                            log_debug(f'Minimized guard: {new_guard}')
                            transition.guard = new_guard
                            break
                    else:
                        log_error(f'Cannot minimize c={c}, k={k}')
                    log_br()

            filename_automaton = f'{self.filename_prefix}_comb3_MIN_automaton'
            with open(filename_automaton, 'wb') as f:
                pickle.dump(efsm, f, pickle.HIGHEST_PROTOCOL)

            filename_gv = f'{self.filename_prefix}_C{self.best.C}_K{self.best.K}_T{self.best.T}_comb3_MIN_efsm.gv'
            os.makedirs(os.path.dirname(filename_gv), exist_ok=True)
            efsm.write_gv(filename_gv)

            output_format = 'svg'
            cmd = f'dot -T{output_format} {filename_gv} -O'
            log_debug(cmd, symbol='$')
            os.system(cmd)

            efsm.pprint()
            efsm.verify(self.scenario_tree)
            # =================================================================

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

        # Declare totalizer+comparator for given T
        # if self.T_start != 0:
        if self.T_start:
            self.T = self.T_start
            self.generate_totalizer()
            _nv = self.number_of_variables  # base+totalizer variables
            _nc = self.number_of_clauses  # base+totalizer clauses
            self.generate_comparator()

        assignment = self.solve()
        if assignment:
            log_debug(f'Initial estimation for number_of_transition: {assignment.T}')
        log_br()

        # Generate totalizer if it wasn't created
        # if assignment and self.T_start == 0:
        if assignment and not self.T_start:
            self.generate_totalizer()
            _nv = self.number_of_variables  # base+totalizer variables
            _nc = self.number_of_clauses  # base+totalizer clauses

        # Try to minimize number of nodes
        while assignment is not None:
            self.best = assignment
            self.T = assignment.T - 1
            log_info(f'Trying T = {self.T}')

            self.number_of_variables = _nv
            self.number_of_clauses = _nc
            self.generate_comparator()

            assignment = self.solve()
            log_br()

        if self.best:
            log_success(f'Best solution with C={self.best.C}, K={self.best.K} has T={self.best.T}')
        else:
            log_error('Completely UNSAT :c')
        log_br()

    def run_once(self):
        self.number_of_variables = 0
        self.number_of_clauses = 0
        self.generate_base_reduction()

        # if self.T_start != 0:
        if self.T_start:
            self.T = self.T_start
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
        assert self.number_of_variables == 0
        assert self.number_of_clauses == 0

        log_debug(f'Generating base reduction for C={C}, K={K}...')
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
        # X = tree.X
        Z = tree.Z
        U = tree.U
        # Y = tree.Y

        # =-=-=-=-=-=
        #  VARIABLES
        # =-=-=-=-=-=

        # automaton variables
        color = self.declare_array(V, C)
        transition = self.declare_array(C, E, K, C, with_zero=True)
        output_event = self.declare_array(C, O)
        algorithm_0 = self.declare_array(C, Z)
        algorithm_1 = self.declare_array(C, Z)
        # guards variables
        # root_value = self.declare_array(C, E, K, U)
        first_fired = self.declare_array(C, E, U, K + 1)
        not_fired = self.declare_array(C, E, U, K)
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
        # 1.0. ALO/AMO(color)
        for v in closed_range(1, V):
            ALO(color[v])
            AMO(color[v])

        # 1.1. Start vertex corresponds to start state
        #   constraint color[1] = 1;
        self.add_clause(color[1][1])

        # 1.2. Color definition
        for v in closed_range(2, V):
            for c in closed_range(1, C):
                if tree.output_event[v] == 0:
                    # IF toe[v]=0 THEN color[v,c] <=> color[parent[v],c]
                    iff(color[v][c], color[tree.parent[v]][c])
                else:
                    # IF toe[v]!=0 THEN not (color[v,c] and color[parent[v],c])
                    self.add_clause(-color[v][c], -color[tree.parent[v]][c])

        log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

        # 2. Transition constraints
        # 2.0 ALO/AMO(transition)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    ALO(transition[c][e][k])
                    AMO(transition[c][e][k])

        # 2.1. (transition + first_fired definitions)
        for i in closed_range(1, C):
            for e in closed_range(1, E):
                for j in closed_range(1, C):
                    for u in closed_range(1, U):
                        # OR_k( transition[i,e,k,j] & first_fired[i,e,u,k] ) <=> ...
                        # ... <=> OR_{v|active}( color[tp[v],i] & color[v,j] )
                        leftright = self.new_variable()

                        lhs = []
                        for k in closed_range(1, K):
                            # aux <-> transition[i,e,k,j] /\ first_fired[i,e,u,k]
                            aux = self.new_variable()
                            iff_and(aux, (transition[i][e][k][j], first_fired[i][e][u][k]))
                            lhs.append(aux)
                        iff_or(leftright, lhs)

                        rhs = []
                        for v in closed_range(2, V):
                            if tree.output_event[v] != 0 and tree.input_event[v] == e and tree.input_number[v] == u:
                                # aux <-> color[parent[v],i] /\ color[v,j]
                                aux = self.new_variable()
                                p = tree.parent[v]
                                iff_and(aux, (color[p][i], color[v][j]))
                                rhs.append(aux)
                        iff_or(leftright, rhs)

        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K - 1):
                    imply(transition[c][e][k][0], transition[c][e][k + 1][0])

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
        # 5.0. ALO/AMO(first_fired)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    ALO(first_fired[c][e][u])
                    AMO(first_fired[c][e][u])

        # 5.1. (not_fired definition)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    # not_fired[c,e,u,K] <=> OR_{v|passive}(color[v,c])
                    rhs = []
                    for v in closed_range(1, V):
                        if tree.output_event[v] == 0 and tree.input_event[v] == e and tree.input_number[v] == u:
                            rhs.append(color[v][c])
                    iff_or(not_fired[c][e][u][K], rhs)

        # 5.2. not fired
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    for k in closed_range(2, K):
                        imply(not_fired[c][e][u][k], not_fired[c][e][u][k - 1])
                    for k in closed_range(1, K - 1):
                        imply(-not_fired[c][e][u][k], -not_fired[c][e][u][k + 1])

        # 5.3. first_fired
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    for k in closed_range(1, K):
                        # ~(ff & nf)
                        self.add_clause(-first_fired[c][e][u][k], -not_fired[c][e][u][k])
                    for k in closed_range(2, K + 1):
                        # ff_k => nf_{k-1}
                        imply(first_fired[c][e][u][k], not_fired[c][e][u][k - 1])

        # 5.4. Value from fired
        # for c in closed_range(1, C):
        #     for e in closed_range(1, E):
        #         for u in closed_range(1, U):
        #             for k in closed_range(1, K):
        #                 # nf => ~value
        #                 imply(not_fired[c][e][u][k], -root_value[c][e][k][u])
        #                 # ff => value
        #                 imply(first_fired[c][e][u][k], root_value[c][e][k][u])
        #                 # else => unconstrained

        log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        if use_bfs:
            # 12. BFS constraints
            # 12.1. F_t
            for i in closed_range(1, C):
                for j in closed_range(1, C):
                    # t_ij <=> OR_{e,k}(transition_iekj)
                    rhs = []
                    for e in closed_range(1, E):
                        for k in closed_range(1, K):
                            rhs.append(transition[i][e][k][j])
                    iff_or(bfs_transition[i][j], rhs)

            # 12.2. F_p
            for i in closed_range(1, C):
                for j in closed_range(1, i):  # to avoid ambiguous unused variable
                    self.add_clause(-bfs_parent[j][i])
                for j in closed_range(i + 1, C):
                    # p_ji <=> t_ij & AND_[k<i](~t_kj)
                    rhs = [bfs_transition[i][j]]
                    for k in closed_range(1, i - 1):
                        rhs.append(-bfs_transition[k][j])
                    iff_and(bfs_parent[j][i], rhs)

            # 12.3. F_ALO(p)
            for j in closed_range(2, C):
                self.add_clause(*[bfs_parent[j][i] for i in closed_range(1, j - 1)])

            # 12.4. F_BFS(p)
            for k in closed_range(1, K):
                for i in closed_range(k + 1, C):
                    for j in closed_range(i + 1, C - 1):
                        # p_ji => ~p_{j+1,k}
                        imply(bfs_parent[j][i], -bfs_parent[j + 1][k])

            log_debug(f'12. Clauses: {so_far()}', symbol='STAT')

        # Declare any ad-hoc you like
        # AD-HOCs

        # adhoc-1. Distinct transitions
        for i in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    # transition[i,e,k,j] => AND_{k_!=k}(~transition[i,e,k_,j])
                    for j in closed_range(1, C):
                        for k_ in closed_range(k + 1, K):
                            imply(transition[i][e][k][j], -transition[i][e][k_][j])

        log_debug(f'A. Clauses: {so_far()}', symbol='STAT')

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.maybe_close_stream(filename)
        self.reduction = self.Reduction(
            color=color,
            transition=transition,
            output_event=output_event,
            algorithm_0=algorithm_0,
            algorithm_1=algorithm_1,
            # root_value=root_value,
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

        _E = [-self.reduction.transition[c][e][k][0]
              for c in closed_range(1, self.C)
              for e in closed_range(1, self.scenario_tree.E)
              for k in closed_range(1, self.K)]  # set of input variables
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
        log_debug(f'Generating comparator for T={self.T}...')
        time_start_comparator = time.time()
        _nc = self.number_of_clauses
        filename = self.get_filename_comparator()
        self.maybe_new_stream(filename)

        if self.is_pysat and self.T_defined is not None:
            T_max = self.T_defined
        else:
            T_max = self.C * self.scenario_tree.E * self.K

        # sum(E) <= T   <=>   sum(E) < T + 1
        for t in reversed(closed_range(self.T + 1, T_max)):
            self.add_clause(-self.reduction.totalizer[t - 1])  # Note: totalizer is 0-based!

        if self.is_pysat:
            self.T_defined = self.T

        self.maybe_close_stream(filename)
        log_debug(f'Done generating comparator ({self.number_of_clauses-_nc} clauses) in {time.time() - time_start_comparator:.2f} s')

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

        transition = wrapper_int(self.reduction.transition)
        assignment = self.Assignment(
            color=wrapper_int(self.reduction.color),
            transition=transition,
            output_event=wrapper_int(self.reduction.output_event),
            algorithm_0=wrapper_algo(self.reduction.algorithm_0),
            algorithm_1=wrapper_algo(self.reduction.algorithm_1),
            # root_value=wrapper_bool(self.reduction.root_value),
            first_fired=wrapper_bool(self.reduction.first_fired),
            not_fired=wrapper_bool(self.reduction.not_fired),
            C=self.C,
            K=self.K,
            T=sum(transition[c][e][k] != 0
                  for c in closed_range(1, self.C)
                  for e in closed_range(1, self.scenario_tree.E)
                  for k in closed_range(1, self.K)),
        )

        Ks = []
        for c in closed_range(1, self.C):
            for e in closed_range(1, self.scenario_tree.E):
                K_ = sum(transition[c][e][k] != 0 for k in closed_range(1, self.K))
                Ks.append(K_)
                log_debug(f'State c={c} over input event e={e} has K={K_} outcoming transitions')
        if max(Ks) < self.C:
            log_warn(f'Max K = {max(Ks)}, while C = {self.C}, consider lowering K')
        else:
            log_debug(f'Max K = {max(Ks)}, while C = {self.C}')

        # for u in closed_range(1, self.scenario_tree.U):
        #     inputs = ''.join({True: '1', False: '0'}[q] for q in self.scenario_tree.unique_input[u][1:])
        #     log_debug(f'tree.unique_input[u={u: >2}] = {inputs}')

        # for c in closed_range(1, self.C):
        #     for e in closed_range(1, self.scenario_tree.E):
        #         for k in closed_range(1, self.K):
        #             if assignment.transition[c][e][k] != 0:
        #                 truth_table = ''
        #                 for u in closed_range(1, self.scenario_tree.U):
        #                     if assignment.not_fired[c][e][u][k]:
        #                         truth_table += '0'
        #                     elif assignment.first_fired[c][e][u][k]:
        #                         truth_table += '1'
        #                     else:
        #                         truth_table += 'x'
        #                 log_debug(f'{k}th transition ({c} -> {assignment.transition[c][e][k]}) over {e} has truth table: {truth_table}')

        log_debug(f'Done building assignment (T={assignment.T}) in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def maybe_t(self):
        return f'_T{self.T}' if self.T else ''

    def get_filename_prefix(self):
        return f'{self.filename_prefix}_C{self.C}_K{self.K}'

    def get_filename_base(self):
        return f'{self.get_filename_prefix()}_base.dimacs'

    def get_filename_totalizer(self):
        return f'{self.get_filename_prefix()}_totalizer.dimacs'

    def get_filename_comparator(self):
        return f'{self.get_filename_prefix()}{self.maybe_t()}_comparator.dimacs'

    def get_filename_header(self):
        return f'{self.get_filename_prefix()}{self.maybe_t()}_header.dimacs'

    def get_filename_merged(self):
        return f'{self.get_filename_prefix()}{self.maybe_t()}_merged.dimacs'

    def get_filenames(self):
        if self.T:
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
        E = tree.E
        input_events = tree.input_events
        output_events = tree.output_events
        unique_inputs = tree.unique_inputs

        efsm = EFSM()
        for c in closed_range(1, C):
            efsm.add_state(c,
                           output_events[assignment.output_event[c] - 1],
                           assignment.algorithm_0[c],
                           assignment.algorithm_1[c])
        efsm.initial_state = 1

        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    dest = assignment.transition[c][e][k]
                    if dest != 0:
                        input_event = input_events[e - 1]
                        truth_table = ''
                        for u in closed_range(1, self.scenario_tree.U):
                            if assignment.not_fired[c][e][u][k]:
                                truth_table += '0'
                            elif assignment.first_fired[c][e][u][k]:
                                truth_table += '1'
                            else:
                                truth_table += 'x'
                        guard = TruthTableGuard(truth_table, unique_inputs)
                        efsm.add_transition(c, dest, input_event, guard)
                    # else:
                    #     log_debug(f'state {c} has no {k}-th transition over {e}')

        # =======================
        efsm.pprint()
        # =======================

        if efsm.number_of_transitions != assignment.T:
            log_error(f'Inequal number of nodes: efsm has {efsm.number_of_transitions}, assignment has {assignment.T}')

        log_debug(f'Done building EFSM with {efsm.number_of_states} states, {efsm.number_of_transitions} transitions in {time.time() - time_start_build:.2f} s')
        return efsm
