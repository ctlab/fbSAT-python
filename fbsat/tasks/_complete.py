import os
import time
from collections import namedtuple
from functools import partial

from ..utils import closed_range, s2b, parse_raw_assignment_int, parse_raw_assignment_bool, parse_raw_assignment_algo
from ..solver import Solver
from ..printers import log_debug, log_success, log_warn, log_br, log_info, log_error
from . import BasicAutomatonTask, MinimalBasicAutomatonTask
from ..efsm import EFSM

__all__ = ['CompleteAutomatonTask']

VARIABLES = 'color transition output_event algorithm_0 algorithm_1 nodetype terminal child_left child_right parent value child_value_left child_value_right first_fired not_fired'


class CompleteAutomatonTask:

    Reduction = namedtuple('Reduction', VARIABLES + ' totalizer')
    Assignment = namedtuple('Assignment', VARIABLES + ' C K T P N')

    # tests-1: C=6, K=3, (T=8), P=3, N=14
    # tests-39: C=8, K=8, (T=16), P=5, N=25 (no-distinct)
    # tests-39: C=8, K=8, (T=15), P=5, N=28 (distinct)

    def __init__(self, scenario_tree, *, C, K=None, P, use_bfs=True, solver_cmd=None, write_strategy=None, outdir=''):
        if K is None:
            K = C

        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.P = P
        self.use_bfs = use_bfs
        self.outdir = outdir
        self.basic_config = dict(use_bfs=use_bfs,
                                 solver_cmd=solver_cmd,
                                 write_strategy=write_strategy,
                                 outdir=outdir)
        self.config = {'cmd': solver_cmd}
        if write_strategy is not None:
            self.config['write_strategy'] = write_strategy

        self._new_solver()

    def get_stem(self, N=None):
        C = self.C
        K = self.K
        P = self.P
        if N is None:
            return f'complete_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_P{P}'
        else:
            return f'complete_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_P{P}_N{N}'

    def get_filename_prefix(self, N=None):
        return os.path.join(self.outdir, self.get_stem(N))

    @property
    def number_of_variables(self):
        return self.solver.number_of_variables

    @property
    def number_of_clauses(self):
        return self.solver.number_of_clauses

    def _new_solver(self):
        self._is_base_declared = False
        self._is_totalizer_declared = False
        self._N_defined = None
        self.solver = Solver(filename_prefix=self.get_filename_prefix(),
                             **self.config)

    def run(self, N=None, *, fast=False):
        # CompleteAutomatonTask: build complete automaton for C, K, P, (N)
        # MinimalCompleteAutomatonTask: finds minimal C and P, minimizes over N

        # find minimal C or use specified
        # find minimal P or use specified
        # minimize over N or use specified

        log_debug(f'CompleteAutomatonTask: running for N={N}...')
        time_start_run = time.time()

        self._declare_base_reduction()
        if N is not None:
            self._declare_totalizer()
            self._declare_comparator(N)

        raw_assignment = self.solver.solve()
        assignment = self.parse_raw_assignment(raw_assignment)

        if fast:
            log_debug(f'CompleteAutomatonTask: done for N={N} in {time.time() - time_start_run:.2f} s')
            return assignment
        else:
            automaton = self.build_efsm(assignment)

            log_debug(f'CompleteAutomatonTask: done for N={N} in {time.time() - time_start_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Complete automaton has {automaton.number_of_states} states, {automaton.number_of_transitions} transitions and {automaton.number_of_nodes} nodes')
            else:
                log_error(f'Complete automaton was not found')
            return automaton

    def _declare_base_reduction(self):
        if self._is_base_declared:
            return
        self._is_base_declared = True

        C = self.C
        K = self.K
        P = self.P
        assert self.number_of_variables == 0
        assert self.number_of_clauses == 0

        log_debug(f'Generating base reduction for C={C}, K={K}, P={P}...')
        time_start_base = time.time()

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

        new_variable = self.solver.new_variable
        add_clause = self.solver.add_clause
        declare_array = self.solver.declare_array
        ALO = self.solver.ALO
        AMO = self.solver.AMO
        imply = self.solver.imply
        iff = self.solver.iff
        iff_and = self.solver.iff_and
        iff_or = self.solver.iff_or

        # =-=-=-=-=-=
        #  VARIABLES
        # =-=-=-=-=-=

        # automaton variables
        color = declare_array(V, C)
        transition = declare_array(C, E, K, C, with_zero=True)
        output_event = declare_array(C, O)
        algorithm_0 = declare_array(C, Z)
        algorithm_1 = declare_array(C, Z)
        # guards variables
        nodetype = declare_array(C, E, K, P, 4, with_zero=True)
        terminal = declare_array(C, E, K, P, X, with_zero=True)
        child_left = declare_array(C, E, K, P, P, with_zero=True)
        child_right = declare_array(C, E, K, P, P, with_zero=True)
        parent = declare_array(C, E, K, P, P, with_zero=True)
        value = declare_array(C, E, K, P, U)
        child_value_left = declare_array(C, E, K, P, U)
        child_value_right = declare_array(C, E, K, P, U)
        first_fired = declare_array(C, E, U, K)
        not_fired = declare_array(C, E, U, K)
        # bfs variables
        if self.use_bfs:
            bfs_transition = declare_array(C, C)
            bfs_parent = declare_array(C, C)

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

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
        add_clause(color[1][1])

        # 1.2. Color definition
        for v in closed_range(2, V):
            for c in closed_range(1, C):
                if tree.output_event[v] == 0:
                    # IF toe[v]=0 THEN color[v,c] <=> color[parent[v],c]
                    iff(color[v][c], color[tree.parent[v]][c])
                else:
                    # IF toe[v]!=0 THEN not (color[v,c] and color[parent[v],c])
                    add_clause(-color[v][c], -color[tree.parent[v]][c])

        log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

        # 2. Transition constraints
        # 2.0. ALO/AMO(transition)
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
                        # ... <=> OR_{v|active,tie[v]=e,tin[v]=u}( color[tp[v],i] & color[v,j] )
                        leftright = new_variable()

                        lhs = []
                        for k in closed_range(1, K):
                            # aux <-> transition[i,e,k,j] & first_fired[i,e,u,k]
                            aux = new_variable()
                            iff_and(aux, (transition[i][e][k][j], first_fired[i][e][u][k]))
                            lhs.append(aux)
                        iff_or(leftright, lhs)

                        rhs = []
                        for v in closed_range(2, V):
                            if tree.output_event[v] != 0 and tree.input_event[v] == e and tree.input_number[v] == u:
                                # aux <-> color[parent[v],i] & color[v,j]
                                aux = new_variable()
                                p = tree.parent[v]
                                iff_and(aux, (color[p][i], color[v][j]))
                                rhs.append(aux)
                        iff_or(leftright, rhs)

        # 2.2. Null-transitions are last
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
        add_clause(output_event[1][tree.output_event[1]])

        # 3.2. Output event is the same as in the tree
        for i in closed_range(1, C):
            for j in closed_range(1, C):
                # OR_{e,k}(transition[i,e,k,j]) <=> ...
                # ... <=> OR_{v|active}( color[tp[v],i] & color[v,j] & output_event[j,toe[v]] )
                leftright = new_variable()

                lhs = []
                for e in closed_range(1, E):
                    for k in closed_range(1, K):
                        lhs.append(transition[i][e][k][j])
                iff_or(leftright, lhs)

                rhs = []
                for v in closed_range(2, V):
                    if tree.output_event[v] != 0:
                        # aux <-> color[parent[v],i] & color[v,j] & output_event[j,toe[v]]
                        aux = new_variable()
                        p = tree.parent[v]
                        o = tree.output_event[v]
                        iff_and(aux, (color[p][i], color[v][j], output_event[j][o]))
                        rhs.append(aux)
                iff_or(leftright, rhs)

        log_debug(f'3. Clauses: {so_far()}', symbol='STAT')

        # 4. Algorithm constraints
        # 4.1. Start state does nothing
        for z in closed_range(1, Z):
            add_clause(-algorithm_0[1][z])
            add_clause(algorithm_1[1][z])

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
                    # ALO(first_fired[c][e][u])
                    AMO(first_fired[c][e][u])

        # 5.1. (not_fired definition)
        for v in closed_range(2, V):
            if tree.output_event[v] == 0:
                for c in closed_range(1, C):
                    # OR_{v|passive}(color[v,c]) => not_fired[c,tie[v],tin[v],K]
                    imply(color[v][c],  # passive: color[v] == color[parent[v]] == color[tpa[v]]
                          not_fired[c][tree.input_event[v]][tree.input_number[v]][K])

        # 5.2. not fired
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    for k in closed_range(2, K):
                        # nf_k => nf_{k-1}
                        imply(not_fired[c][e][u][k], not_fired[c][e][u][k - 1])
                    for k in closed_range(1, K - 1):
                        # ~nf_k => ~nf_{k+1}
                        imply(-not_fired[c][e][u][k], -not_fired[c][e][u][k + 1])

        # 5.3. first_fired
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    for k in closed_range(1, K):
                        # ~(ff & nf)
                        add_clause(-first_fired[c][e][u][k], -not_fired[c][e][u][k])
                    for k in closed_range(2, K):
                        # ff_k => nf_{k-1}
                        imply(first_fired[c][e][u][k], not_fired[c][e][u][k - 1])

        # 5.4. Value from fired
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    for k in closed_range(1, K):
                        # nf => ~root_value
                        imply(not_fired[c][e][u][k], -value[c][e][k][1][u])
                        # ff => root_value
                        imply(first_fired[c][e][u][k], value[c][e][k][1][u])
                        # else => unconstrained

        log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        # 6. Guard conditions constraints
        # 6.1. ALO/AMO(nodetype)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        ALO(nodetype[c][e][k][p])
                        AMO(nodetype[c][e][k][p])

        # 6.2. ALO/AMO(terminal)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        ALO(terminal[c][e][k][p])
                        AMO(terminal[c][e][k][p])

        # 6.3. ALO/AMO(child_left)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        ALO(child_left[c][e][k][p])
                        AMO(child_left[c][e][k][p])

        # 6.4. ALO/AMO(child_right)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        ALO(child_right[c][e][k][p])
                        AMO(child_right[c][e][k][p])

        # 6.5. ALO/AMO(parent)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        ALO(parent[c][e][k][p])
                        AMO(parent[c][e][k][p])

        log_debug(f'6. Clauses: {so_far()}', symbol='STAT')

        # 7. Extra guard conditions constraints
        # 7.1. Root has no parent
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    add_clause(parent[c][e][k][1][0])

        # 7.2. BFS: typed nodes (except root) have parent with lesser number
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(2, P):
                        # nodetype[p] != 4  =>  OR_par(parent[p] = par)
                        add_clause(nodetype[c][e][k][p][4],
                                   *[parent[c][e][k][p][par] for par in closed_range(1, p - 1)])

        # 7.3. parent<->child relation
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 1):
                        for ch in closed_range(p + 1, P):
                            # parent[ch]=p => (child_left[p]=ch | child_right[p]=ch)
                            add_clause(-parent[c][e][k][ch][p], child_left[c][e][k][p][ch], child_right[c][e][k][p][ch])

        log_debug(f'7. Clauses: {so_far()}', symbol='STAT')

        # 8. None-type nodes constraints
        # 8.1. None-type nodes have largest numbers
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 1):
                        imply(nodetype[c][e][k][p][4], nodetype[c][e][k][p + 1][4])

        # 8.2. None-type nodes have no parent
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        imply(nodetype[c][e][k][p][4], parent[c][e][k][p][0])

        # 8.3. None-type nodes have no children
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        imply(nodetype[c][e][k][p][4], child_left[c][e][k][p][0])
                        imply(nodetype[c][e][k][p][4], child_right[c][e][k][p][0])

        # 8.4. None-type nodes have False value and child_values
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        for u in closed_range(1, U):
                            imply(nodetype[c][e][k][p][4], -value[c][e][k][p][u])
                            imply(nodetype[c][e][k][p][4], -child_value_left[c][e][k][p][u])
                            imply(nodetype[c][e][k][p][4], -child_value_right[c][e][k][p][u])

        log_debug(f'8. Clauses: {so_far()}', symbol='STAT')

        # 9. Terminals constraints
        # 9.1. Only terminals have associated terminal variables
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        iff(nodetype[c][e][k][p][0], -terminal[c][e][k][p][0])

        # 9.2. Terminals have no children
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        imply(nodetype[c][e][k][p][0], child_left[c][e][k][p][0])
                        imply(nodetype[c][e][k][p][0], child_right[c][e][k][p][0])

        # 9.3. Terminals have value from associated input variable
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        for x in closed_range(1, X):
                            for u in closed_range(1, U):
                                if tree.unique_input[u][x]:
                                    imply(terminal[c][e][k][p][x], value[c][e][k][p][u])
                                else:
                                    imply(terminal[c][e][k][p][x], -value[c][e][k][p][u])

        log_debug(f'9. Clauses: {so_far()}', symbol='STAT')

        # 10. AND/OR nodes constraints
        # 10.0. AND/OR nodes cannot have numbers P-1 or P
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    if P >= 1:
                        add_clause(-nodetype[c][e][k][P][1])
                        add_clause(-nodetype[c][e][k][P][2])
                    if P >= 2:
                        add_clause(-nodetype[c][e][k][P - 1][1])
                        add_clause(-nodetype[c][e][k][P - 1][2])

        # 10.1. AND/OR: left child has greater number
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 2):
                        # nodetype[p]=nt => OR_{ch from p+1 to P-1}( child_left[p]=ch )
                        _cons = [child_left[c][e][k][p][ch] for ch in closed_range(p + 1, P - 1)]
                        add_clause(-nodetype[c][e][k][p][1], *_cons)
                        add_clause(-nodetype[c][e][k][p][2], *_cons)

        # 10.2. AND/OR: right child is adjacent (+1) to left
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 2):
                        for ch in closed_range(p + 1, P - 1):
                            for nt in [1, 2]:
                                # (nodetype[p]=nt & child_left[p]=ch) => child_right[p]=ch+1
                                add_clause(-nodetype[c][e][k][p][nt],
                                           -child_left[c][e][k][p][ch],
                                           child_right[c][e][k][p][ch + 1])

        # 10.3. AND/OR: children`s parents
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 2):
                        for ch in closed_range(p + 1, P - 1):
                            for nt in [1, 2]:
                                # (nodetype[p]=nt & child_left[p]=ch) => (parent[ch]=p & parent[ch+1]=p)
                                _cons = [-nodetype[c][e][k][p][nt], -child_left[c][e][k][p][ch]]
                                add_clause(*_cons, parent[c][e][k][ch][p])
                                add_clause(*_cons, parent[c][e][k][ch + 1][p])

        # 10.4a. AND/OR: child_value_left is a value of left child
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 2):
                        for ch in closed_range(p + 1, P - 1):
                            for u in closed_range(1, U):
                                for nt in [1, 2]:
                                    # (nodetype[p]=nt & child_left[p]=ch) => (child_value_left[p,u] <=> value[ch,u])
                                    x1 = nodetype[c][e][k][p][nt]
                                    x2 = child_left[c][e][k][p][ch]
                                    x3 = child_value_left[c][e][k][p][u]
                                    x4 = value[c][e][k][ch][u]
                                    add_clause(-x1, -x2, -x3, x4)
                                    add_clause(-x1, -x2, x3, -x4)

        # 10.4b. AND/OR: child_value_right is a value of right child
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 2):
                        for ch in closed_range(p + 2, P):
                            for u in closed_range(1, U):
                                for nt in [1, 2]:
                                    # (nodetype[p]=nt & child_right[p]=ch) => (child_value_right[p,u] <=> value[ch,u])
                                    x1 = nodetype[c][e][k][p][nt]
                                    x2 = child_right[c][e][k][p][ch]
                                    x3 = child_value_right[c][e][k][p][u]
                                    x4 = value[c][e][k][ch][u]
                                    add_clause(-x1, -x2, -x3, x4)
                                    add_clause(-x1, -x2, x3, -x4)

        # 10.5a. AND: value is calculated as a conjunction of children
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 2):
                        for u in closed_range(1, U):
                            # nodetype[p]=1 => (value[p,u] <=> cvl[p,u] & cvr[p,u])
                            x1 = nodetype[c][e][k][p][1]
                            x2 = value[c][e][k][p][u]
                            x3 = child_value_left[c][e][k][p][u]
                            x4 = child_value_right[c][e][k][p][u]
                            add_clause(-x1, x2, -x3, -x4)
                            add_clause(-x1, -x2, x3)
                            add_clause(-x1, -x2, x4)

        # 10.5b. OR: value is calculated as a disjunction of children
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 2):
                        for u in closed_range(1, U):
                            # nodetype[p]=2 => (value[p,u] <=> cvl[p,u] | cvr[p,u])
                            x1 = nodetype[c][e][k][p][2]
                            x2 = value[c][e][k][p][u]
                            x3 = child_value_left[c][e][k][p][u]
                            x4 = child_value_right[c][e][k][p][u]
                            add_clause(-x1, -x2, x3, x4)
                            add_clause(-x1, x2, -x3)
                            add_clause(-x1, x2, -x4)

        log_debug(f'10. Clauses: {so_far()}', symbol='STAT')

        # 11. NOT nodes constraints
        # 11.0. NOT nodes cannot have number P
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    add_clause(-nodetype[c][e][k][P][3])

        # 11.1. NOT: left child has greater number
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 1):
                        # nodetype[p]=3 => OR_ch(child_left[p]=ch)
                        _cons = [child_left[c][e][k][p][ch] for ch in closed_range(p + 1, P)]
                        add_clause(-nodetype[c][e][k][p][3], *_cons)

        # 11.2. NOT: no right child
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 1):
                        imply(nodetype[c][e][k][p][3], child_right[c][e][k][p][0])

        # 11.3. NOT: child`s parents
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 1):
                        for ch in closed_range(p + 1, P):
                            # (nodetype[p]=3 & child_left[p]=ch) => parent[ch] = p
                            add_clause(-nodetype[c][e][k][p][3],
                                       -child_left[c][e][k][p][ch],
                                       parent[c][e][k][ch][p])

        # 11.4a. NOT: child_value_left is a value of left child
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 1):
                        for ch in closed_range(p + 1, P):
                            for u in closed_range(1, U):
                                # (nodetype[p]=3 & child_left[p]=ch) => (value[ch,u] <=> cvl[p,u])
                                x1 = nodetype[c][e][k][p][3]
                                x2 = child_left[c][e][k][p][ch]
                                x3 = value[c][e][k][ch][u]
                                x4 = child_value_left[c][e][k][p][u]
                                add_clause(-x1, -x2, -x3, x4)
                                add_clause(-x1, -x2, x3, -x4)

        # 11.4b. NOT: child_value_right is False
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 1):
                        for u in closed_range(1, U):
                            imply(nodetype[c][e][k][p][3], -child_value_right[c][e][k][p][u])

        # 11.5. NOT: value is calculated as a negation of child
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 1):
                        for u in closed_range(1, U):
                            # nodetype[p]=3 => (value[p,u] <=> ~cvl[p,u])
                            x1 = nodetype[c][e][k][p][3]
                            x2 = value[c][e][k][p][u]
                            x3 = -child_value_left[c][e][k][p][u]
                            add_clause(-x1, -x2, x3)
                            add_clause(-x1, x2, -x3)

        log_debug(f'11. Clauses: {so_far()}', symbol='STAT')

        if self.use_bfs:
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
                    add_clause(-bfs_parent[j][i])
                for j in closed_range(i + 1, C):
                    # p_ji <=> t_ij & AND_[k<i](~t_kj)
                    rhs = [bfs_transition[i][j]]
                    for k in closed_range(1, i - 1):
                        rhs.append(-bfs_transition[k][j])
                    iff_and(bfs_parent[j][i], rhs)

            # 12.3. F_ALO(p)
            for j in closed_range(2, C):
                add_clause(*[bfs_parent[j][i] for i in closed_range(1, j - 1)])

            # 12.4. F_BFS(p)
            for k in closed_range(1, K):
                for i in closed_range(k + 1, C):
                    for j in closed_range(i + 1, C - 1):
                        # p_ji => ~p_{j+1,k}
                        imply(bfs_parent[j][i], -bfs_parent[j + 1][k])

            log_debug(f'12. Clauses: {so_far()}', symbol='STAT')

        # A. Declare any ad-hoc you like

        # adhoc 1. Distinct transitions
        for i in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    # transition[i,e,k,j] => AND_{k_!=k}(~transition[i,e,k_,j])
                    for j in closed_range(1, C):
                        for k_ in closed_range(k + 1, K):
                            imply(transition[i][e][k][j], -transition[i][e][k_][j])

        # adhoc 2. (comb)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    iff(transition[c][e][k][0], nodetype[c][e][k][1][4])

        # adhoc 3. (comb)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for u in closed_range(1, U):
                        # (t!=0 & nf) => nodetype[1]!=4
                        add_clause(transition[c][e][k][0], -not_fired[c][e][u][k], -nodetype[c][e][k][1][4])
                        # ff => nodetype[1]!=4
                        imply(first_fired[c][e][u][k], -nodetype[c][e][k][1][4])

        # adhoc 4. Forbid double negation
        # for c in closed_range(1, C):
        #     for e in closed_range(1, E):
        #         for k in closed_range(1, K):
        #             for p in closed_range(1, P - 1):
        #                 for ch in closed_range(p + 1, P):
        #                     # (nodetype[p]=3 & child_left[p]=ch) => nodetype[ch]!=3
        #                     add_clause(-nodetype[c][e][k][p][3],
        #                                -child_left[c][e][k][p][ch],
        #                                -nodetype[c][e][k][ch][3])
        # adhoc 5. Allow only negation of terminals
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P - 1):
                        for ch in closed_range(p + 1, P):
                            # (nodetype[p]=3 & child_left[p]=ch) => nodetype[ch]=0
                            add_clause(-nodetype[c][e][k][p][3],
                                       -child_left[c][e][k][p][ch],
                                       nodetype[c][e][k][ch][0])

        # adhoc 99. Forbid OR...
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    for p in closed_range(1, P):
                        add_clause(-nodetype[c][e][k][p][2])

        log_debug(f'A. Clauses: {so_far()}', symbol='STAT')

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.reduction = self.Reduction(
            color=color,
            transition=transition,
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

    def _declare_totalizer(self):
        if self._is_totalizer_declared:
            return
        self._is_totalizer_declared = True

        log_debug('Generating totalizer...')
        time_start_totalizer = time.time()
        _nv = self.number_of_variables
        _nc = self.number_of_clauses

        _E = [-self.reduction.nodetype[c][e][k][p][4]
              for c in closed_range(1, self.C)
              for e in closed_range(1, self.scenario_tree.E)
              for p in closed_range(1, self.P)
              for k in closed_range(1, self.K)]
        totalizer = self.solver.get_totalizer(_E)
        self.reduction = self.reduction._replace(totalizer=totalizer)

        log_debug(f'Done generating totalizer ({self.number_of_variables-_nv} variables, {self.number_of_clauses-_nc} clauses) in {time.time() - time_start_totalizer:.2f} s')

    def _declare_comparator(self, N):
        log_debug(f'Generating comparator for N={N}...')
        time_start_comparator = time.time()
        _nc = self.number_of_clauses

        if self._N_defined is not None:
            N_max = self._N_defined
        else:
            N_max = self.C * self.scenario_tree.E * self.K * self.P

        # sum(E) <= N   <=>   sum(E) < N + 1
        for n in reversed(closed_range(N + 1, N_max)):
            self.solver.add_clause(-self.reduction.totalizer[n - 1])  # Note: totalizer is 0-based!

        self._N_defined = N

        log_debug(f'Done generating nodes comparator ({self.number_of_clauses-_nc} clauses) in {time.time() - time_start_comparator:.2f} s')

    def parse_raw_assignment(self, raw_assignment):
        if raw_assignment is None:
            return None

        log_debug('Building assignment...')
        time_start_assignment = time.time()

        wrapper_int = partial(parse_raw_assignment_int, raw_assignment)
        wrapper_bool = partial(parse_raw_assignment_bool, raw_assignment)
        wrapper_algo = partial(parse_raw_assignment_algo, raw_assignment)

        transition = wrapper_int(self.reduction.transition)
        nodetype = wrapper_int(self.reduction.nodetype)
        assignment = self.Assignment(
            color=wrapper_int(self.reduction.color),
            transition=transition,
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
            T=sum(transition[c][e][k] != 0
                  for c in closed_range(1, self.C)
                  for e in closed_range(1, self.scenario_tree.E)
                  for k in closed_range(1, self.K)),
            N=sum(nodetype[c][e][k][p] != 4
                  for c in closed_range(1, self.C)
                  for e in closed_range(1, self.scenario_tree.E)
                  for k in closed_range(1, self.K)
                  for p in closed_range(1, self.P)),
        )

        log_debug(f'Done building assignment (T={assignment.T}, N={assignment.N}) in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def build_efsm(self, assignment, *, dump=True):
        if assignment is None:
            return None

        log_br()
        log_info('CompleteAutomatonTask: building automaton...')
        automaton = EFSM.new_with_parse_trees(self.scenario_tree, assignment)

        if dump:
            filename_gv = self.get_filename_prefix(assignment.N) + '.gv'
            automaton.write_gv(filename_gv)
            output_format = 'svg'
            cmd = f'dot -T{output_format} {filename_gv} -O'
            log_debug(cmd, symbol='$')
            os.system(cmd)

        log_success('Complete automaton:')
        automaton.pprint()
        automaton.verify(self.scenario_tree)

        return automaton
