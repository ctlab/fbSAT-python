import os
import time
from collections import namedtuple
from functools import partial

from . import Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success
from ..solver import IncrementalSolver, StreamSolver
from ..utils import closed_range, parse_raw_assignment_algo, parse_raw_assignment_bool, parse_raw_assignment_int

__all__ = ['FullAutomatonTask']

VARIABLES = 'color transition output_event algorithm_0 algorithm_1'


class FullAutomatonTask:

    Reduction = namedtuple('Reduction', VARIABLES + ' totalizer')
    Assignment = namedtuple('Assignment', VARIABLES + ' C T')

    def __init__(self, scenario_tree, *, C, use_bfs=True, solver_cmd=None, is_incremental=False, outdir=''):
        assert C is not None

        if is_incremental:
            raise NotImplementedError('Wait, incremental solving is not supported just yet!')

        self.scenario_tree = scenario_tree
        self.C = C
        self.use_bfs = use_bfs
        self.is_incremental = is_incremental
        self.outdir = outdir
        self.solver_config = dict(cmd=solver_cmd)
        self._new_solver()

    def _new_solver(self):
        self._is_base_declared = False
        self._is_totalizer_declared = False
        self._T_defined = None
        if self.is_incremental:
            self.solver = IncrementalSolver(**self.solver_config)
        else:
            self.solver = StreamSolver(**self.solver_config)

    def get_stem(self, T=None):
        C = self.C
        if T is None:
            return f'full_{self.scenario_tree.scenarios_stem}_C{C}'
        else:
            return f'full_{self.scenario_tree.scenarios_stem}_C{C}_T{T}'

    def get_filename_prefix(self, T=None):
        return os.path.join(self.outdir, self.get_stem(T))

    @property
    def number_of_variables(self):
        return self.solver.number_of_variables

    @property
    def number_of_clauses(self):
        return self.solver.number_of_clauses

    def run(self, T=None, *, fast=False):
        log_debug(f'FullAutomatonTask: running for T={T}...')
        time_start_run = time.time()

        self._declare_base_reduction()
        if T is not None:
            self._declare_totalizer()
            self._declare_comparator(T)

        raw_assignment = self.solver.solve()
        assignment = self.parse_raw_assignment(raw_assignment)

        if fast:
            log_debug(f'FullAutomatonTask: done for T={T} in {time.time() - time_start_run:.2f} s')
            return assignment
        else:
            automaton = self.build_efsm(assignment)

            log_debug(f'FullAutomatonTask: done for T={T} in {time.time() - time_start_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Full automaton has {automaton.number_of_states} states and {automaton.number_of_transitions} transitions')
            else:
                log_error(f'Full automaton was not found')
            return automaton

    def finalize(self):
        if self.is_incremental:
            self.solver.process.kill()

    def _declare_base_reduction(self):
        if self._is_base_declared:
            return
        self._is_base_declared = True

        C = self.C
        assert self.number_of_variables == 0
        assert self.number_of_clauses == 0

        log_debug(f'Declaring base reduction for C = {C}...')
        time_start_base = time.time()

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
        transition = declare_array(C, E, U, C)
        algorithm_0 = declare_array(C, Z)
        algorithm_1 = declare_array(C, Z)
        output_event = declare_array(C, O)
        # bfs variables
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

        # 1.1. Root corresponds to start state
        add_clause(color[1][1])

        # 1.2. Color definition
        for v in closed_range(2, V):
            for c in closed_range(1, C):
                if tree.output_event[v] == 0:
                    iff(color[v][c], color[tree.parent[v]][c])
                else:
                    add_clause(-color[v][c], -color[tree.parent[v]][c])

        log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

        # 2. Transition constraints
        # 2.0. ALO/AMO(transition)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    ALO(transition[c][e][u])
                    AMO(transition[c][e][u])

        # 2.1. Transition definition
        for i in closed_range(1, C):
            for j in closed_range(1, C):
                if i == j:
                    continue
                for e in closed_range(1, E):
                    for u in closed_range(1, U):
                        # transition[i,e,u,j] <=> OR_{v|...}( color[parent[v],i] & color[v,j] )
                        rhs = []
                        for v in closed_range(2, V):
                            if tree.output_event[v] != 0 and tree.input_event[v] == e and tree.input_number[v] == u:
                                # aux <-> color[parent[v],i] & color[v,j]
                                aux = new_variable()
                                iff_and(aux, (color[tree.parent[v]][i], color[v][j]))
                                rhs.append(aux)
                        iff_or(transition[i][e][u][j], rhs)

        # 2.2. Loop-transitions
        for v in closed_range(2, V):
            for c in closed_range(1, C):
                if tree.output_event[v] == 0:
                    imply(color[v][c], transition[c][tree.input_event[v]][tree.input_number[v]][c])

        log_debug(f'2. Clauses: {so_far()}', symbol='STAT')

        # 3. Algorithm constraints
        # 3.1. Start state does nothing
        for z in closed_range(1, Z):
            add_clause(-algorithm_0[1][z])
            add_clause(algorithm_1[1][z])

        # 3.2. Algorithms definition
        for v in closed_range(2, V):
            if tree.output_event[v] != 0:
                for z in closed_range(1, Z):
                    old = tree.output_value[tree.parent[v]][z]  # tp/tpa, no difference
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

        log_debug(f'3. Clauses: {so_far()}', symbol='STAT')

        # 4. Output event constraints
        # 4.0. ALO/AMO(output_event)
        for c in closed_range(1, C):
            ALO(output_event[c])
            AMO(output_event[c])

        # 4.1. Start state does INITO (root's output event)
        add_clause(output_event[1][tree.output_event[1]])

        # 4.2. Output event is the same as in the tree
        for v in closed_range(2, V):
            o = tree.output_event[v]
            for c in closed_range(1, C):
                if o == 0:
                    imply(color[v][c], output_event[c][tree.output_event[tree.previous_active[v]]])
                else:
                    imply(color[v][c], output_event[c][o])

        log_debug(f'4. Clauses: {so_far()}', symbol='STAT')

        if self.use_bfs:
            # 5. BFS constraints
            # 5.1. F_t
            for i in closed_range(1, C):
                for j in closed_range(1, C):
                    # t_ij <=> OR_{e,u}(transition_ieuj)
                    rhs = []
                    for e in closed_range(1, E):
                        for u in closed_range(1, U):
                            rhs.append(transition[i][e][u][j])
                    iff_or(bfs_transition[i][j], rhs)

            # 5.2. F_p
            for i in closed_range(1, C):
                for j in closed_range(1, i):  # to avoid ambiguous unused variable
                    add_clause(-bfs_parent[j][i])
                for j in closed_range(i + 1, C):
                    # p_ji <=> t_ij & AND_[k<i](~t_kj)
                    rhs = [bfs_transition[i][j]]
                    for k in closed_range(1, i - 1):
                        rhs.append(-bfs_transition[k][j])
                    iff_and(bfs_parent[j][i], rhs)

            # 5.3. F_ALO(p)
            for j in closed_range(2, C):
                ALO(bfs_parent[j][1:j])

            # 5.4. F_BFS(p)
            for k in closed_range(1, C):
                for i in closed_range(k + 1, C):
                    for j in closed_range(i + 1, C - 1):
                        # p_ji => ~p_{j+1,k}
                        imply(bfs_parent[j][i], -bfs_parent[j + 1][k])

            log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        # A. AD-HOCs
        # A.1. ...

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
            totalizer=None,
        )

        log_debug(f'Done declaring base reduction ({self.number_of_variables} variables, {self.number_of_clauses} clauses) in {time.time() - time_start_base:.2f} s')

    def _declare_totalizer(self):
        if self._is_totalizer_declared:
            return
        self._is_totalizer_declared = True

        log_debug('Declaring totalizer...')
        time_start_totalizer = time.time()
        _nv = self.number_of_variables
        _nc = self.number_of_clauses

        C = self.C
        E = self.scenario_tree.E
        U = self.scenario_tree.U

        _E = [-self.reduction.transition[c][e][u][c]
              for c in closed_range(1, C)
              for e in closed_range(1, E)
              for u in closed_range(1, U)]
        totalizer = self.solver.get_totalizer(_E)
        self.reduction = self.reduction._replace(totalizer=totalizer)

        log_debug(f'Done declaring totalizer ({self.number_of_variables-_nv} variables, {self.number_of_clauses-_nc} clauses) in {time.time() - time_start_totalizer:.2f} s')

    def _declare_comparator(self, T):
        log_debug(f'Declaring comparator for T={T}...')
        time_start_comparator = time.time()
        _nc = self.number_of_clauses

        if self._T_defined is not None:
            T_max = self._T_defined
        else:
            T_max = self.C * self.scenario_tree.E * self.scenario_tree.U

        # sum(E) <= T   <=>   sum(E) < T + 1
        for t in reversed(closed_range(T + 1, T_max)):
            self.solver.add_clause(-self.reduction.totalizer[t - 1])  # Note: totalizer is 0-based!

        self._T_defined = T

        log_debug(f'Done declaring comparator ({self.number_of_clauses-_nc} clauses) in {time.time() - time_start_comparator:.2f} s')

    def parse_raw_assignment(self, raw_assignment):
        if raw_assignment is None:
            return None

        log_debug('Building assignment...')
        time_start_assignment = time.time()

        wrapper_int = partial(parse_raw_assignment_int, raw_assignment)
        wrapper_bool = partial(parse_raw_assignment_bool, raw_assignment)
        wrapper_algo = partial(parse_raw_assignment_algo, raw_assignment)

        transition = wrapper_int(self.reduction.transition)
        assignment = self.Assignment(
            color=wrapper_int(self.reduction.color),
            transition=transition,
            output_event=wrapper_int(self.reduction.output_event),
            algorithm_0=wrapper_algo(self.reduction.algorithm_0),
            algorithm_1=wrapper_algo(self.reduction.algorithm_1),
            C=self.C,
            T=sum(transition[c][e][u] != c
                  for c in closed_range(1, self.C)
                  for e in closed_range(1, self.scenario_tree.E)
                  for u in closed_range(1, self.scenario_tree.U)),
        )

        log_debug(f'Done building assignment (T={assignment.T}) in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def build_efsm(self, assignment, *, dump=True):
        if assignment is None:
            return None

        log_br()
        log_info('FullAutomatonTask: building automaton...')
        automaton = EFSM.new_with_full_guards(self.scenario_tree, assignment)

        if dump:
            automaton.dump(self.get_filename_prefix(assignment.T))

        log_success('Full automaton:')
        automaton.pprint()
        automaton.verify()

        return automaton
