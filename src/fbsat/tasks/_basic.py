import pathlib
import time
from collections import namedtuple
from functools import partial

from . import Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success, log_warn
from ..solver import FileSolver, IncrementalSolver, StreamSolver
from ..utils import (auto_finalize, closed_range, json_dump, parse_raw_assignment_algo, parse_raw_assignment_bool,
                     parse_raw_assignment_int)

__all__ = ['BasicAutomatonTask']

VARIABLES = 'color transition output_event algorithm_0 algorithm_1 first_fired not_fired'


class BasicAutomatonTask(Task):
    Reduction = namedtuple('Reduction', VARIABLES + ' totalizer')
    Assignment = namedtuple('Assignment', VARIABLES + ' C K T')

    def __init__(self, scenario_tree, *, C, K=None, use_bfs=True, is_distinct=False, solver_cmd,
                 is_incremental=False, is_filesolver=False, outdir=None, path_output=None):
        assert C is not None

        if K is None:
            K = C

        if path_output is None:
            assert outdir is not None, 'specify either outdir or path_output'
            path_output = pathlib.Path(outdir)
        elif outdir is not None:
            log_warn(f'Ignoring specified outdir <{outdir}>')
        path_output.mkdir(parents=True, exist_ok=True)

        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.use_bfs = use_bfs
        self.is_distinct = is_distinct
        self.solver_cmd = solver_cmd
        self.is_incremental = is_incremental
        self.is_filesolver = is_filesolver
        self.path_output = path_output
        self.save_params()

        self.solver_config = dict(cmd=solver_cmd)
        self._new_solver()

    def save_params(self):
        params = dict(C=self.C, K=self.K, use_bfs=self.use_bfs, is_distinct=self.is_distinct,
                      solver_cmd=self.solver_cmd, is_incremental=self.is_incremental,
                      is_filesolver=self.is_filesolver, outdir=str(self.path_output))
        path_params = self.path_output / 'info_params.json'
        json_dump(params, path_params)

    def save_results(self, assignment_or_automaton, T, time_total, path_run):
        A = assignment_or_automaton
        results = dict(C=self.C, K=self.K, T=T)
        if A:
            results = dict(**results, Cres=A.C, Kres=A.K, Tres=A.T, SAT=True, time=time_total)
        else:
            results = dict(**results, SAT=False, time=time_total)
        path_results = path_run / 'info_results.json'
        # log_debug(f'Saving results info into <{path_results!s}>...')
        json_dump(results, path_results)

    @staticmethod
    def save_efsm(efsm, path_run):
        if efsm is None:
            return

        C = efsm.C
        K = efsm.K
        T = efsm.T
        efsm_info = dict(C=C, K=K, T=T)

        path_efsm_info = path_run / 'info_efsm.json'
        # log_debug(f'Saving EFSM info into <{path_efsm_info!s}>...')
        json_dump(efsm_info, path_efsm_info)

        path_efsm = path_run / f'efsm_C{C}_K{K}_T{T}'
        # log_debug(f'Dumping EFSM into <{path_efsm!s}>...')
        efsm.dump(str(path_efsm))

    def _new_solver(self):
        self._is_base_declared = False
        self._is_totalizer_declared = False
        self._T_defined = None

        if self.is_incremental:
            self.solver = IncrementalSolver(self.solver_cmd)
        elif self.is_filesolver:
            # TODO: deprecated
            self.solver = FileSolver(self.solver_cmd, filename_prefix=self.get_filename_prefix())
        else:
            self.solver = StreamSolver(self.solver_cmd)

    @property
    def number_of_variables(self):
        return self.solver.number_of_variables

    @property
    def number_of_clauses(self):
        return self.solver.number_of_clauses

    @auto_finalize
    def run(self, T=None, *, multirun, fast=False):
        # TODO: rename `fast` to `only_assignment`
        log_debug(f'BasicAutomatonTask: running for T={T}...')
        time_start_run = time.time()

        if multirun:
            path_run = self.path_output / f'run_T{T}'
        else:
            path_run = self.path_output
        path_run.mkdir(parents=True, exist_ok=True)

        self._declare_base_reduction()
        if T is not None:
            self._declare_totalizer()
            self._declare_comparator(T)

        raw_assignment = self.solver.solve()
        assignment = self.parse_raw_assignment(raw_assignment)

        if fast:
            time_total_run = time.time() - time_start_run
            self.save_results(assignment, T, time_total_run, path_run)
            log_debug(f'BasicAutomatonTask: done in {time_total_run:.2f} s')
            return assignment
        else:
            automaton = self.build_efsm(assignment)
            time_total_run = time.time() - time_start_run
            self.save_efsm(automaton, path_run)
            self.save_results(automaton, T, time_total_run, path_run)
            log_debug(f'BasicAutomatonTask: done in {time_total_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Basic automaton has {automaton.number_of_states} states'
                            f' and {automaton.number_of_transitions} transitions')
            else:
                log_error(f'Basic automaton was not found')
            return automaton

    def finalize(self):
        # log_debug('BasicAutomatonTask: finalizing...')
        if self.is_incremental:
            self.solver.process.kill()

    def _declare_base_reduction(self):
        if self._is_base_declared:
            return
        self._is_base_declared = True

        C = self.C
        K = self.K
        assert self.number_of_variables == 0
        assert self.number_of_clauses == 0

        # log_debug(f'Declaring base reduction for C={C}, K={K}...')
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
        Y = tree.Y

        # log_debug(f'V = {V}')
        # log_debug(f'E = {E}')
        # log_debug(f'O = {O}')
        # log_debug(f'X = {X}')
        # log_debug(f'Z = {Z}')
        # log_debug(f'U = {U}')
        # log_debug(f'Y = {Y}')

        comment = self.solver.comment
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
        first_fired = declare_array(C, E, U, K)
        not_fired = declare_array(C, E, U, K)
        # bfs variables
        if self.use_bfs:
            bfs_transition = declare_array(C, C)
            bfs_parent = declare_array(C, C)

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

        so_far_state = [self.number_of_clauses, time.time()]

        def so_far():
            now = self.number_of_clauses
            time_now = time.time()
            ans = now - so_far_state[0]
            timing = time_now - so_far_state[1]
            so_far_state[0] = now
            so_far_state[1] = time_now
            return f'{ans} in {timing:.2f} s'

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('1. Color constraints')
        comment('1.0. ALO/AMO(color)')
        for v in closed_range(1, V):
            ALO(color[v])
            AMO(color[v])

        comment('1.1. Start vertex corresponds to start state')
        add_clause(color[1][1])

        comment('1.2. Color definition')
        for c in closed_range(1, C):
            for v in tree.V_passive:
                iff(color[v][c], color[tree.parent[v]][c])

        # log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

        comment('2. Transition constraints')
        comment('2.0. ALO/AMO(transition)')
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    ALO(transition[c][e][k])
                    AMO(transition[c][e][k])

        comment('2.1. (transition + first_fired definitions)')
        for i in closed_range(1, C):
            for e in closed_range(1, E):
                for j in closed_range(1, C):
                    for u in closed_range(1, U):
                        # OR_k( transition[i,e,k,j] & first_fired[i,e,u,k] ) <=> ...
                        # ... <=> OR_{v|active,tie(v)=e,tin(v)=u}( color[tp(v),i] & color[v,j] )
                        leftright = new_variable()

                        lhs = []
                        for k in closed_range(1, K):
                            # aux <-> transition[i,e,k,j] & first_fired[i,e,u,k]
                            aux = new_variable()
                            iff_and(aux, (transition[i][e][k][j], first_fired[i][e][u][k]))
                            lhs.append(aux)
                        iff_or(leftright, lhs)

                        rhs = []
                        for v in tree.V_active_eu[e][u]:
                            # aux <-> color[tp(v),i] & color[v,j]
                            aux = new_variable()
                            p = tree.parent[v]
                            iff_and(aux, (color[p][i], color[v][j]))
                            rhs.append(aux)
                        iff_or(leftright, rhs)

        comment('2.2. Null-transitions are last')
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K - 1):
                    imply(transition[c][e][k][0], transition[c][e][k + 1][0])

        # log_debug(f'2. Clauses: {so_far()}', symbol='STAT')

        comment('3. Firing constraints')
        comment('3.0. only AMO(first_fired)')
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    AMO(first_fired[c][e][u])

        comment('3.1. (not_fired definition)')
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    # not_fired[c,e,u,K] <=> OR_{v|passive,tie(v)=e,tin(v)=u}(color[v,c])
                    rhs = []
                    for v in tree.V_passive_eu[e][u]:
                        rhs.append(color[v][c])  # passive: color[v] == color[tp(v)] == color[tpa(v)]
                    iff_or(not_fired[c][e][u][K], rhs)

        comment('3.2. not_fired extension')
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    for k in closed_range(2, K):
                        # nf_k => nf_{k-1}
                        imply(not_fired[c][e][u][k], not_fired[c][e][u][k - 1])
                    for k in closed_range(1, K - 1):
                        # ~nf_k => ~nf_{k+1}
                        imply(-not_fired[c][e][u][k], -not_fired[c][e][u][k + 1])

                    # Trial: (AND_k(~ff_k) & ~nf_K) => ~nf_1  # aux = new_variable()  # rhs = []  # for k in closed_range(1, K):  #     rhs.append(-first_fired[c][e][u][k])  # rhs.append(-not_fired[c][e][u][K])  # iff_and(aux, rhs)  # imply(aux, -not_fired[c][e][u][1])

        comment('3.3. first_fired and not_fired interaction')
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    for k in closed_range(1, K):
                        # ~(ff & nf)
                        add_clause(-first_fired[c][e][u][k], -not_fired[c][e][u][k])
                    for k in closed_range(2, K):
                        # ff_k => nf_{k-1}
                        imply(first_fired[c][e][u][k], not_fired[c][e][u][k - 1])

        # log_debug(f'3. Clauses: {so_far()}', symbol='STAT')

        comment('4. Output event constraints')
        comment('4.0. ALO/AMO(output_event)')
        for c in closed_range(1, C):
            ALO(output_event[c])
            AMO(output_event[c])

        comment('4.1. Start state does INITO (root`s output event)')
        add_clause(output_event[1][tree.output_event[1]])

        comment('4.2. Output event is the same as in the tree')
        for i in closed_range(1, C):
            for j in closed_range(1, C):
                # OR_{e,k}(transition[i,e,k,j]) <=> ...
                # ... <=> OR_{v|active}( color[tp(v),i] & color[v,j] & output_event[j,toe(v)] )
                leftright = new_variable()

                lhs = []
                for e in closed_range(1, E):
                    for k in closed_range(1, K):
                        lhs.append(transition[i][e][k][j])
                iff_or(leftright, lhs)

                rhs = []
                for v in tree.V_active:
                    # aux <-> color[tp(v),i] & color[v,j] & output_event[j,toe(v)]
                    aux = new_variable()
                    p = tree.parent[v]
                    o = tree.output_event[v]
                    iff_and(aux, (color[p][i], color[v][j], output_event[j][o]))
                    rhs.append(aux)
                iff_or(leftright, rhs)

        # log_debug(f'4. Clauses: {so_far()}', symbol='STAT')

        comment('5. Algorithm constraints')
        comment('5.1. Start state does nothing')
        for z in closed_range(1, Z):
            add_clause(-algorithm_0[1][z])
            add_clause(algorithm_1[1][z])

        comment('5.2. Algorithms definition')
        for v in tree.V_active:
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

        # log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        if self.use_bfs:
            comment('6. BFS constraints')
            comment('6.1. F_t')
            for i in closed_range(1, C):
                for j in closed_range(1, C):
                    # t_ij <=> OR_{e,k}(transition_iekj)
                    rhs = []
                    for e in closed_range(1, E):
                        for k in closed_range(1, K):
                            rhs.append(transition[i][e][k][j])
                    iff_or(bfs_transition[i][j], rhs)

            comment('6.2. F_p')
            for i in closed_range(1, C):
                for j in closed_range(1, i):  # to avoid ambiguous unused variable
                    add_clause(-bfs_parent[j][i])
                for j in closed_range(i + 1, C):
                    # p_ji <=> t_ij & AND_[k<i](~t_kj)
                    rhs = [bfs_transition[i][j]]
                    for k in closed_range(1, i - 1):
                        rhs.append(-bfs_transition[k][j])
                    iff_and(bfs_parent[j][i], rhs)

            comment('6.3. F_ALO(p)')
            for j in closed_range(2, C):
                add_clause(*[bfs_parent[j][i] for i in closed_range(1, j - 1)])

            comment('6.4. F_BFS(p)')
            for k in closed_range(1, C):
                for i in closed_range(k + 1, C):
                    for j in closed_range(i + 1, C - 1):
                        # p_ji => ~p_{j+1,k}
                        imply(bfs_parent[j][i], -bfs_parent[j + 1][k])

            # log_debug(f'6. Clauses: {so_far()}', symbol='STAT')

        comment('A. AD-HOCs')
        comment('A.1. Distinct transitions')
        if self.is_distinct:
            for i in closed_range(1, C):
                for e in closed_range(1, E):
                    for k in closed_range(1, K):
                        # transition[i,e,k,j] => AND_{k_!=k}(~transition[i,e,k_,j])
                        for j in closed_range(1, C):
                            for k_ in closed_range(k + 1, K):
                                imply(transition[i][e][k][j], -transition[i][e][k_][j])

        # log_debug(f'A. Clauses: {so_far()}', symbol='STAT')

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.reduction = self.Reduction(color=color, transition=transition, output_event=output_event,
                                        algorithm_0=algorithm_0, algorithm_1=algorithm_1,
                                        first_fired=first_fired, not_fired=not_fired, totalizer=None)

        # log_debug(f'Done declaring base reduction ({self.number_of_variables} variables, {self.number_of_clauses} clauses) in {time.time() - time_start_base:.2f} s')

    def _declare_totalizer(self):
        if self._is_totalizer_declared:
            return
        self._is_totalizer_declared = True

        # log_debug('Declaring totalizer...')
        # time_start_totalizer = time.time()
        _nv = self.number_of_variables
        _nc = self.number_of_clauses

        _E = [-self.reduction.transition[c][e][k][0] for c in closed_range(1, self.C) for e in
              closed_range(1, self.scenario_tree.E) for k in closed_range(1, self.K)]
        totalizer = self.solver.get_totalizer(_E)
        self.reduction = self.reduction._replace(totalizer=totalizer)

        # log_debug(f'Done declaring totalizer ({self.number_of_variables-_nv} variables, {self.number_of_clauses-_nc} clauses) in {time.time() - time_start_totalizer:.2f} s')

    def _declare_comparator(self, T):
        # log_debug(f'Declaring comparator for T={T}...')
        time_start_comparator = time.time()
        _nc = self.number_of_clauses

        if self._T_defined is not None:
            T_max = self._T_defined
        else:
            T_max = self.C * self.scenario_tree.E * self.K

        # sum(E) <= T   <=>   sum(E) < T + 1
        for t in reversed(closed_range(T + 1, T_max)):
            self.solver.add_clause(-self.reduction.totalizer[t - 1])  # Note: totalizer is 0-based!

        self._T_defined = T

        # log_debug(f'Done declaring comparator ({self.number_of_clauses-_nc} clauses) in {time.time() - time_start_comparator:.2f} s')

    def parse_raw_assignment(self, raw_assignment):
        if raw_assignment is None:
            return None

        # log_debug('Building assignment...')
        # time_start_assignment = time.time()

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
                first_fired=wrapper_bool(self.reduction.first_fired),
                not_fired=wrapper_bool(self.reduction.not_fired),
                C=self.C,
                K=self.K,
                T=sum(transition[c][e][k] != 0
                      for c in closed_range(1, self.C)
                      for e in closed_range(1, self.scenario_tree.E)
                      for k in closed_range(1, self.K)),
        )

        # ==================
        # Ks = []
        # for c in closed_range(1, self.C):
        #     for e in closed_range(1, self.scenario_tree.E):
        #         Ks.append(sum(transition[c][e][k] != 0 for k in closed_range(1, self.K)))
        # log_debug(f'max(K) = {max(Ks)}')
        # ==================

        # log_debug(f'Done building assignment (T={assignment.T}) in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def build_efsm(self, assignment):
        if assignment is None:
            return None

        log_br()
        log_info('BasicAutomatonTask: building automaton...')
        automaton = EFSM.new_with_truth_tables(self.scenario_tree, assignment)

        log_success('Basic automaton:')
        automaton.pprint()
        automaton.verify(self.scenario_tree)

        return automaton
