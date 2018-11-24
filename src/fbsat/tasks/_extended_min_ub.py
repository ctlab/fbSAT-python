import pathlib
import time
from itertools import count

from . import MinimalExtendedAutomatonTask, MinimalBasicAutomatonTask, Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success, log_warn
from ..utils import json_dump

__all__ = ['MinimalExtendedUBAutomatonTask']


class MinimalExtendedUBAutomatonTask(Task):

    def __init__(self, scenario_tree, *, C=None, K=None, w=None, use_bfs=True, is_distinct=False, is_forbid_or=False,
                 solver_cmd=None, is_incremental=False, is_filesolver=False, outdir=None, path_output=None):
        # w :: None=inf: only UB, 0: first-sat, 1-2: heuristic, >2: very-heuristic

        if path_output is None:
            assert outdir is not None, 'specify either outdir or path_output'
            path_output = pathlib.Path(outdir)
        elif outdir is not None:
            log_warn(f'Ignoring specified outdir <{outdir}>')
        path_output.mkdir(parents=True, exist_ok=True)

        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.w = w
        self.use_bfs = use_bfs
        self.is_distinct = is_distinct
        self.is_forbid_or = is_forbid_or
        self.solver_cmd = solver_cmd
        self.is_incremental = is_incremental
        self.is_filesolver = is_filesolver
        self.path_output = path_output
        self.subtask_config_minbasic = dict(scenario_tree=scenario_tree,
                                              use_bfs=use_bfs,
                                              solver_cmd=solver_cmd,
                                              is_incremental=is_incremental,
                                              is_filesolver=is_filesolver)
        self.subtask_config_minextended = dict(**self.subtask_config_minbasic,
                                               is_distinct=is_distinct,
                                               is_forbid_or=is_forbid_or)
        self.save_params()

        self.path_intermediate = self.path_output / 'intermediate'
        self.path_intermediate.mkdir(parents=True, exist_ok=True)
        self.intermediate_calls = []

    def save_params(self):
        params = dict(C=self.C, K=self.K, w=self.w, use_bfs=self.use_bfs, is_distinct=self.is_distinct,
                      is_forbid_or=self.is_forbid_or, solver_cmd=self.solver_cmd, is_incremental=self.is_incremental,
                      is_filesolver=self.is_filesolver, outdir=str(self.path_output))
        path_params = self.path_output / 'info_params.json'
        json_dump(params, path_params)

    def save_results(self, assignment_or_automaton, time_total):
        A = assignment_or_automaton
        results = dict(C=self.C, K=self.K, w=self.w)
        if A:
            results = dict(**results,
                           Cres=A.C, Kres=A.K, Pres=A.P, Tres=A.T, Nres=A.N,
                           SAT=True, time=time_total)
        else:
            results = dict(**results, SAT=False, time=time_total)

        path_results = self.path_output / 'info_results.json'
        # log_debug(f'Saving results info into <{path_results!s}>...')
        json_dump(results, path_results)

        path_calls = self.path_intermediate / 'info_calls.json'
        # log_debug(f'Saving intermediate calls info into <{path_calls}>...')
        json_dump(self.intermediate_calls, path_calls)

        # TODO: merge intermediate calls from all subtasks... somehow...

    def save_efsm(self, efsm):
        if efsm is None:
            return

        C = efsm.C
        K = efsm.K
        P = efsm.P
        T = efsm.T
        N = efsm.N
        efsm_info = dict(C=C, K=K, P=P, T=T, N=N)

        path_efsm_info = self.path_output / 'info_efsm.json'
        # log_debug(f'Saving EFSM info into <{path_efsm_info!s}>...')
        json_dump(efsm_info, path_efsm_info)

        path_efsm = self.path_output / f'efsm_C{C}_K{K}_P{P}_T{T}_N{N}'
        # log_debug(f'Dumping EFSM into <{path_efsm!s}>...')
        efsm.dump(str(path_efsm))

    def create_basic_min_subtask_call(self):
        call_name = 'basic-min'
        path_call = self.path_intermediate / f'{len(self.intermediate_calls):0>4}_{call_name}'
        path_call.mkdir(parents=True, exist_ok=True)

        config = dict(**self.subtask_config_minbasic, path_output=path_call)
        task = MinimalBasicAutomatonTask(**config)

        def subtask_call():
            time_start_call = time.time()
            assignment = task.run(fast=True)
            time_total_call = time.time() - time_start_call

            if assignment:
                call_results = dict(C=assignment.C, K=assignment.K, T=assignment.T,
                                    SAT=True, time=time_total_call)
            else:
                call_results = dict(SAT=False, time=time_total_call)

            call_info = dict(call=call_name,
                             params=dict(),
                             results=call_results)
            self.intermediate_calls.append(call_info)

            return assignment

        return subtask_call

    def create_extended_min_subtask_call(self, C, K, P, N_init):
        call_name = 'extended-min'
        call_params = dict(C=C, K=K, P=P, N_init=N_init)
        path_call = self.path_intermediate / f'{len(self.intermediate_calls):0>4}_{call_name}_C{C}_K{K}_P{P}_Ninit{N_init}'
        path_call.mkdir(parents=True, exist_ok=True)

        config = dict(**self.subtask_config_minextended, **call_params, path_output=path_call)
        task = MinimalExtendedAutomatonTask(**config)

        def subtask_call():
            time_start_call = time.time()
            assignment = task.run(fast=True)
            time_total_call = time.time() - time_start_call

            if assignment:
                call_results = dict(C=assignment.C, K=assignment.K, P=assignment.P, T=assignment.T, N=assignment.N,
                                    SAT=True, time=time_total_call)
            else:
                call_results = dict(SAT=False, time=time_total_call)

            call_info = dict(call=call_name,
                             params=call_params,
                             results=call_results)
            self.intermediate_calls.append(call_info)

            return assignment

        return subtask_call

    def run(self, *, fast=False):
        log_debug(f'MinimalExtendedUBAutomatonTask: running...')
        time_start_run = time.time()

        if self.C is None:
            log_debug('MinimalExtendedUBAutomatonTask: searching for minimal C...')
            task_call = self.create_basic_min_subtask_call()
            assignment = task_call()
            C = assignment.C
            T_min = assignment.T
            log_debug(f'MinimalExtendedUBAutomatonTask: found minimal C={C}')
        else:
            raise NotImplementedError('T_min is unknown with specified C')
            # C = self.C
            # log_debug(f'MinimalExtendedUBAutomatonTask: using specified C={C}')

        if self.K is None:
            K = C
            # log_debug(f'MinimalExtendedUBAutomatonTask: using K=C={K}')
        else:
            K = self.K
            # log_debug(f'MinimalExtendedUBAutomatonTask: using specified K={K}')

        log_debug('MinimalExtendedUBAutomatonTask: searching for P...')
        best = None
        prev = None
        P_low = None
        # for P in chain([1, 3, 5, 7, 9], count(10)):
        for P in count(1):
            log_br()
            log_info(f'Trying P={P}...')

            if best and P > (best.N - T_min):
                log_warn(f'Upper bound reached on P={P}, N_best={best.N}, T_min={T_min}')
                break

            # Note: w=None == w=inf == no width limit,
            #       but w=0 is okay 1-"width" limit,
            #       because w is simply a difference between Pcurrent and Plow
            if P_low is not None and self.w is not None and (P - P_low) > self.w:
                log_warn(f'Maximum width reached! P={P}, P_low={P_low}, w={self.w}')
                break

            if best:
                N_new = best.N - 1
            else:
                N_new = None
            task_call = self.create_extended_min_subtask_call(C, K, P, N_new)
            assignment = task_call()

            if assignment:
                if best is None or assignment.N < best.N:
                    best = assignment
                if prev is None or assignment.N != prev.N:
                    P_low = P
                prev = assignment
        else:
            log_error('MinimalExtendedUBAutomatonTask: P was not found')

        if fast:
            time_total_run = time.time() - time_start_run
            self.save_results(best, time_total_run)
            log_debug(f'MinimalExtendedUBAutomatonTask: done in {time_total_run:.2f} s')
            return best
        else:
            automaton = self.build_efsm(best)
            time_total_run = time.time() - time_start_run
            self.save_efsm(automaton)
            self.save_results(automaton, time_total_run)
            log_debug(f'MinimalExtendedUBAutomatonTask: done in {time_total_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Minimal extended automaton has {automaton.C} states,'
                            f' {automaton.T} transitions and {automaton.N} nodes')
            else:
                log_error(f'Minimal extended automaton was not found')
            return automaton

    def build_efsm(self, assignment):
        if assignment is None:
            return None

        log_br()
        log_info('MinimalExtendedUBAutomatonTask: building automaton...')
        automaton = EFSM.new_with_parse_trees(self.scenario_tree, assignment)

        log_success('Minimal extended automaton:')
        automaton.pprint()
        automaton.verify(self.scenario_tree)

        return automaton
