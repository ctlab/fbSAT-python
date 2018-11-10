import time
import pathlib

from . import CompleteAutomatonTask, MinimalPartialAutomatonTask, Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success, log_warn
from ..utils import json_dump

__all__ = ['MinimalCompleteAutomatonTask']


class MinimalCompleteAutomatonTask(Task):

    def __init__(self, scenario_tree, *, C=None, K=None, P, N_init=None, use_bfs=True, is_distinct=False, is_forbid_or=False, solver_cmd=None, is_incremental=False, is_filesolver=False, outdir=None, path_output=None):
        assert P is not None

        if path_output is None:
            assert outdir is not None, 'specify either outdir or path_output'
            path_output = pathlib.Path(outdir)
        elif outdir is not None:
            log_warn(f'Ignoring specified outdir <{outdir}>')

        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.P = P
        self.N_init = N_init
        self.use_bfs = use_bfs
        self.is_distinct = is_distinct
        self.is_forbid_or = is_forbid_or
        self.is_incremental = is_incremental
        self.is_filesolver = is_filesolver
        self.path_output = path_output
        self.subtask_config_minpartial = dict(scenario_tree=scenario_tree,
                                              use_bfs=use_bfs,
                                              solver_cmd=solver_cmd,
                                              is_incremental=is_incremental,
                                              is_filesolver=is_filesolver)
        self.subtask_config_complete = dict(**self.subtask_config_minpartial,
                                            is_distinct=is_distinct,
                                            is_forbid_or=is_forbid_or)

        self.params = dict(C=C, K=K, N_init=N_init, use_bfs=use_bfs, is_distinct=is_distinct, is_forbid_or=is_forbid_or, solver_cmd=solver_cmd, is_incremental=is_incremental, is_filesolver=is_filesolver, outdir=str(path_output))
        self.save_params()

        self.path_intermediate = self.path_output / 'intermediate'
        self.path_intermediate.mkdir(parents=True)
        self.intermediate_calls = []

    def save_params(self):
        path_params = self.path_output / 'info_params.json'
        json_dump(self.params, path_params)

    def save_results(self, assignment_or_automaton, time_total):
        A = assignment_or_automaton
        results = dict(C=self.C, K=self.K, P=self.P, N_init=self.N_init)
        if A:
            results = dict(**results,
                           Cres=A.C, Kres=A.K, Pres=A.P, Tres=A.T, Nres=A.N,
                           SAT=True, time=time_total)
        else:
            results = dict(**results, SAT=False, time=time_total)

        path_results = self.path_output / 'info_results.json'
        log_debug(f'Saving results info into <{path_results!s}>...')
        json_dump(results, path_results)

        path_calls = self.path_intermediate / 'info_calls.json'
        log_debug(f'Saving intermediate calls info into <{path_calls}>...')
        json_dump(self.intermediate_calls, path_calls)

        # TODO: merge intermediate calls from all subtasks... somehow...

    def save_efsm(self, efsm):
        if efsm is None:
            return

        C = efsm.C
        K = efsm.K
        T = efsm.T
        efsm_info = dict(C=C, K=K, T=T)

        path_efsm_info = self.path_output / 'info_efsm.json'
        log_debug(f'Saving EFSM info into <{path_efsm_info!s}>...')
        json_dump(efsm_info, path_efsm_info)

        path_efsm = self.path_output / f'efsm_C{C}_K{K}_T{T}'
        log_debug(f'Dumping EFSM into <{path_efsm!s}>...')
        efsm.dump(str(path_efsm))

    def create_basic_min_subtask_call(self):
        call_name = 'basic-min-onlyC'
        path_call = self.path_intermediate / f'{len(self.intermediate_calls):0>4}_{call_name}'
        path_call.mkdir(parents=True)

        config = dict(**self.subtask_config_minpartial, path_output=path_call)
        task = MinimalPartialAutomatonTask(**config)

        def subtask_call():
            time_start_call = time.time()
            assignment = task.run(fast=True, only_C=True)
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

    def create_extended_subtask_call(self, C, K, P):
        call_name = 'extended'
        call_params = dict(C=C, K=K, P=P)
        path_call = self.path_intermediate / f'{len(self.intermediate_calls):0>4}_{call_name}_C{C}_K{K}_P{P}'
        path_call.mkdir(parents=True)

        config = dict(**self.subtask_config_complete, **call_params, path_output=path_call)
        task = CompleteAutomatonTask(**config)

        def subtask_call(N):
            time_start_call = time.time()
            assignment = task.run(N, fast=True, finalize=False)
            time_total_call = time.time() - time_start_call

            if assignment:
                call_results = dict(C=assignment.C, K=assignment.K, P=assignment.P,
                                    T=assignment.T, N=assignment.N,
                                    SAT=True, time=time_total_call)
            else:
                call_results = dict(SAT=False, time=time_total_call)

            call_info = dict(call=call_name,
                             params=dict(**call_params, N=N),
                             results=call_results)
            self.intermediate_calls.append(call_info)

            return assignment

        def subtask_finalize():
            task.finalize()

        return subtask_call, subtask_finalize

    def run(self, *, fast=False):
        # TODO: rename 'fast' to 'only_assignment'
        log_debug(f'MinimalCompleteAutomatonTask: running...')
        time_start_run = time.time()
        best = None

        if self.C is None:
            log_debug('MinimalCompleteAutomatonTask: searching for minimal C...')
            task_call = self.create_basic_min_subtask_call()
            assignment = task_call()
            C = assignment.C
            log_debug(f'MinimalCompleteAutomatonTask: found minimal C={C}')
        else:
            C = self.C
            # log_debug(f'MinimalCompleteAutomatonTask: using specified C={C}')

        if self.K is None:
            K = C
            # log_debug(f'MinimalCompleteAutomatonTask: using K=C={K}')
        else:
            K = self.K
            # log_debug(f'MinimalCompleteAutomatonTask: using specified K={K}')

        task_call, task_finalize = self.create_extended_subtask_call(C, K, self.P)
        assignment = task_call(self.N_init)

        while assignment:
            log_debug('MinimalCompleteAutomatonTask: searching for minimal N...')
            best = assignment
            N = best.N - 1
            log_br()
            log_info(f'Trying N = {N}...')
            assignment = task_call(N)

        task_finalize()

        if fast:
            time_total_run = time.time() - time_start_run
            self.save_results(best, time_total_run)
            log_debug(f'MinimalCompleteAutomatonTask: done in {time_total_run:.2f} s')
            return best
        else:
            automaton = self.build_efsm(best)
            time_total_run = time.time() - time_start_run
            self.save_efsm(automaton)
            self.save_results(automaton, time_total_run)
            log_debug(f'MinimalCompleteAutomatonTask: done in {time_total_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Minimal complete automaton has {automaton.C} states, {automaton.T} transitions and {automaton.N} nodes')
            else:
                log_error(f'Minimal complete automaton was not found')
            return automaton

    def build_efsm(self, assignment):
        if assignment is None:
            return None

        log_br()
        log_info('MinimalCompleteAutomatonTask: building automaton...')
        automaton = EFSM.new_with_parse_trees(self.scenario_tree, assignment)

        log_success('Minimal complete automaton:')
        automaton.pprint()
        automaton.verify(self.scenario_tree)

        return automaton
