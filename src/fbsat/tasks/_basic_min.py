import itertools
import pathlib
import time

from . import BasicAutomatonTask, Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success, log_warn
from ..utils import json_dump

__all__ = ['MinimalBasicAutomatonTask']


class MinimalBasicAutomatonTask(Task):

    def __init__(self, scenario_tree, *, C=None, K=None, T_init=None, use_bfs=True, is_distinct=False, solver_cmd,
                 is_incremental=False, is_filesolver=False, outdir=None, path_output=None):
        assert not (C is None and K is not None), 'do not specify only K'

        if path_output is None:
            assert outdir is not None, 'specify either outdir or path_output'
            path_output = pathlib.Path(outdir)
        elif outdir is not None:
            log_warn(f'Ignoring specified outdir <{outdir}>')
        path_output.mkdir(parents=True, exist_ok=True)

        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.T_init = T_init
        self.use_bfs = use_bfs
        self.is_distinct = is_distinct
        self.solver_cmd = solver_cmd
        self.is_incremental = is_incremental
        self.is_filesolver = is_filesolver
        self.path_output = path_output
        self.subtask_config_basic = dict(scenario_tree=scenario_tree,
                                           use_bfs=use_bfs,
                                           is_distinct=is_distinct,
                                           solver_cmd=solver_cmd,
                                           is_incremental=is_incremental,
                                           is_filesolver=is_filesolver)
        self.save_params()

        self.path_intermediate = self.path_output / 'intermediate'
        self.path_intermediate.mkdir(parents=True, exist_ok=True)
        self.intermediate_calls = []

    def save_params(self):
        params = dict(C=self.C, K=self.K, T_init=self.T_init, use_bfs=self.use_bfs, is_distinct=self.is_distinct,
                      solver_cmd=self.solver_cmd, is_incremental=self.is_incremental,
                      is_filesolver=self.is_filesolver, outdir=str(self.path_output))
        path_params = self.path_output / 'info_params.json'
        json_dump(params, path_params)

    def save_results(self, assignment_or_automaton, time_total):
        A = assignment_or_automaton
        results = dict(C=self.C, K=self.K, T_init=self.T_init)
        if A:
            results = dict(**results, Cres=A.C, Kres=A.K, Tres=A.T, SAT=True, time=time_total)
        else:
            results = dict(**results, SAT=False, time=time_total)

        path_results = self.path_output / 'info_results.json'
        # log_debug(f'Saving results info into <{path_results!s}>...')
        json_dump(results, path_results)

        path_calls = self.path_intermediate / 'info_calls.json'
        # log_debug(f'Saving intermediate calls info into <{path_calls}>...')
        json_dump(self.intermediate_calls, path_calls)

    def save_efsm(self, efsm):
        if efsm is None:
            return

        C = efsm.C
        K = efsm.K
        T = efsm.T
        efsm_info = dict(C=C, K=K, T=T)

        path_efsm_info = self.path_output / 'info_efsm.json'
        # log_debug(f'Saving EFSM info into <{path_efsm_info!s}>...')
        json_dump(efsm_info, path_efsm_info)

        path_efsm = self.path_output / f'efsm_C{C}_K{K}_T{T}'
        # log_debug(f'Dumping EFSM into <{path_efsm!s}>...')
        efsm.dump(str(path_efsm))

    def create_basic_subtask_call(self, C, K):
        call_name = 'basic'
        call_params = dict(C=C, K=K)
        path_call = self.path_intermediate / f'{len(self.intermediate_calls):0>4}_{call_name}_C{C}_K{K}'
        path_call.mkdir(parents=True, exist_ok=True)

        config = dict(**self.subtask_config_basic, **call_params, path_output=path_call)
        task = BasicAutomatonTask(**config)

        def subtask_call(T):
            time_start_call = time.time()
            assignment = task.run(T, fast=True, finalize=False)
            time_total_call = time.time() - time_start_call

            if assignment:
                call_results = dict(C=assignment.C, K=assignment.K, T=assignment.T,
                                    SAT=True, time=time_total_call)
            else:
                call_results = dict(SAT=False, time=time_total_call)

            call_info = dict(call=call_name,
                             params=dict(**call_params, T=T),
                             results=call_results)
            self.intermediate_calls.append(call_info)

            return assignment

        def subtask_finalize():
            task.finalize()

        return subtask_call, subtask_finalize

    def run(self, *, fast=False, only_C=False):
        # TODO: rename `fast` to `only_assignment`
        log_debug(f'MinimalBasicAutomatonTask: running...')
        time_start_run = time.time()

        if self.C is None:
            log_debug('MinimalBasicAutomatonTask: searching for minimal C...')
            best = None
            for C in itertools.islice(itertools.count(1), 15):
                log_br()
                log_info(f'Trying C = {C}...')

                K = C
                task_call, task_finalize = self.create_basic_subtask_call(C, K)
                assignment = task_call(self.T_init)

                if assignment:
                    best = assignment
                    log_debug(f'MinimalBasicAutomatonTask: found minimal C={C}')
                    break
                else:
                    task_finalize()
            else:
                log_error('MinimalBasicAutomatonTask: minimal C was not found')
        else:
            log_debug(f'MinimalBasicAutomatonTask: using specified C={C}, K={K}')
            C = self.C
            if self.K is None:
                K = self.C
            else:
                K = self.K

            task_call, task_finalize = self.create_basic_subtask_call(C, K)
            best = task_call(self.T_init)

        if not only_C and best:
            log_debug('MinimalBasicAutomatonTask: searching for minimal T...')
            assignment = best
            while True:
                best = assignment
                T = best.T - 1
                log_br()
                log_info(f'Trying T={T}...')
                assignment = task_call(T)
                if assignment is None:
                    log_debug(f'MinimalBasicAutomatonTask: found minimal T={best.T}')
                    break

        task_finalize()

        if fast:
            time_total_run = time.time() - time_start_run
            self.save_results(best, time_total_run)
            log_debug(f'MinimalBasicAutomatonTask: done in {time_total_run:.2f} s')
            return best
        else:
            automaton = self.build_efsm(best)
            time_total_run = time.time() - time_start_run
            self.save_efsm(automaton)
            self.save_results(automaton, time_total_run)
            log_debug(f'MinimalBasicAutomatonTask: done in {time_total_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Minimal basic automaton has {automaton.C} states and {automaton.T} transitions')
            else:
                log_error(f'Minimal basic automaton was not found')
            return automaton

    def build_efsm(self, assignment):
        if assignment is None:
            return None

        log_br()
        log_info('MinimalBasicAutomatonTask: building automaton...')
        automaton = EFSM.new_with_truth_tables(self.scenario_tree, assignment)

        log_success('Minimal basic automaton:')
        automaton.pprint()
        automaton.verify(self.scenario_tree)

        return automaton
