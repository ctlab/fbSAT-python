import itertools
import os
import time
import pathlib

from . import PartialAutomatonTask, Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success
from ..utils import json_dump

__all__ = ['MinimalPartialAutomatonTask']


class MinimalPartialAutomatonTask(Task):

    def __init__(self, scenario_tree, *, C=None, K=None, T_init=None, use_bfs=True, is_distinct=False, solver_cmd=None, is_incremental=False, is_filesolver=False, outdir='', path_output=None):
        assert not (C is None and K is not None)

        if path_output is None:
            if outdir == '':
                raise ValueError('specify either non-empty outdir or path_output')
            path_output = pathlib.Path(outdir)

        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.T_init = T_init
        self.outdir = outdir
        self.subtask_config = dict(scenario_tree=scenario_tree,
                                   use_bfs=use_bfs,
                                   is_distinct=is_distinct,
                                   solver_cmd=solver_cmd,
                                   is_incremental=is_incremental,
                                   is_filesolver=is_filesolver,
                                   outdir=outdir)

        self.path_output = path_output
        self.save_params(dict(C=C, K=K, T_init=T_init, use_bfs=use_bfs, is_distinct=is_distinct, solver_cmd=solver_cmd, is_incremental=is_incremental, is_filesolver=is_filesolver, outdir=outdir))

        self.path_intermediate = self.path_output / 'intermediate'
        self.path_intermediate.mkdir(parents=True)
        self.intermediate_calls = []

    def save_params(self, params):
        path_params = self.path_output / 'info_params.json'
        json_dump(params, path_params)

    def save_results(self, results):
        path_results = self.path_output / 'info_results.json'
        log_debug(f'Saving results info into <{path_results!s}>...')
        json_dump(results, path_results)

    def save_efsm(self, efsm):
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

    def get_stem(self, C, K, T):
        return f'minimal_partial_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_T{T}'

    def get_filename_prefix(self, C, K, T):
        return os.path.join(self.outdir, self.get_stem(C, K, T))

    def create_subtask(self, C, K):
        call = dict(call='basic', C=C, K=K)
        path_call = self.path_intermediate / f'{len(self.intermediate_calls):0>4}_basic_C{C}_K{K}'
        path_call.mkdir(parents=True)
        self.intermediate_calls.append(call)

        config = dict(**self.subtask_config,
                      C=C, K=K, path_output=path_call)
        task = PartialAutomatonTask(**config)
        return task

    def run(self, *, fast=False, only_C=False):
        # TODO: rename `fast` to `only_assignment`

        log_debug(f'MinimalPartialAutomatonTask: running...')
        time_start_run = time.time()
        best = None

        if self.C is None:
            log_debug('MinimalPartialAutomatonTask: searching for minimal C...')
            for C in itertools.islice(itertools.count(1), 15):
                log_br()
                log_info(f'Trying C = {C}...')

                K = C
                task = self.create_subtask(C, K)
                assignment = task.run(self.T_init, fast=True, finalize=False)

                if assignment:
                    best = assignment
                    log_debug(f'MinimalPartialAutomatonTask: found minimal C={C}')
                    break
                else:
                    task.finalize()
            else:
                log_error('MinimalPartialAutomatonTask: minimal C was not found')
        else:
            log_debug(f'MinimalPartialAutomatonTask: using specified C={self.C}')

            C = self.C
            if self.K is None:
                K = self.C
            else:
                K = self.K
            task = self.create_subtask(C, K)
            best = task.run(self.T_init, fast=True, finalize=False)

        if not only_C and best:
            log_debug('MinimalPartialAutomatonTask: searching for minimal T...')
            assignment = best
            while True:
                best = assignment
                T = best.T - 1
                log_br()
                log_info(f'Trying T={T}...')
                assignment = task.run(T, fast=True, finalize=False)
                if assignment is None:
                    log_debug(f'MinimalPartialAutomatonTask: found minimal T={best.T}')
                    break

        task.finalize()

        if fast:
            time_total_run = time.time() - time_start_run
            if best:
                results = dict(Cres=best.C, Kres=best.K, Tres=best.T,
                               SAT=True, time=time_total_run)
            else:
                results = dict(SAT=False, time=time_total_run)
            self.save_results(results)

            log_debug(f'MinimalPartialAutomatonTask: done in {time_total_run:.2f} s')
            return best
        else:
            automaton = self.build_efsm(best, dump=False)
            time_total_run = time.time() - time_start_run
            if automaton:
                results = dict(Cres=automaton.C, Kres=automaton.K, Tres=automaton.T,
                               SAT=True, time=time_total_run)
            else:
                results = dict(SAT=False, time=time_total_run)
            self.save_results(results)
            self.save_efsm(automaton)

            log_debug(f'MinimalPartialAutomatonTask: done in {time_total_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Minimal partial automaton has {automaton.number_of_states} states and {automaton.number_of_transitions} transitions')
            else:
                log_error(f'Minimal partial automaton was not found')
            return automaton

    def build_efsm(self, assignment, *, dump=True):
        if assignment is None:
            return None

        log_br()
        log_info('MinimalPartialAutomatonTask: building automaton...')
        automaton = EFSM.new_with_truth_tables(self.scenario_tree, assignment)

        if dump:
            automaton.dump(self.get_filename_prefix(assignment.C, assignment.K, assignment.T))

        log_success('Minimal partial automaton:')
        automaton.pprint()
        automaton.verify(self.scenario_tree)

        return automaton
