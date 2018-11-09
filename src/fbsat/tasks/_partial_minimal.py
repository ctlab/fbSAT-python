import itertools
import os
import time

from . import PartialAutomatonTask, Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success

__all__ = ['MinimalPartialAutomatonTask']


class MinimalPartialAutomatonTask(Task):

    def __init__(self, scenario_tree, *, C=None, K=None, T=None, use_bfs=True, is_distinct=False, solver_cmd=None, is_incremental=False, is_filesolver=False, outdir=''):
        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.T_init = T
        self.outdir = outdir
        self.subtask_config = dict(scenario_tree=scenario_tree,
                                   use_bfs=use_bfs,
                                   is_distinct=is_distinct,
                                   solver_cmd=solver_cmd,
                                   is_incremental=is_incremental,
                                   is_filesolver=is_filesolver,
                                   outdir=outdir)

    def get_stem(self, C, K, T):
        return f'minimal_partial_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_T{T}'

    def get_filename_prefix(self, C, K, T):
        return os.path.join(self.outdir, self.get_stem(C, K, T))

    def run(self, *, fast=False, only_C=False):
        log_debug(f'MinimalPartialAutomatonTask: running...')
        time_start_run = time.time()
        best = None

        if self.C is None:
            log_debug('MinimalPartialAutomatonTask: searching for minimal C...')
            for C in itertools.islice(itertools.count(1), 15):
                log_br()
                log_info(f'Trying C = {C}...')
                task = PartialAutomatonTask(C=C, K=self.K, **self.subtask_config)
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
            task = PartialAutomatonTask(C=self.C, K=self.K, **self.subtask_config)
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
            log_debug(f'MinimalPartialAutomatonTask: done in {time.time() - time_start_run:.2f} s')
            return best
        else:
            automaton = self.build_efsm(best)

            log_debug(f'MinimalPartialAutomatonTask: done in {time.time() - time_start_run:.2f} s')
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
