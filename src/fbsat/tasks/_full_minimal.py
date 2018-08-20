import itertools
import os
import time

from . import FullAutomatonTask, Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success

__all__ = ['MinimalFullAutomatonTask']


class MinimalFullAutomatonTask(Task):

    def __init__(self, scenario_tree, *, C=None, T=None, use_bfs=True, solver_cmd=None, is_incremental=False, is_filesolver=False, outdir=''):
        self.scenario_tree = scenario_tree
        self.C = C
        self.T_init = T
        self.outdir = outdir
        self.subtask_config = dict(use_bfs=use_bfs,
                                   solver_cmd=solver_cmd,
                                   is_incremental=is_incremental,
                                   is_filesolver=is_filesolver,
                                   outdir=outdir)

    def get_stem(self, C, T):
        return f'minimal_full_{self.scenario_tree.scenarios_stem}_C{C}_T{T}'

    def get_filename_prefix(self, C, T):
        return os.path.join(self.outdir, self.get_stem(C, T))

    def run(self, *, fast=False, only_C=False):
        log_debug(f'MinimalFullAutomatonTask: running...')
        time_start_run = time.time()
        best = None

        if self.C is None:
            log_debug('MinimalFullAutomatonTask: searching for minimal C...')
            for C in itertools.islice(itertools.count(1), 15):
                log_br()
                log_info(f'Trying C = {C}...')
                task = FullAutomatonTask(self.scenario_tree, C=C, **self.subtask_config)
                assignment = task.run(self.T_init, fast=True, finalize=False)
                if assignment:
                    best = assignment
                    log_debug(f'MinimalFullAutomatonTask: found minimal C={C}')
                    break
                else:
                    task.finalize()
            else:
                log_error('MinimalFullAutomatonTask: minimal C was not found')
        else:
            log_debug(f'MinimalFullAutomatonTask: using specified C={self.C}')
            task = FullAutomatonTask(self.scenario_tree, C=self.C, **self.subtask_config)
            best = task.run(self.T_init, fast=True, finalize=False)

        if not only_C and best:
            log_debug('MinimalFullAutomatonTask: searching for minimal T...')
            assignment = best
            while True:
                best = assignment
                T = best.T - 1
                log_br()
                log_info(f'Trying T={T}...')
                assignment = task.run(T, fast=True, finalize=False)
                if assignment is None:
                    log_debug(f'MinimalFullAutomatonTask: found minimal T={best.T}')
                    break

        task.finalize()

        if fast:
            log_debug(f'MinimalFullAutomatonTask: done in {time.time() - time_start_run:.2f} s')
            return best
        else:
            automaton = self.build_efsm(best)

            log_debug(f'MinimalFullAutomatonTask: done in {time.time() - time_start_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Minimal full automaton has {automaton.number_of_states} states and {automaton.number_of_transitions} transitions')
            else:
                log_error(f'Minimal full automaton was not found')
            return automaton

    def build_efsm(self, assignment, *, dump=True):
        if assignment is None:
            return None

        log_br()
        log_info('MinimalFullAutomatonTask: building automaton...')
        automaton = EFSM.new_with_full_guards(self.scenario_tree, assignment)

        if dump:
            automaton.dump(self.get_filename_prefix(assignment.C, assignment.T))

        log_success('Minimal full automaton:')
        automaton.pprint()
        automaton.verify()

        return automaton
