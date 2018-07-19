import itertools
import os
import time

from . import BasicAutomatonTask, Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success, log_warn
from ..utils import closed_range

__all__ = ['MinimalBasicAutomatonTask']


class MinimalBasicAutomatonTask(Task):

    def __init__(self, scenario_tree, *, C=None, K=None, T=None, use_bfs=True, solver_cmd=None, is_incremental=False, outdir=''):
        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.T_init = T
        self.outdir = outdir
        self.subtask_config = dict(use_bfs=use_bfs,
                                   solver_cmd=solver_cmd,
                                   is_incremental=is_incremental,
                                   outdir=outdir)

    def get_stem(self, C, K, T):
        return f'minimal_basic_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_T{T}'

    def get_filename_prefix(self, C, K, T):
        return os.path.join(self.outdir, self.get_stem(C, K, T))

    def run(self, *, fast=False, only_C=False):
        log_debug(f'MinimalBasicAutomatonTask: running...')
        time_start_run = time.time()

        if self.C is None:
            log_debug('MinimalBasicAutomatonTask: searching for minimal C...')
            for C in itertools.islice(itertools.count(1), 15):
                log_br()
                log_info(f'Trying C = {C}...')
                task = BasicAutomatonTask(self.scenario_tree, C=C, K=self.K, **self.subtask_config)
                assignment = task.run(self.T_init, fast=True)
                if assignment:
                    best = assignment
                    log_debug(f'MinimalBasicAutomatonTask: found minimal C={C}')
                    break
                else:
                    task.finalize()
            else:
                log_error('MinimalBasicAutomatonTask: minimal C was not found')
        else:
            log_debug(f'MinimalBasicAutomatonTask: using specified C={self.C}')
            task = BasicAutomatonTask(self.scenario_tree, C=self.C, K=self.K, **self.subtask_config)
            best = task.run(self.T_init, fast=True)

        if not only_C and assignment:
            log_debug('MinimalBasicAutomatonTask: searching for minimal T...')
            while True:
                best = assignment
                T = best.T - 1
                log_br()
                log_info(f'Trying T={T}...')
                assignment = task.run(T, fast=True)
                if assignment is None:
                    log_debug(f'MinimalBasicAutomatonTask: found minimal T={best.T}')
                    break

        task.finalize()

        if fast:
            log_debug(f'MinimalBasicAutomatonTask: done in {time.time() - time_start_run:.2f} s')
            return best
        else:
            automaton = self.build_efsm(best)

            log_debug(f'MinimalBasicAutomatonTask: done in {time.time() - time_start_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Minimal basic automaton has {automaton.number_of_states} states and {automaton.number_of_transitions} transitions')
            else:
                log_error(f'Minimal basic automaton was not found')
            return automaton

    def build_efsm(self, assignment, *, dump=True):
        if assignment is None:
            return None

        log_br()
        log_info('MinimalBasicAutomatonTask: building automaton...')
        automaton = EFSM.new_with_truth_tables(self.scenario_tree, assignment)

        if dump:
            filename_gv = self.get_filename_prefix(assignment.C, assignment.K, assignment.T) + '.gv'
            automaton.write_gv(filename_gv)
            output_format = 'svg'
            cmd = f'dot -T{output_format} {filename_gv} -O'
            log_debug(cmd, symbol='$')
            os.system(cmd)

        log_success('Minimal basic automaton:')
        automaton.pprint()
        automaton.verify(self.scenario_tree)

        return automaton
