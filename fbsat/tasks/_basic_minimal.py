import os
import time
import itertools
from collections import namedtuple

from ..utils import closed_range
from ..efsm import EFSM
from ..printers import log_debug, log_success, log_warn, log_br, log_info, log_error
from . import BasicAutomatonTask

__all__ = ['MinimalBasicAutomatonTask']


class MinimalBasicAutomatonTask:

    def __init__(self, scenario_tree, C=None, T=None, *, use_bfs=True, solver_cmd=None, write_strategy=None, outdir=''):
        self.scenario_tree = scenario_tree
        if C:
            self.C_start = C
        else:
            self.C_start = 1
        self.T_init = T
        self.outdir = outdir
        self.basic_config = dict(use_bfs=use_bfs,
                                 solver_cmd=solver_cmd,
                                 write_strategy=write_strategy,
                                 outdir=outdir)

    def get_stem(self, C, K, T):
        return f'minimal_basic_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_T{T}'

    def get_filename_prefix(self, C, K, T):
        return os.path.join(self.outdir, self.get_stem(C, K, T))

    def run(self, *, fast=False, only_C=False):
        log_debug(f'MinimalBasicAutomatonTask: running...')
        time_start_run = time.time()
        best = None

        for C in itertools.islice(itertools.count(self.C_start), 10):
            log_br()
            log_info(f'Trying C = {C}')
            task = BasicAutomatonTask(self.scenario_tree, C, **self.basic_config)
            assignment = task.run(self.T_init, fast=True)

            if assignment:
                if only_C:
                    best = assignment
                else:
                    while True:
                        best = assignment
                        T = best.T - 1
                        log_br()
                        log_info(f'Trying T = {T}...')
                        assignment = task.run(T, fast=True)
                        if assignment is None:
                            break
                break

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
