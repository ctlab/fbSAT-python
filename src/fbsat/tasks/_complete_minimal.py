import os
import time

from . import CompleteAutomatonTask, MinimalBasicAutomatonTask, Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success
from ..utils import parse_raw_assignment_algo, parse_raw_assignment_bool, parse_raw_assignment_int, s2b

__all__ = ['MinimalCompleteAutomatonTask']


class MinimalCompleteAutomatonTask(Task):

    def __init__(self, scenario_tree, *, C=None, K=None, P=None, N=None, use_bfs=True, solver_cmd=None, is_incremental=False, outdir=''):
        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.P = P
        self.N_init = N
        self.outdir = outdir
        self.subtask_config = dict(use_bfs=use_bfs,
                                   solver_cmd=solver_cmd,
                                   is_incremental=is_incremental,
                                   outdir=outdir)

    def get_stem(self, C, K, P, N=None):
        if N is None:
            return f'minimal_complete_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_P{P}'
        else:
            return f'minimal_complete_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_P{P}_N{N}'

    def get_filename_prefix(self, C, K, P, N=None):
        return os.path.join(self.outdir, self.get_stem(C, K, P, N))

    @property
    def number_of_variables(self):
        return self.solver.number_of_variables

    @property
    def number_of_clauses(self):
        return self.solver.number_of_clauses

    def run(self, *, fast=False):
        log_debug(f'MinimalCompleteAutomatonTask: running...')
        time_start_run = time.time()
        best = None

        if self.C is None:
            log_debug('MinimalCompleteAutomatonTask: searching for minimal C...')
            task = MinimalBasicAutomatonTask(self.scenario_tree, **self.subtask_config)
            assignment = task.run(fast=True, only_C=True)
            C = assignment.C
            log_debug(f'MinimalCompleteAutomatonTask: found minimal C={C}')
        else:
            C = self.C
            log_debug(f'MinimalCompleteAutomatonTask: using specified C={C}')

        if self.K is None:
            K = C
            log_debug(f'MinimalCompleteAutomatonTask: using K=C={K}')
        else:
            K = self.K
            log_debug(f'MinimalCompleteAutomatonTask: using specified K={K}')

        if self.P is None:
            log_debug('MinimalCompleteAutomatonTask: searching for P...')
            for P in [1, 3, 5, 7, 9, 15]:
                log_br()
                log_info(f'Trying P={P}...')
                task = CompleteAutomatonTask(self.scenario_tree, C=C, K=K, P=P, **self.subtask_config)
                assignment = task.run(self.N_init, fast=True)

                if assignment:
                    log_success(f'MinimalCompleteAutomatonTask: found P={P}')
                    break
                else:
                    task.finalize()
            else:
                log_error('MinimalCompleteAutomatonTask: P was not found')
        else:
            P = self.P
            log_br()
            log_info(f'MinimalCompleteAutomatonTask: pre-solving for specified P={P}...')
            task = CompleteAutomatonTask(self.scenario_tree, C=C, K=K, P=P, **self.subtask_config)
            assignment = task.run(self.N_init, fast=True)
            if assignment:
                log_success(f'MinimalCompleteAutomatonTask: pre-solved for P={P}')
                # TODO: show presolved semi-minimal automaton
            else:
                log_error(f'MinimalCompleteAutomatonTask: no solution for P={P}')

        while assignment:
            best = assignment
            N = best.N - 1
            log_br()
            log_info(f'Trying N = {N}...')
            assignment = task.run(N, fast=True)

        task.finalize()

        if fast:
            log_debug(f'MinimalCompleteAutomatonTask: done in {time.time() - time_start_run:.2f} s')
            return best
        else:
            automaton = self.build_efsm(best)

            log_debug(f'MinimalCompleteAutomatonTask: done in {time.time() - time_start_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Minimal complete automaton has {automaton.number_of_states} states, {automaton.number_of_transitions} transitions and {automaton.number_of_nodes} nodes')
            else:
                log_error(f'Minimal complete automaton was not found')
            return automaton

        log_debug(f'MinimalCompleteAutomatonTask: done in {time.time() - time_start_run:.2f} s')
        log_br()
        if automaton:
            log_success('')
        else:
            log_error('')
        return automaton

    def build_efsm(self, assignment, *, dump=True):
        if assignment is None:
            return None

        log_br()
        log_info('MinimalCompleteAutomatonTask: building automaton...')
        automaton = EFSM.new_with_parse_trees(self.scenario_tree, assignment)

        if dump:
            filename_gv = self.get_filename_prefix(assignment.C, assignment.K, assignment.P, assignment.N) + '.gv'
            automaton.write_gv(filename_gv)
            output_format = 'svg'
            cmd = f'dot -T{output_format} {filename_gv} -O'
            log_debug(cmd, symbol='$')
            os.system(cmd)

        log_success('Minimal complete automaton:')
        automaton.pprint()
        automaton.verify(self.scenario_tree)

        return automaton
