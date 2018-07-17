import os
import time
from collections import namedtuple
from functools import partial

from ..utils import closed_range, s2b, parse_raw_assignment_int, parse_raw_assignment_bool, parse_raw_assignment_algo
from ..solver import Solver
from ..printers import log_debug, log_success, log_warn, log_br, log_info, log_error
from . import BasicAutomatonTask, MinimalBasicAutomatonTask, CompleteAutomatonTask
from ..efsm import EFSM

__all__ = ['MinimalCompleteAutomatonTask']


class MinimalCompleteAutomatonTask:

    def __init__(self, scenario_tree, *, C=None, K=None, P=None, N=None, use_bfs=True, solver_cmd=None, write_strategy=None, outdir=''):
        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.P = P
        self.N_init = N
        self.use_bfs = use_bfs
        self.outdir = outdir
        self.basic_config = dict(use_bfs=use_bfs,
                                 solver_cmd=solver_cmd,
                                 write_strategy=write_strategy,
                                 outdir=outdir)
        self.config = {'cmd': solver_cmd}
        if write_strategy is not None:
            self.config['write_strategy'] = write_strategy

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
            task = MinimalBasicAutomatonTask(self.scenario_tree, **self.basic_config)
            assignment = task.run(fast=True, only_C=True)
            C = assignment.C
            log_debug(f'MinimalCompleteAutomatonTask: found minimal C={C}...')
        else:
            C = self.C
            log_debug(f'MinimalCompleteAutomatonTask: using specified C={C}...')

        if self.K is None:
            K = C
            log_debug(f'Using K=C={K}')
        else:
            K = self.K
            log_debug(f'Using specified K={K}')

        if self.P is None:
            log_br()
            log_info('MinimalCompleteAutomatonTask: searching for P...')
            for P in [1, 3, 5, 7, 9, 15]:
                # log_br()
                log_info(f'Trying P = {P}...')
                task = CompleteAutomatonTask(self.scenario_tree, C=C, K=K, P=P, **self.basic_config)
                assignment = task.run(self.N_init, fast=True)

                if assignment:
                    log_success(f'MinimalCompleteAutomatonTask: found P={P}')
                    break
            else:
                log_error('MinimalCompleteAutomatonTask: P was not found')
        else:
            P = self.P
            log_br()
            log_info(f'MinimalCompleteAutomatonTask: pre-solving for specified P={P}...')
            task = CompleteAutomatonTask(self.scenario_tree, C=C, K=K, P=P, **self.basic_config)
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
