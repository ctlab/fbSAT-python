import os
import time

from . import CompleteAutomatonTask, MinimalCompleteAutomatonTask, MinimalPartialAutomatonTask, Task
from ..efsm import EFSM
from ..printers import log_br, log_debug, log_error, log_info, log_success, log_warn

__all__ = ['MinimalCompleteUBAutomatonTask']


class MinimalCompleteUBAutomatonTask(Task):

    def __init__(self, scenario_tree, *, C=None, K=None, w=None, use_bfs=True, is_distinct=False, is_forbid_or=False, solver_cmd=None, is_incremental=False, is_filesolver=False, outdir=''):
        # w :: 0=None=inf: only UB, 1: first-sat, 2: heuristic, >2: very-heuristic
        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.w = w
        self.outdir = outdir
        self.subtask_config_minpartial = dict(scenario_tree=scenario_tree,
                                              use_bfs=use_bfs,
                                              solver_cmd=solver_cmd,
                                              is_incremental=is_incremental,
                                              is_filesolver=is_filesolver,
                                              outdir=outdir)
        self.subtask_config_mincomplete = dict(**self.subtask_config_minpartial,
                                               is_distinct=is_distinct,
                                               is_forbid_or=is_forbid_or)

    def get_stem(self, C, K, P, N=None):
        if N is None:
            return f'minimal_complete_ub_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_P{P}'
        else:
            return f'minimal_complete_ub_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_P{P}_N{N}'

    def get_filename_prefix(self, C, K, P, N=None):
        return os.path.join(self.outdir, self.get_stem(C, K, P, N))

    def run(self, *, fast=False):
        log_debug(f'MinimalCompleteUBAutomatonTask: running...')
        time_start_run = time.time()

        if self.C is None:
            log_debug('MinimalCompleteUBAutomatonTask: searching for minimal C...')
            task = MinimalPartialAutomatonTask(**self.subtask_config_minpartial)
            assignment = task.run(fast=True)
            C = assignment.C
            T_min = assignment.T
            log_debug(f'MinimalCompleteUBAutomatonTask: found minimal C={C}')
        else:
            raise NotImplementedError('T_min is unknown with specified C')
            C = self.C
            log_debug(f'MinimalCompleteUBAutomatonTask: using specified C={C}')

        if self.K is None:
            K = C
            log_debug(f'MinimalCompleteUBAutomatonTask: using K=C={K}')
        else:
            K = self.K
            log_debug(f'MinimalCompleteUBAutomatonTask: using specified K={K}')

        P_low = None
        best = None
        prev = None

        log_debug('MinimalCompleteUBAutomatonTask: searching for P...')
        from itertools import chain, count
        for P in chain([1, 3, 5, 7, 9], count(10)):
        # for P in count(1):
            log_br()
            log_info(f'Trying P={P}...')

            if best and P > (best.N - T_min):
                log_warn(f'Upper bound reached! P={P}, N_best={best.N}, T_min={T_min}')
                break
            # THIS IS FALSE Note: w=0 == w=None == w=+inf == no width limit
            # Note: w=None == w=inf == no width limit,
            #       but w=0 is okay 1-"width" limit
            if P_low is not None and self.w is not None and (P - P_low) > self.w:
                log_warn(f'Maximum width reached! P={P}, P_low={P_low}, w={self.w}')
                break

            if best:
                N = best.N - 1
            else:
                N = None
            task = MinimalCompleteAutomatonTask(C=C, K=K, P=P, N=N, **self.subtask_config_mincomplete)
            assignment = task.run(fast=True)

            if assignment:
                if best is None or assignment.N < best.N:
                    best = assignment
                if prev is None or assignment.N != prev.N:
                    P_low = P
                prev = assignment
        else:
            log_error('MinimalCompleteUBAutomatonTask: P was not found')

        if fast:
            log_debug(f'MinimalCompleteUBAutomatonTask: done in {time.time() - time_start_run:.2f} s')
            return best
        else:
            automaton = self.build_efsm(best)

            log_debug(f'MinimalCompleteUBAutomatonTask: done in {time.time() - time_start_run:.2f} s')
            log_br()
            if automaton:
                log_success(f'Minimal complete automaton has {automaton.number_of_states} states, {automaton.number_of_transitions} transitions and {automaton.number_of_nodes} nodes')
            else:
                log_error(f'Minimal complete automaton was not found')
            return automaton

        log_debug(f'MinimalCompleteUBAutomatonTask: done in {time.time() - time_start_run:.2f} s')
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
        log_info('MinimalCompleteUBAutomatonTask: building automaton...')
        automaton = EFSM.new_with_parse_trees(self.scenario_tree, assignment)

        if dump:
            automaton.dump(self.get_filename_prefix(assignment.C, assignment.K, assignment.P, assignment.N))

        log_success('Minimal complete automaton:')
        automaton.pprint()
        automaton.verify()

        return automaton
