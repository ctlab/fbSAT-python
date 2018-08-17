import os
import time

from . import PartialAutomatonTask, MinimalPartialAutomatonTask, MinimizeGuardTask
from ..printers import log_br, log_debug, log_error, log_info, log_success, log_warn
from ..utils import closed_range, s2b

__all__ = ['MinimizeAllGuardsTask']


class MinimizeAllGuardsTask:

    def __init__(self, scenario_tree, *, partial_automaton=None, C=None, K=None, T=None, use_bfs=True, solver_cmd=None, outdir=''):
        self.scenario_tree = scenario_tree
        self.partial_automaton = partial_automaton
        self.C = C
        self.K = K
        self.T = T
        self.outdir = outdir
        self.subtask_config_guard = dict(scenario_tree=scenario_tree,
                                         solver_cmd=solver_cmd,
                                         outdir=outdir)
        self.subtask_config_automaton = dict(**self.subtask_config_guard,
                                             C=self.C, K=self.K,
                                             use_bfs=use_bfs)

    def get_stem(self, C, K, T):
        return f'minimized_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_T{T}'

    def get_filename_prefix(self, C, K, T):
        return os.path.join(self.outdir, self.get_stem(C, K, T))

    def run(self):
        log_debug('MinimizeAllGuardsTask: running...')
        time_start_run = time.time()

        if self.partial_automaton:
            automaton = self.partial_automaton
        else:
            log_debug('MinimizeAllGuardsTask: building partial automaton...')
            if self.T is not None:
                task = PartialAutomatonTask(**self.subtask_config_automaton)
                automaton = task.run(self.T)
            else:
                task = MinimalPartialAutomatonTask(**self.subtask_config_automaton)
                automaton = task.run()

        if automaton:
            log_br()
            log_info('MinimizeAllGuardsTask: minimizing guards...')
            C = automaton.number_of_states
            if self.K is None or self.T:
                K = C
            else:
                K = self.K
            T = automaton.number_of_transitions

            for c in closed_range(1, C):
                state = automaton.states[c]
                for k, transition in enumerate(state.transitions, start=1):
                    log_br()
                    log_info(f'MinimizeAllGuardsTask: minimizing guard on transition k={k} from {transition.source.id} to {transition.destination.id}...')
                    task = MinimizeGuardTask(**self.subtask_config_guard, guard=transition.guard)
                    minimized_guard = task.run()
                    if minimized_guard:
                        log_debug(f'MinimizeAllGuardsTask: guard on transition k={k} from {transition.source.id} to {transition.destination.id} was minimized to {minimized_guard}')
                        transition.guard = minimized_guard
                    else:
                        log_debug(f'MinimizeAllGuardsTask: guard on transition k={k} from {transition.source.id} to {transition.destination.id} was not minimized')

            automaton.dump(self.get_filename_prefix(C, K, T))

            log_success('Automaton with minimized guards:')
            automaton.pprint()
            automaton.verify()

        log_debug(f'MinimizeAllGuardsTask: done in {time.time() - time_start_run:.2f} s')
        log_br()
        if automaton:
            log_success('')
        else:
            log_error('')
        return automaton
