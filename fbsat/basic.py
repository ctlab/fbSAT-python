__all__ = ('Instance', )

import time

from .utils import *
from .printers import *


class Instance:

    def __init__(self, scenario_tree, sat_solver, filename_prefix=''):
        self.scenario_tree = scenario_tree
        self.sat_solver = sat_solver
        self.filename_prefix = filename_prefix
        self.best = None

    def run(self):
        for C in closed_range(1, 10):
            self.generate_base(C)

            assignment = self.solve()

            if assignment is not None:
                self.best = assignment

                for K in reversed(closed_range(0, C - 1)):
                    self.generate_transition_constraints(K)

                    assignment = self.solve()

                    if assignment is None:
                        self.best = assignment
                        break
                else:
                    self.best = assignment  # from inside loop

                break
        else:
            log_error('CAN`T FIND ANYTHING :C')

    def generate_base(self, C):
        log_debug(f'Generating base reduction for C = {C}...')
        time_start_base = time.time()

        pass

        log_debug(f'Done generating base reduction in {time.time() - time_start_base:.2f} s')

    def generate_transition_constraints(self, K):
        log_debug(f'Generating transition constraints for K = {K}...')
        time_start_transition = time.time()

        pass

        log_debug(f'Done generating transition constraints in {time.time() - time_start_transition:.2f} s')

    def solve(self):
        log_debug('Solving...')
        time_start_solve = time.time()

        pass

        log_debug(f'Done solving ._.')
