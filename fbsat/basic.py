__all__ = ('Instance', )

import time
import regex
import shutil
import tempfile
import itertools
import subprocess
from collections import namedtuple

from .utils import *
from .printers import *

VARIABLES = 'color transition output_event algorithm_0 algorithm_1'


class Instance:

    Reduction = namedtuple('Reduction', VARIABLES + ' tr nut')
    Assignment = namedtuple('Assignment', VARIABLES + ' C K')

    def __init__(self, scenario_tree, sat_solver, filename_prefix=''):
        self.scenario_tree = scenario_tree
        self.sat_solver = sat_solver
        self.filename_prefix = filename_prefix

        self.C = None
        self.K = None
        self.best = None
        self.number_of_clauses = 0
        self.bomba = itertools.count(1)

    def run(self):
        for C in closed_range(1, 10):
            self.C = C
            self.generate_base()
            self.number_of_base_clauses = self.number_of_clauses
            self.number_of_base_variables = self.number_of_variables

            assignment = self.solve()
            log_br()

            if assignment is not None:
                log_br()
                self.best = assignment
                self.generate_exists()

                for K in reversed(closed_range(0, C - 1)):
                    self.K = K
                    self.number_of_clauses = self.number_of_base_clauses
                    self.generate_cardinality()

                    assignment = self.solve()
                    log_br()

                    if assignment is None:
                        break

                    self.best = assignment
                else:
                    log_warn('Reached K = 0, weird...')

                log_success(f'BEST: C={self.best.C}, K={self.best.K}')
                log_br()
                break
        else:
            log_error('CAN`T FIND ANYTHING :C')

    @property
    def number_of_variables(self):
        return int(str(self.bomba)[6:-1]) - 1

    def declare_array(self, *dims, with_zero=False):
        def last():
            if with_zero:
                return [next(self.bomba) for _ in closed_range(0, dims[-1])]
            else:
                return [None] + [next(self.bomba) for _ in closed_range(1, dims[-1])]
        n = len(dims)
        if n == 1:
            return last()
        elif n == 2:
            return [None] + [last()
                             for _ in closed_range(1, dims[0])]
        elif n == 3:
            return [None] + [[None] + [last()
                                       for _ in closed_range(1, dims[1])]
                             for _ in closed_range(1, dims[0])]
        elif n == 4:
            return [None] + [[None] + [[None] + [last()
                                                 for _ in closed_range(1, dims[2])]
                                       for _ in closed_range(1, dims[1])]
                             for _ in closed_range(1, dims[0])]
        else:
            raise ValueError(f'unsupported number of dimensions ({n})')

    def add_clause(self, *vs):
        self.number_of_clauses += 1
        self.stream.write(' '.join(map(str, vs)) + ' 0\n')

    def generate_base(self):
        C = self.C

        log_debug(f'Generating base reduction for C = {C}...')
        time_start_base = time.time()

        self.stream = tempfile.NamedTemporaryFile('w', delete=False)

        # =-=-=-=-=-=
        #  CONSTANTS
        # =-=-=-=-=-=

        tree = self.scenario_tree
        V = tree.V
        E = tree.E
        O = tree.O
        X = tree.X
        Z = tree.Z
        U = tree.U
        Y = tree.Y

        # =-=-=-=-=-=
        #  VARIABLES
        # =-=-=-=-=-=

        # automaton variables
        color = self.declare_array(V, C)
        transition = self.declare_array(C, E, U, C)
        algorithm_0 = self.declare_array(C, Z)
        algorithm_1 = self.declare_array(C, Z)
        output_event = self.declare_array(C, O)
        # TODO: bfs variables

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

        def ALO(data):
            lower = 1 if data[0] is None else 0
            self.add_clause(*data[lower:])

        def AMO(data):
            lower = 1 if data[0] is None else 0
            upper = len(data) - 1
            for a in range(lower, upper):
                for b in closed_range(a + 1, upper):
                    self.add_clause(-data[a], -data[b])

        so_far_state = [self.number_of_clauses]

        def so_far():
            ans = self.number_of_clauses - so_far_state[0]
            so_far_state[0] = self.number_of_clauses
            return ans

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # 1. Color constraints
        # 1.0a ALO(color)
        for v in closed_range(1, V):
            ALO(color[v])
        # 1.0b AMO(color)
        for v in closed_range(1, V):
            AMO(color[v])

        # 1.1. Root corresponds to start state
        self.add_clause(color[1][1])

        log_debug(f'1. Clauses: {so_far()}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # 2. Transition constraints
        # 2.0. AMO(transition)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    AMO(transition[c][e][u])

        # 2.1. Transition definition
        for v in closed_range(2, V):
            p = tree.parent[v]
            e = tree.input_event[v]
            u = tree.input_number[v]
            for c1 in closed_range(1, C):
                for c2 in closed_range(1, C):
                    self.add_clause(-color[p][c1], -color[v][c2], transition[c1][e][u][c2])

        # 2.2. Transition coverage
        # for c in closed_range(1, C):
        #     for v in closed_range(1, V):
        #         p = tree.parent[v]
        #         e = tree.input_event[v]
        #         u = tree.input_number[v]
        #         self.add_clause()
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    self.add_clause(transition[c][e][u][c],
                                    *[color[tree.parent[v]][c] for v in closed_range(1, V)
                                      if tree.input_event[v] == e and tree.input_number[v] == u])

        log_debug(f'2. Clauses: {so_far()}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # 3. Algorithm constraints
        # 3.1. Start state does nothing
        for z in closed_range(1, Z):
            self.add_clause(-algorithm_0[1][z])
            self.add_clause(algorithm_1[1][z])

        # 3.2. Algorithms definition
        for v in closed_range(2, V):
            p = tree.parent[v]
            for c in closed_range(1, C):
                for z in closed_range(1, Z):
                    old = tree.output_value[p][z]
                    new = tree.output_value[v][z]
                    if (old, new) == (False, False):
                        self.add_clause(-color[v][c], -algorithm_0[c][z])
                    elif (old, new) == (False, True):
                        self.add_clause(-color[v][c], algorithm_0[c][z])
                    elif (old, new) == (True, False):
                        self.add_clause(-color[v][c], -algorithm_1[c][z])
                    elif (old, new) == (True, True):
                        self.add_clause(-color[v][c], algorithm_1[c][z])

        log_debug(f'3. Clauses: {so_far()}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # 4. Output event constraints
        # 4.0a. ALO(output_event)
        for c in closed_range(1, C):
            ALO(output_event[c])
        # 4.0b. AMO(output_event)
        for c in closed_range(1, C):
            AMO(output_event[c])

        # 4.1. Start state does INITO (root's output event)
        self.add_clause(output_event[1][tree.output_event[1]])

        # 4.2. Output event is the same as in the tree
        for v in closed_range(2, V):
            if tree.output_event[v] != 0:
                for c in closed_range(1, C):
                    self.add_clause(-color[v][c], output_event[c][tree.output_event[v]])

        log_debug(f'4. Clauses: {so_far()}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # TODO: 5. BFS constraints

        # log_debug(f'5. Clauses: {so_far()}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.stream.close()
        shutil.move(self.stream.name, self.get_filename_base())

        self.reduction = self.Reduction(
            color=color,
            transition=transition,
            output_event=output_event,
            algorithm_0=algorithm_0,
            algorithm_1=algorithm_1,
            tr=None,
            nut=None
        )

        log_debug(f'Done generating base reduction in {time.time() - time_start_base:.2f} s')

    def generate_exists(self):
        C = self.C

        log_debug(f'Generating transitions existance constraints for C = {C}...')
        time_start_exists = time.time()

        self.stream = tempfile.NamedTemporaryFile('w', delete=False)

        # =-=-=-=-=-=
        #  CONSTANTS
        # =-=-=-=-=-=

        tree = self.scenario_tree
        E = tree.E
        U = tree.U

        # =-=-=-=-=-=
        #  VARIABLES
        # =-=-=-=-=-=

        transition = self.reduction.transition
        tr = self.declare_array(C, C)
        nut = self.declare_array(C, C, C + 1)

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

        # Transitions existance
        for c1 in closed_range(1, C):
            for c2 in closed_range(1, C):
                if c1 == c2:
                    continue
                _cons = [-tr[c1][c2]]
                # FIXME: my favorite moment is "for (int e = 0; e < 1; e++)", seriously, up to 1???
                for e in closed_range(1, E):
                    for u in closed_range(1, U):
                        t = transition[c1][e][u][c2]
                        self.add_clause(-t, tr[c1][c2])
                        _cons.append(t)
                self.add_clause(*_cons)

        log_debug(f'Clauses: {self.number_of_clauses - self.number_of_base_clauses}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.stream.close()
        shutil.move(self.stream.name, self.get_filename_exists())

        self.reduction = self.reduction._replace(tr=tr, nut=nut)

        log_debug(f'Done generating transitions exists in {time.time() - time_start_exists:.2f} s')

    def generate_cardinality(self):
        C = self.C
        K = self.K

        log_debug(f'Generating transitions cardinality for C = {C} and K = {K}...')
        time_start_cardinality = time.time()

        self.stream = tempfile.NamedTemporaryFile('w', delete=False)
        _number_of_clauses = self.number_of_clauses

        # =-=-=-=-=-=
        #  VARIABLES
        # =-=-=-=-=-=

        tr = self.reduction.tr
        nut = self.reduction.nut

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

        # Transitions cardinality: leq K
        #  "any state has at least zero transitions"
        for c in closed_range(1, C):
            self.add_clause(nut[c][1][1])

        #  *silence*
        for c1 in closed_range(1, C):
            for c2 in closed_range(2, C):
                for c3 in closed_range(1, C):
                    self.add_clause(-nut[c1][c2 - 1][c3], -tr[c1][c2], nut[c1][c2][c3 + 1])
                for c3 in closed_range(1, C + 1):
                    self.add_clause(-nut[c1][c2 - 1][c3], tr[c1][c2], nut[c1][c2][c3])

        #  "forbid all states to have K+1 or more transitions"
        for c1 in closed_range(1, C):
            for c3 in closed_range(K + 1, C + 1):
                self.add_clause(-nut[c1][C][c3])

        log_debug(f'Clauses: {self.number_of_clauses - _number_of_clauses}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.stream.close()
        shutil.move(self.stream.name, self.get_filename_cardinality())

        log_debug(f'Done generating transitions cardinality in {time.time() - time_start_cardinality:.2f} s')

    def solve(self):
        if self.K is not None:
            log_info(f'Solving for C = {self.C} and K = {self.K}...')
        else:
            log_info(f'Solving for C = {self.C}...')
        time_start_solve = time.time()

        self.write_header()
        self.write_merged()

        cmd = f'{self.sat_solver} {self.get_filename_merged()}'
        log_debug(cmd)
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                           universal_newlines=True)

        if p.returncode == 10:
            log_success(f'SAT in {time.time() - time_start_solve:.2f} s')

            raw_assignment = [None]  # 1-based
            for line in p.stdout.split('\n'):
                if line.startswith('v'):
                    for value in map(int, regex.findall(r'-?\d+', line)):
                        if value == 0:
                            break
                        assert abs(value) == len(raw_assignment)
                        raw_assignment.append(value)
            return self.parse_raw_assignment(raw_assignment)
        elif p.returncode == 20:
            log_error(f'UNSAT in {time.time() - time_start_solve:.2f} s')
        else:
            log_error(f'returncode {p.returncode} in {time.time() - time_start_solve:.2f} s')

    def parse_raw_assignment(self, raw_assignment):
        log_info('Building assignment...')
        time_start_assignment = time.time()

        def wrapper_int(data):
            if isinstance(data[1], (list, tuple)):
                return [None] + list(map(wrapper_int, data[1:]))
            else:
                for i, x in enumerate(data):
                    if x is not None and raw_assignment[x] > 0:
                        return i
                log_warn('data[...] is unknown')

        def wrapper_bool(data):
            if isinstance(data[1], (list, tuple)):
                return [None] + list(map(wrapper_bool, data[1:]))
            else:
                if data[0] is None:
                    return [NotBool] + [raw_assignment[x] > 0 for x in data[1:]]
                else:
                    return [raw_assignment[x] > 0 for x in data]

        def wrapper_algo(data):
            return [None] + [''.join('1' if raw_assignment[item] > 0 else '0'
                                     for item in subdata[1:])
                             for subdata in data[1:]]

        assignment = self.Assignment(
            color=wrapper_int(self.reduction.color),
            transition=wrapper_int(self.reduction.transition),
            output_event=wrapper_int(self.reduction.output_event),
            algorithm_0=wrapper_algo(self.reduction.algorithm_0),
            algorithm_1=wrapper_algo(self.reduction.algorithm_1),
            C=self.C,
            K=self.K
        )

        log_success(f'Done building assignment in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def maybe_k(self):
        return f'_K{self.K}' if self.K is not None else ''

    def get_filename_base(self):
        return f'{self.filename_prefix}_C{self.C}_base.dimacs'

    def get_filename_exists(self):
        return f'{self.filename_prefix}_C{self.C}_exists.dimacs'

    def get_filename_cardinality(self):
        return f'{self.filename_prefix}_C{self.C}_K{self.K}_cardinality.dimacs'

    def get_filename_header(self):
        return f'{self.filename_prefix}_C{self.C}{self.maybe_k()}_header.dimacs'

    def get_filename_merged(self):
        return f'{self.filename_prefix}_C{self.C}{self.maybe_k()}_merged.dimacs'

    def get_filenames(self):
        if self.K is not None:
            return ' '.join((self.get_filename_header(),
                             self.get_filename_base(),
                             self.get_filename_exists(),
                             self.get_filename_cardinality()))
        else:
            return ' '.join((self.get_filename_header(),
                             self.get_filename_base()))

    def write_header(self, filename=None):
        if filename is None:
            filename = self.get_filename_header()

        # if self.is_reuse and os.path.exists(filename):
        #     log_debug(f'Reusing header from <{filename}>')
        #     return

        log_debug(f'Writing header to <{filename}>...')
        with open(filename, 'w') as f:
            f.write(f'p cnf {self.number_of_variables} {self.number_of_clauses}\n')

    def write_merged(self, filename=None):
        if filename is None:
            filename = self.get_filename_merged()

        # if self.is_reuse and os.path.exists(filename):
        #     log_debug(f'Reusing merged reduction from <{filename}>')
        #     return

        log_debug(f'Writing merged reduction to <{filename}>...')
        cmd_cat = f'cat {self.get_filenames()} > {filename}'
        log_debug(cmd_cat, symbol='$')
        subprocess.run(cmd_cat, shell=True)
