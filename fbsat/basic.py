__all__ = ('Instance', )

import os
import time
import regex
import shutil
import tempfile
import itertools
import subprocess
from io import StringIO
from collections import namedtuple

from .utils import *
from .printers import *

VARIABLES = 'color transition output_event algorithm_0 algorithm_1'


class Instance:

    Reduction = namedtuple('Reduction', VARIABLES + ' nut')
    Assignment = namedtuple('Assignment', VARIABLES)

    def __init__(self, *, scenario_tree, C_start=1, C_end=10, is_incremental=False, sat_solver=None, sat_isolver=None, filename_prefix='', write_strategy='StringIO', is_reuse=False):
        assert write_strategy in ('direct', 'tempfile', 'StringIO')

        if is_incremental:
            assert sat_isolver is not None, "You need to specify incremental sat-solver using `--sat-isolver` option"
        else:
            assert sat_solver is not None, "You need to specify sat-solver using `--sat-solver` option"

        self.scenario_tree = scenario_tree
        self.is_incremental = is_incremental
        self.sat_solver = sat_solver
        self.sat_isolver = sat_isolver
        self.filename_prefix = filename_prefix
        self.write_strategy = write_strategy
        self.is_reuse = is_reuse

        if C_end:
            self.C_iter = closed_range(C_start, C_end)
        else:
            self.C_iter = itertools.count(C_start)
            self.C_iter = itertools.islice(self.C_iter, 20)  # FIXME: stub
        self.best = None
        self.K_best = None
        self.K_defined = None

    def run(self):
        for C in self.C_iter:
            log_info(f'Trying C = {C}')
            if self.is_incremental:
                self.solver_process = subprocess.Popen(self.sat_isolver, shell=True, universal_newlines=True,
                                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                self.stream = self.solver_process.stdin  # to write uniformly
            self.C = C
            self.K = None
            self.number_of_variables = 0
            self.number_of_clauses = 0
            self.generate_base()
            number_of_base_variables = self.number_of_variables
            number_of_base_clauses = self.number_of_clauses
            log_debug(f'Base variables: {number_of_base_variables}')
            log_debug(f'Base clauses: {number_of_base_clauses}')

            assignment = self.solve()
            log_br()

            if assignment:  # SAT
                self.best = assignment
                self.generate_pre()
                number_of_pre_variables = self.number_of_variables - number_of_base_variables
                number_of_pre_clauses = self.number_of_clauses - number_of_base_clauses
                log_debug(f'Pre variables: {number_of_pre_variables}')
                log_debug(f'Pre clauses: {number_of_pre_clauses}')

                # Note: K=C-1 == unconstrained base reduction
                for K in reversed(closed_range(0, C - 2)):
                    log_info(f'Trying C = {C}, K = {K}')
                    self.K = K
                    # Revert possible previous variables and constraints
                    self.number_of_variables = number_of_base_variables + number_of_pre_variables
                    self.number_of_clauses = number_of_pre_clauses + number_of_base_clauses
                    self.generate_cardinality()

                    assignment = self.solve()
                    log_br()

                    if assignment is None:  # UNSAT -> answer is last SAT (self.best)
                        break

                    self.best = assignment
                    self.K_best = K
                else:
                    log_warn('Reached K = 0, weird...')

                if self.is_incremental:
                    self.solver_process.kill()

                log_success(f'BEST: C={self.C}, K={self.K_best}')
                log_br()
                break
            else:
                if self.is_incremental:
                    self.solver_process.kill()
        else:
            log_error('CAN`T FIND ANYTHING :C')

    def maybe_new_stream(self, filename):
        if not self.is_incremental:
            if self.is_reuse and os.path.exists(filename):
                log_debug(f'Reusing <{filename}>')
                self.stream = None
                return

            if self.write_strategy == 'direct':
                self.stream = open(filename, 'w')
            elif self.write_strategy == 'tempfile':
                self.stream = tempfile.NamedTemporaryFile('w', delete=False)
            elif self.write_strategy == 'StringIO':
                self.stream = StringIO()

    def maybe_close_stream(self, filename):
        if not self.is_incremental and self.stream is not None:
            if self.write_strategy == 'direct':
                self.stream.close()
            elif self.write_strategy == 'tempfile':
                self.stream.close()
                shutil.move(self.stream.name, filename)
            elif self.write_strategy == 'StringIO':
                with open(filename, 'w') as f:
                    self.stream.seek(0)
                    shutil.copyfileobj(self.stream, f)
                self.stream.close()

    def new_variable(self):
        self.number_of_variables += 1
        return self.number_of_variables

    def add_clause(self, *vs):
        self.number_of_clauses += 1
        if self.stream is not None:
            self.stream.write(' '.join(map(str, vs)) + ' 0\n')

    def declare_array(self, *dims, with_zero=False):
        def last():
            if with_zero:
                return [self.new_variable() for _ in closed_range(0, dims[-1])]
            else:
                return [None] + [self.new_variable() for _ in closed_range(1, dims[-1])]
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

    def generate_base(self):
        C = self.C

        log_debug(f'Generating base reduction for C = {C}...')
        time_start_base = time.time()

        self.maybe_new_stream(self.get_filename_base())

        # =-=-=-=-=-=
        #  CONSTANTS
        # =-=-=-=-=-=

        tree = self.scenario_tree
        V = tree.V
        E = tree.E
        O = tree.O
        # X = tree.X
        Z = tree.Z
        U = tree.U
        # Y = tree.Y

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

        log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

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
                    # color_p,c1 & color_v,c2 => transition_c1,e,u,c2
                    self.add_clause(-color[p][c1], -color[v][c2], transition[c1][e][u][c2])

        # 2.2. Transition coverage
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    # not transition_c,e,u,c => OR_{v in V | tie[v]==e and tin[v]=u} color_parent[v],c
                    rhs = []
                    for v in closed_range(1, V):
                        if tree.input_event[v] == e and tree.input_number[v] == u:
                            rhs.append(color[tree.parent[v]][c])
                    self.add_clause(transition[c][e][u][c], *rhs)

        log_debug(f'2. Clauses: {so_far()}', symbol='STAT')

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

        log_debug(f'3. Clauses: {so_far()}', symbol='STAT')

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

        log_debug(f'4. Clauses: {so_far()}', symbol='STAT')

        # TODO: 5. BFS constraints

        # log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.maybe_close_stream(self.get_filename_base())

        self.reduction = self.Reduction(
            color=color,
            transition=transition,
            output_event=output_event,
            algorithm_0=algorithm_0,
            algorithm_1=algorithm_1,
            nut=None
        )

        log_debug(f'Done generating base reduction in {time.time() - time_start_base:.2f} s')

    def generate_pre(self):
        C = self.C

        log_debug(f'Generating transitions existance and pre-cardinality constraints for C = {C}...')
        time_start_pre = time.time()

        self.maybe_new_stream(self.get_filename_pre())

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
        nut = self.declare_array(C, C, C, with_zero=True)

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

        # Transitions existance
        for i in closed_range(1, C):
            for j in closed_range(1, C):
                if i == j:
                    continue
                rhs = []
                # FIXME: my favorite moment is "for (int e = 0; e < 1; e++)", seriously, up to 1???
                for e in closed_range(1, E):
                    for u in closed_range(1, U):
                        t = transition[i][e][u][j]
                        self.add_clause(-t, tr[i][j])
                        rhs.append(t)
                self.add_clause(*rhs, -tr[i][j])

        # Transitions cardinality
        #  "any state has at least zero transitions"
        for c in closed_range(1, C):
            self.add_clause(nut[c][1][0])

        #  *silence*
        for i in closed_range(1, C):
            for j in closed_range(2, C):
                for k in closed_range(0, C - 1):
                    self.add_clause(-nut[i][j - 1][k], -tr[i][j], nut[i][j][k + 1])
                for k in closed_range(0, C):
                    self.add_clause(-nut[i][j - 1][k], tr[i][j], nut[i][j][k])

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.maybe_close_stream(self.get_filename_pre())

        self.reduction = self.reduction._replace(nut=nut)

        log_debug(f'Done generating pre in {time.time() - time_start_pre:.2f} s')

    def generate_cardinality(self):
        C = self.C
        K = self.K

        log_debug(f'Generating transitions cardinality...')
        time_start_cardinality = time.time()

        self.maybe_new_stream(self.get_filename_cardinality())

        if self.is_incremental and self.K_defined is not None:
            K_max = self.K_defined
        else:
            K_max = C

        log_debug(f'K_defined = {self.K_defined}')
        log_debug(f'CARDINALITY from K+1 to K_max = {K+1}..{K_max}')
        log_debug(f'Variables for color C: {[-self.reduction.nut[C][C][k] for k in reversed(closed_range(K + 1, K_max))]}')

        # Transitions cardinality: leq K
        #  "forbid all states to have K+1 or more transitions"
        for c in closed_range(1, C):
            for k in reversed(closed_range(K + 1, K_max)):
                self.add_clause(-self.reduction.nut[c][C][k])

        if self.is_incremental:
            self.K_defined = K

        self.maybe_close_stream(self.get_filename_cardinality())

        log_debug(f'Done generating transitions cardinality in {time.time() - time_start_cardinality:.2f} s')

    def solve(self):
        log_info(f'Solving...')
        time_start_solve = time.time()

        if self.is_incremental:
            p = self.solver_process
            p.stdin.write('solve 0\n')  # TODO: pass timeout?
            p.stdin.flush()

            answer = p.stdout.readline().rstrip()

            if answer == 'SAT':
                log_success(f'SAT in {time.time() - time_start_solve:.2f} s')
                line_assignment = p.stdout.readline().rstrip()
                if line_assignment.startswith('v '):
                    raw_assignment = [None] + list(map(int, line_assignment[2:].split()))  # 1-based
                    return self.parse_raw_assignment(raw_assignment)
                else:
                    log_error('Error reading line with assignment')
            elif answer == 'UNSAT':
                log_error(f'UNSAT in {time.time() - time_start_solve:.2f} s')
            elif answer == 'UNKNOWN':
                log_error(f'UNKNOWN in {time.time() - time_start_solve:.2f} s')
        else:
            self.write_header()
            self.write_merged()

            cmd = f'{self.sat_solver} {self.get_filename_merged()}'
            log_debug(cmd)
            p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)

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
        log_debug('Building assignment...')
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
        )

        log_debug(f'Done building assignment in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def maybe_k(self):
        return f'_K{self.K}' if self.K is not None else ''

    def get_filename_base(self):
        return f'{self.filename_prefix}_C{self.C}_base.dimacs'

    def get_filename_pre(self):
        return f'{self.filename_prefix}_C{self.C}_pre.dimacs'

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
                             self.get_filename_pre(),
                             self.get_filename_cardinality()))
        else:
            return ' '.join((self.get_filename_header(),
                             self.get_filename_base()))

    def write_header(self):
        filename = self.get_filename_header()
        if self.is_reuse and os.path.exists(filename):
            log_debug(f'Reusing header from <{filename}>')
            return
        log_debug(f'Writing header to <{filename}>...')
        with open(filename, 'w') as f:
            f.write(f'p cnf {self.number_of_variables} {self.number_of_clauses}\n')

    def write_merged(self):
        filename = self.get_filename_merged()
        if self.is_reuse and os.path.exists(filename):
            log_debug(f'Reusing merged reduction from <{filename}>')
            return
        log_debug(f'Writing merged reduction to <{filename}>...')
        cmd_cat = f'cat {self.get_filenames()} > {filename}'
        log_debug(cmd_cat, symbol='$')
        subprocess.run(cmd_cat, shell=True)
