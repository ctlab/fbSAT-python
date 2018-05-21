__all__ = ('Instance', )

import os
import time
import regex
import shutil
import tempfile
import subprocess
from io import StringIO
from collections import deque, namedtuple

from .utils import *
from .printers import *

VARIABLES = 'color transition trans_event output_event algorithm_0 algorithm_1 nodetype terminal child_left child_right parent value child_value_left child_value_right fired_only not_fired'


class Instance:

    Reduction = namedtuple('Reduction', VARIABLES + ' totalizer')
    Assignment = namedtuple('Assignment', VARIABLES + ' number_of_nodes')

    def __init__(self, *, scenario_tree, C, K, P, N_start=0, is_minimize=False, is_incremental=False, sat_solver=None, sat_isolver=None, filename_prefix='', write_strategy='StringIO', is_reuse=False):
        assert write_strategy in ('direct', 'tempfile', 'StringIO')

        if is_incremental:
            assert sat_isolver is not None, "You need to specify incremental SAT solver using `--sat-isolver` option"
        else:
            assert sat_solver is not None, "You need to specify sat-solver using `--sat-solver` option"

        self.scenario_tree = scenario_tree
        self.C = C
        self.K = K
        self.P = P
        self.N_start = N_start
        self.is_minimize = is_minimize
        self.is_incremental = is_incremental
        self.sat_solver = sat_solver
        self.sat_isolver = sat_isolver
        self.filename_prefix = filename_prefix
        self.write_strategy = write_strategy
        self.is_reuse = is_reuse

        self.best = None
        self.N = None
        self.N_defined = None

    def run(self):
        if self.is_incremental:
            self.solver_process = subprocess.Popen(self.sat_isolver, shell=True, universal_newlines=True,
                                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            self.stream = self.solver_process.stdin  # to write uniformly

        self.number_of_variables = 0
        self.number_of_clauses = 0
        self.generate_base_reduction()
        number_of_base_variables = self.number_of_variables
        number_of_base_clauses = self.number_of_clauses
        log_debug(f'Base variables: {number_of_base_variables}')
        log_debug(f'Base clauses: {number_of_base_clauses}')

        if self.is_minimize:
            if self.N_start != 0:
                self.N = self.N_start

                self.generate_totalizer()
                number_of_totalizer_variables = self.number_of_variables - number_of_base_variables
                number_of_totalizer_clauses = self.number_of_clauses - number_of_base_clauses
                log_debug(f'Totalizer variables: {number_of_totalizer_variables}')
                log_debug(f'Totalizer clauses: {number_of_totalizer_clauses}')

                self.generate_comparator()
                number_of_comparator_variables = self.number_of_variables - number_of_base_variables - number_of_totalizer_variables
                number_of_comparator_clauses = self.number_of_clauses - number_of_base_clauses - number_of_totalizer_clauses
                log_debug(f'Comparator variables: {number_of_comparator_variables}')
                log_debug(f'Comparator clauses: {number_of_comparator_clauses}')

            assignment = self.solve()
            if assignment:
                log_debug(f'Initial estimation of number_of_nodes = {assignment.number_of_nodes}')
            log_br()

            if assignment and self.N_start == 0:
                self.generate_totalizer()
                number_of_totalizer_variables = self.number_of_variables - number_of_base_variables
                number_of_totalizer_clauses = self.number_of_clauses - number_of_base_clauses
                log_debug(f'Totalizer variables: {number_of_totalizer_variables}')
                log_debug(f'Totalizer clauses: {number_of_totalizer_clauses}')

            while assignment is not None:
                self.best = assignment
                self.N = assignment.number_of_nodes - 1
                log_info(f'Trying with N = {self.N}...')

                self.number_of_variables = number_of_base_variables + number_of_totalizer_variables
                self.number_of_clauses = number_of_base_clauses + number_of_totalizer_clauses
                self.generate_comparator()
                number_of_comparator_variables = self.number_of_variables - number_of_base_variables - number_of_totalizer_variables
                number_of_comparator_clauses = self.number_of_clauses - number_of_base_clauses - number_of_totalizer_clauses
                log_debug(f'Comparator variables: {number_of_comparator_variables}')
                log_debug(f'Comparator clauses: {number_of_comparator_clauses}')

                assignment = self.solve()
                log_br()

            if self.best:
                log_success(f'Best: C={self.C}, K={self.K}, P={self.P}, N={self.best.number_of_nodes}')
            else:
                log_error('Completely UNSAT :c')
            log_br()

        else:  # not is_minimize
            if self.N_start != 0:
                self.N = self.N_start

                self.generate_totalizer()
                number_of_totalizer_variables = self.number_of_variables - number_of_base_variables
                number_of_totalizer_clauses = self.number_of_clauses - number_of_base_clauses
                log_debug(f'Totalizer variables: {number_of_totalizer_variables}')
                log_debug(f'Totalizer clauses: {number_of_totalizer_clauses}')

                self.generate_comparator()
                number_of_comparator_variables = self.number_of_variables - number_of_base_variables - number_of_totalizer_variables
                number_of_comparator_clauses = self.number_of_clauses - number_of_base_clauses - number_of_totalizer_clauses
                log_debug(f'Comparator variables: {number_of_comparator_variables}')
                log_debug(f'Comparator clauses: {number_of_comparator_clauses}')

            assignment = self.solve()
            log_br()

            if assignment:
                log_success(f'number_of_nodes = {assignment.number_of_nodes}')
            else:
                if self.N is None:
                    log_error(f'No solution with C={self.C}, K={self.K}, P={self.P}')
                else:
                    log_error(f'No solution with C={self.C}, K={self.K}, P={self.P}, N={self.N}')
            log_br()

        # in the end
        if self.is_incremental:
            self.solver_process.kill()

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

    def generate_base_reduction(self):
        C = self.C
        K = self.K
        P = self.P

        log_debug(f'Generating base reduction for C={C}, K={K}, P={P}...')
        time_start_base = time.time()

        self.maybe_new_stream(self.get_filename_base())

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
        # Y = tree.Y

        # =-=-=-=-=-=
        #  VARIABLES
        # =-=-=-=-=-=

        # automaton variables
        color = self.declare_array(V, C, with_zero=True)
        transition = self.declare_array(C, K, C, with_zero=True)
        trans_event = self.declare_array(C, K, E)
        output_event = self.declare_array(C, O)
        algorithm_0 = self.declare_array(C, Z)
        algorithm_1 = self.declare_array(C, Z)
        # guards variables
        nodetype = self.declare_array(C, K, P, 4, with_zero=True)
        terminal = self.declare_array(C, K, P, X, with_zero=True)
        child_left = self.declare_array(C, K, P, P, with_zero=True)
        child_right = self.declare_array(C, K, P, P, with_zero=True)
        parent = self.declare_array(C, K, P, P, with_zero=True)
        value = self.declare_array(C, K, P, U)
        child_value_left = self.declare_array(C, K, P, U)
        child_value_right = self.declare_array(C, K, P, U)
        fired_only = self.declare_array(C, K, U)
        not_fired = self.declare_array(C, K, U)
        # bfs variables
        bfs_transition = self.declare_array(C, C)
        bfs_parent = self.declare_array(C, C)
        # bfs_minsymbol = self.declare_array(K, C, C)

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
            now = self.number_of_clauses
            ans = now - so_far_state[0]
            so_far_state[0] = now
            return ans

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # 1. Color constraints
        # 1.0a ALO(color)
        for v in closed_range(1, V):
            ALO(color[v])
        # 1.0b AMO(color)
        for v in closed_range(1, V):
            AMO(color[v])

        # 1.1. Start vertex corresponds to start state
        #   constraint color[1] = 1;
        self.add_clause(color[1][1])

        # 1.2. Only passive vertices (except root) have aux color)
        #   constraint forall (v in 2..V where tree_output_event[v] == O+1) (
        #       color[v] = C+1
        #   );
        for v in closed_range(2, V):
            if tree.output_event[v] == 0:
                self.add_clause(color[v][0])
            else:
                self.add_clause(-color[v][0])

        log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

        # 2. Transition constraints
        # 2.0a ALO(transition)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                ALO(transition[c][k])
        # 2.0a AMO(transition)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                AMO(transition[c][k])

        # 2.0b ALO(trans_event)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                ALO(trans_event[c][k])
        # 2.0b AMO(trans_event)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                AMO(trans_event[c][k])

        # 2.1. (transition + trans_event + fired_only definitions)
        #   constraint forall (v in 2..V where tree_output_event[v] != O+1) (
        #       exists (k in 1..K) (
        #           y[color[tree_previous_active[v]], k] = color[v]
        #           /\ w[color[tree_previous_active[v]], k] = tree_input_event[v]
        #           /\ fired_only[color[tree_previous_active[v]], k, input_nums[v]]
        #       )
        #   );
        for v in closed_range(2, V):
            if tree.output_event[v] != 0:
                for cv in closed_range(0, C):
                    for ctpa in closed_range(1, C):
                        constraint = [-color[v][cv], -color[tree.previous_active[v]][ctpa]]
                        for k in closed_range(1, K):
                            # aux <-> y[ctpa,k,cv] /\ w[ctpa,k,tie[v]] /\ fired_only[ctpa,k,input_number[v]]
                            # x <-> a /\ b /\ c
                            # CNF: (~a ~b ~c x) & (a ~x) & (b ~x) & (c ~x)
                            x1 = transition[ctpa][k][cv]
                            x2 = trans_event[ctpa][k][tree.input_event[v]]
                            x3 = fired_only[ctpa][k][tree.input_number[v]]
                            aux = self.new_variable()
                            self.add_clause(-x1, -x2, -x3, aux)
                            self.add_clause(-aux, x1)
                            self.add_clause(-aux, x2)
                            self.add_clause(-aux, x3)
                            constraint.append(aux)
                        self.add_clause(*constraint)

        # 2.2. Forbid transition self-loops
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                self.add_clause(-transition[c][k][c])

        # 2.3. Forbid transitions to start state
        for c in closed_range(2, C):
            for k in closed_range(1, K):
                self.add_clause(-transition[c][k][1])

        log_debug(f'2. Clauses: {so_far()}', symbol='STAT')

        # 3. Output event constraints
        # 3.0a ALO(output_event)
        for c in closed_range(1, C):
            ALO(output_event[c])
        # 3.0b AMO(output_event)
        for c in closed_range(1, C):
            AMO(output_event[c])

        # 3.1. Start state does INITO
        # self.add_clause(output_event[1, unique_output_events.index('INITO') + 1])
        self.add_clause(output_event[1][tree.output_event[1]])

        # 3.2. Output event is the same as in the tree
        for v in closed_range(2, V):
            if tree.output_event[v] != 0:
                for cv in closed_range(1, C):
                    self.add_clause(-color[v][cv], output_event[cv][tree.output_event[v]])

        log_debug(f'3. Clauses: {so_far()}', symbol='STAT')

        # 4. Algorithm constraints
        # 4.1. Start state does nothing
        #   constraint forall (z in 1..Z) (
        #       d_0[1, z] = 0 /\
        #       d_1[1, z] = 1
        #   );
        for z in closed_range(1, Z):
            self.add_clause(-algorithm_0[1][z])
            self.add_clause(algorithm_1[1][z])

        # 4.2. What to do with zero
        #   constraint forall (v in 2..V, z in 1..Z where tree_output_event[v] != O+1) (
        #       not tree_z[output_nums[tree_previous_active[v]], z] ->
        #           d_0[color[v], z] = tree_z[output_nums[v], z]
        #   );
        for v in closed_range(2, V):
            if tree.output_event[v] != 0:
                for z in closed_range(1, Z):
                    if not tree.unique_output[tree.output_number[tree.previous_active[v]]][z]:
                        for cv in closed_range(1, C):
                            if tree.unique_output[tree.output_number[v]][z]:
                                self.add_clause(-color[v][cv], algorithm_0[cv][z])
                            else:
                                self.add_clause(-color[v][cv], -algorithm_0[cv][z])

        # 4.3. What to do with one
        #   constraint forall (v in 2..V, z in 1..Z where tree_output_event[v] != O+1) (
        #       tree_z[output_nums[tree_previous_active[v]], z] ->
        #           d_1[color[v], z] = tree_z[output_nums[v], z]
        #   );
        for v in closed_range(2, V):
            if tree.output_event[v] != 0:
                for z in closed_range(1, Z):
                    if tree.unique_output[tree.output_number[tree.previous_active[v]]][z]:
                        for cv in closed_range(1, C):
                            if tree.unique_output[tree.output_number[v]][z]:
                                self.add_clause(-color[v][cv], algorithm_1[cv][z])
                            else:
                                self.add_clause(-color[v][cv], -algorithm_1[cv][z])

        log_debug(f'4. Clauses: {so_far()}', symbol='STAT')

        # 5. Firing constraints
        # 5.1. (not_fired definition)
        #   constraint forall (v in 2..V, ctpa in 1..C where tree_output_event[v] == O+1) (
        #       ctpa = color[tree_previous_active[v]] ->
        #           not_fired[ctpa, K, input_nums[v]]
        #   );
        for v in closed_range(2, V):
            if tree.output_event[v] == 0:
                for ctpa in closed_range(1, C):
                    self.add_clause(-color[tree.previous_active[v]][ctpa],
                                    not_fired[ctpa][K][tree.input_number[v]])

        # 5.2. not fired
        #   // part a
        #   constraint forall (c in 1..C, g in 1..U) (
        #       not_fired[c, 1, g] <->
        #           not value[c, 1, 1, g]
        #   );
        #   // part b
        #   constraint forall (c in 1..C, k in 2..K, g in 1..U) (
        #       not_fired[c, k, g] <->
        #           not value[c, k, 1, g]
        #           /\ not_fired[c, k-1, g]
        #   );
        for c in closed_range(1, C):
            for u in closed_range(1, U):
                self.add_clause(-not_fired[c][1][u], -value[c][1][1][u])
                self.add_clause(not_fired[c][1][u], value[c][1][1][u])
                for k in closed_range(2, K):
                    x1 = not_fired[c][k][u]
                    x2 = -value[c][k][1][u]
                    x3 = not_fired[c][k - 1][u]
                    self.add_clause(-x1, x2)
                    self.add_clause(-x1, x3)
                    self.add_clause(x1, -x2, -x3)

        # 5.3. fired_only
        #   // part a
        #   constraint forall (c in 1..C, g in 1..U) (
        #       fired_only[c, 1, g] <-> value[c, 1, 1, g]
        #   );
        #   // part b
        #   constraint forall (c in 1..C, k in 2..K, g in 1..U) (
        #       fired_only[c, k, g] <-> value[c, k, 1, g] /\ not_fired[c, k-1, g]
        #   );
        for c in closed_range(1, C):
            for u in closed_range(1, U):
                self.add_clause(-fired_only[c][1][u], value[c][1][1][u])
                self.add_clause(fired_only[c][1][u], -value[c][1][1][u])
                for k in closed_range(2, K):
                    x1 = fired_only[c][k][u]
                    x2 = value[c][k][1][u]
                    x3 = not_fired[c][k - 1][u]
                    self.add_clause(-x1, x2)
                    self.add_clause(-x1, x3)
                    self.add_clause(x1, -x2, -x3)

        log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        # 6. Guard conditions constraints
        # 6.1a ALO(nodetype)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(nodetype[c][k][p])
        # 6.1b AMO(nodetype)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(nodetype[c][k][p])

        # 6.2a ALO(terminal)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(terminal[c][k][p])
        # 6.2b AMO(terminal)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(terminal[c][k][p])

        # 6.3a ALO(child_left)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(child_left[c][k][p])
        # 6.3b AMO(child_left)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(child_left[c][k][p])

        # 6.4a ALO(child_right)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(child_right[c][k][p])
        # 6.4b AMO(child_right)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(child_right[c][k][p])

        # 6.5a ALO(parent)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(parent[c][k][p])
        # 6.5b AMO(parent)
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(parent[c][k][p])

        log_debug(f'6. Clauses: {so_far()}', symbol='STAT')

        # 7. Extra guard conditions constraints
        # 7.1. Root has no parent
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                self.add_clause(parent[c][k][1][0])

        # 7.2. BFS: typed nodes (except root) have parent with lesser number
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(2, P):
                    self.add_clause(nodetype[c][k][p][4],
                                    *[parent[c][k][p][par] for par in range(1, p)])

        # 7.3. parent<->child relation
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for ch in closed_range(p + 1, P):
                        self.add_clause(-parent[c][k][ch][p], child_left[c][k][p][ch], child_right[c][k][p][ch])
        log_debug(f'7. Clauses: {so_far()}', symbol='STAT')

        # 8. None-type nodes constraints
        # 8.1. None-type nodes have largest numbers
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    self.add_clause(-nodetype[c][k][p][4], nodetype[c][k][p + 1][4])

        # 8.2. None-type nodes have no parent
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    self.add_clause(-nodetype[c][k][p][4], parent[c][k][p][0])

        # 8.3. None-type nodes have no children
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    self.add_clause(-nodetype[c][k][p][4], child_left[c][k][p][0])
                    self.add_clause(-nodetype[c][k][p][4], child_right[c][k][p][0])

        # 8.4. None-type nodes have False value and child_values
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    for u in closed_range(1, U):
                        self.add_clause(-nodetype[c][k][p][4], -value[c][k][p][u])
                        self.add_clause(-nodetype[c][k][p][4], -child_value_left[c][k][p][u])
                        self.add_clause(-nodetype[c][k][p][4], -child_value_right[c][k][p][u])

        log_debug(f'8. Clauses: {so_far()}', symbol='STAT')

        # 9. Terminals constraints
        # 9.1. Only terminals have associated terminal variables
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    self.add_clause(-nodetype[c][k][p][0], -terminal[c][k][p][0])
                    self.add_clause(nodetype[c][k][p][0], terminal[c][k][p][0])

        # 9.2. Terminals have no children
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    self.add_clause(-nodetype[c][k][p][0], child_left[c][k][p][0])
                    self.add_clause(-nodetype[c][k][p][0], child_right[c][k][p][0])

        # 9.3. Terminals have value from associated input variable
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    for x in closed_range(1, X):
                        for u in closed_range(1, U):
                            x1 = terminal[c][k][p][x]
                            x2 = value[c][k][p][u]
                            if tree.unique_input[u][x]:
                                self.add_clause(-x1, x2)
                            else:
                                self.add_clause(-x1, -x2)

        log_debug(f'9. Clauses: {so_far()}', symbol='STAT')

        # 10. AND/OR nodes constraints
        # 10.0. AND/OR nodes cannot have numbers P-1 or P
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                if P >= 1:
                    self.add_clause(-nodetype[c][k][P][1])
                    self.add_clause(-nodetype[c][k][P][2])
                if P >= 2:
                    self.add_clause(-nodetype[c][k][P - 1][1])
                    self.add_clause(-nodetype[c][k][P - 1][2])

        # 10.1. AND/OR: left child has greater number
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    _cons = [child_left[c][k][p][ch] for ch in closed_range(p + 1, P - 1)]
                    self.add_clause(-nodetype[c][k][p][1], *_cons)
                    self.add_clause(-nodetype[c][k][p][2], *_cons)

        # 10.2. AND/OR: right child is adjacent (+1) to left
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 1, P - 1):
                        for nt in [1, 2]:
                            self.add_clause(-nodetype[c][k][p][nt],
                                            -child_left[c][k][p][ch],
                                            child_right[c][k][p][ch + 1])

        # 10.3. AND/OR: children`s parents
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 1, P - 1):
                        for nt in [1, 2]:
                            _cons = [-nodetype[c][k][p][nt], -child_left[c][k][p][ch]]
                            self.add_clause(*_cons, parent[c][k][ch][p])
                            self.add_clause(*_cons, parent[c][k][ch + 1][p])

        # 10.4a AND/OR: child_value_left is a value of left child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 1, P - 1):
                        for u in closed_range(1, U):
                            for nt in [1, 2]:
                                x1 = nodetype[c][k][p][nt]
                                x2 = child_left[c][k][p][ch]
                                x3 = child_value_left[c][k][p][u]
                                x4 = value[c][k][ch][u]
                                self.add_clause(-x1, -x2, -x3, x4)
                                self.add_clause(-x1, -x2, x3, -x4)

        # 10.4b AND/OR: child_value_right is a value of right child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 2, P):
                        for u in closed_range(1, U):
                            for nt in [1, 2]:
                                x1 = nodetype[c][k][p][nt]
                                x2 = child_right[c][k][p][ch]
                                x3 = child_value_right[c][k][p][u]
                                x4 = value[c][k][ch][u]
                                self.add_clause(-x1, -x2, -x3, x4)
                                self.add_clause(-x1, -x2, x3, -x4)

        # 10.5a AND: value is calculated as a conjunction of children
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for u in closed_range(1, U):
                        x1 = nodetype[c][k][p][1]
                        x2 = value[c][k][p][u]
                        x3 = child_value_left[c][k][p][u]
                        x4 = child_value_right[c][k][p][u]
                        self.add_clause(-x1, x2, -x3, -x4)
                        self.add_clause(-x1, -x2, x3)
                        self.add_clause(-x1, -x2, x4)

        # 10.5b OR: value is calculated as a disjunction of children
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for u in closed_range(1, U):
                        x1 = nodetype[c][k][p][2]
                        x2 = value[c][k][p][u]
                        x3 = child_value_left[c][k][p][u]
                        x4 = child_value_right[c][k][p][u]
                        self.add_clause(-x1, -x2, x3, x4)
                        self.add_clause(-x1, x2, -x3)
                        self.add_clause(-x1, x2, -x4)

        log_debug(f'10. Clauses: {so_far()}', symbol='STAT')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # 11. NOT nodes constraints
        # 11.0. NOT nodes cannot have number P
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                self.add_clause(-nodetype[c][k][P][3])

        # 11.1. NOT: left child has greater number
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    _cons = [child_left[c][k][p][ch] for ch in closed_range(p + 1, P)]
                    self.add_clause(-nodetype[c][k][p][3], *_cons)

        # 11.2. NOT: no right child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    self.add_clause(-nodetype[c][k][p][3], child_right[c][k][p][0])

        # 11.3. NOT: child`s parents
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for ch in closed_range(p + 1, P):
                        self.add_clause(-nodetype[c][k][p][3],
                                        -child_left[c][k][p][ch],
                                        parent[c][k][ch][p])

        # 11.4a NOT: child_value_left is a value of left child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for ch in closed_range(p + 1, P):
                        for u in closed_range(1, U):
                            x1 = nodetype[c][k][p][3]
                            x2 = child_left[c][k][p][ch]
                            x3 = child_value_left[c][k][p][u]
                            x4 = value[c][k][ch][u]
                            self.add_clause(-x1, -x2, -x3, x4)
                            self.add_clause(-x1, -x2, x3, -x4)

        # 11.4b NOT: child_value_right is False
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for u in closed_range(1, U):
                        self.add_clause(-nodetype[c][k][p][3], -child_value_right[c][k][p][u])

        # 11.5. NOT: value is calculated as a negation of child
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for u in closed_range(1, U):
                        x1 = nodetype[c][k][p][3]
                        x2 = value[c][k][p][u]
                        x3 = child_value_left[c][k][p][u]
                        self.add_clause(-x1, -x2, -x3)
                        self.add_clause(-x1, x2, x3)

        log_debug(f'11. Clauses: {so_far()}', symbol='STAT')

        # 12. BFS constraints
        # 12.1. F_t
        for i in closed_range(1, C):
            # FIXME: j should iterate from 1 to C, excluding i
            for j in closed_range(i + 1, C):
                # t_ij <=> OR_k(transition_ikj)
                aux = bfs_transition[i][j]
                rhs = []
                for k in closed_range(1, K):
                    xi = transition[i][k][j]
                    self.add_clause(-xi, aux)
                    rhs.append(xi)
                self.add_clause(*rhs, -aux)

        # 12.2. F_p
        for i in closed_range(1, C):
            for j in closed_range(i + 1, C):
                # p_ji <=> t_ij & AND_[k<i](~t_kj)
                aux = bfs_parent[j][i]
                xi = bfs_transition[i][j]
                self.add_clause(xi, -aux)
                rhs = [xi]
                for k in closed_range(1, i - 1):
                    xi = -bfs_transition[k][j]  # negated in formula
                    self.add_clause(xi, -aux)
                    rhs.append(xi)
                self.add_clause(*[-xi for xi in rhs], aux)

        # 12.3. F_ALO(p)
        for j in closed_range(2, C):
            self.add_clause(*[bfs_parent[j][i] for i in closed_range(1, j - 1)])

        # 12.4. F_BFS(p)
        for k in closed_range(1, C):
            for i in closed_range(k + 1, C):
                for j in closed_range(i + 1, C - 1):
                    # p_ji => ~p_j+1,k
                    self.add_clause(-bfs_parent[j][i], -bfs_parent[j + 1][k])

        log_debug(f'12. Clauses: {so_far()}', symbol='STAT')

        # Declare any ad-hoc you like
        # AD-HOCs

        # adhoc-1
        #   constraint forall (c in 1..C, k in 1..K) (
        #       y[c, k] = C+1 <->
        #           nodetype[c, k, 1] = 4
        #   );
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                self.add_clause(-transition[c][k][0], nodetype[c][k][1][4])
                self.add_clause(transition[c][k][0], -nodetype[c][k][1][4])

        # adhoc-2
        #   constraint forall (c in 1..C, k in 1..K-1) (
        #       y[c, k] = C+1 ->
        #           y[c, k+1] = C+1
        #   );
        for c in closed_range(1, C):
            for k in closed_range(1, K - 1):
                self.add_clause(-transition[c][k][0], transition[c][k + 1][0])

        # adhoc-4
        #   constraint forall (c in 1..C, k in 1..K, g in 1..U) (
        #       fired_only[c, k, g] ->
        #           nodetype[c, k, 1] != 4
        #   );
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for u in closed_range(1, U):
                    self.add_clause(-fired_only[c][k][u], -nodetype[c][k][1][4])

        log_debug(f'A. Clauses: {so_far()}', symbol='STAT')

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.maybe_close_stream(self.get_filename_base())

        self.reduction = self.Reduction(
            color=color,
            transition=transition,
            trans_event=trans_event,
            output_event=output_event,
            algorithm_0=algorithm_0,
            algorithm_1=algorithm_1,
            nodetype=nodetype,
            terminal=terminal,
            child_left=child_left,
            child_right=child_right,
            parent=parent,
            value=value,
            child_value_left=child_value_left,
            child_value_right=child_value_right,
            fired_only=fired_only,
            not_fired=not_fired,
            totalizer=None
        )

        log_debug(f'Done generating base reduction ({self.number_of_variables} variables, {self.number_of_clauses} clauses) in {time.time() - time_start_base:.2f} s')

    def generate_totalizer(self):
        log_debug(f'Generating totalizer...')
        time_start_totalizer = time.time()
        _nv = self.number_of_variables
        _nc = self.number_of_clauses

        self.maybe_new_stream(self.get_filename_totalizer())

        _E = [-self.reduction.nodetype[c][k][p][4]
              for c in closed_range(1, self.C)
              for k in closed_range(1, self.K)
              for p in closed_range(1, self.P)]  # set of input variables
        _L = []  # set of linking variables

        q = deque([e] for e in _E)
        while len(q) != 1:
            a = q.popleft()  # 0-based
            b = q.popleft()  # 0-based

            m1 = len(a)
            m2 = len(b)
            m = m1 + m2

            r = [self.new_variable() for _ in range(m)]  # 0-based

            if len(q) != 0:
                _L.extend(r)

            for alpha in closed_range(m1):
                for beta in closed_range(m2):
                    sigma = alpha + beta

                    if sigma == 0:
                        C1 = None
                    elif alpha == 0:
                        C1 = [-b[beta - 1], r[sigma - 1]]
                    elif beta == 0:
                        C1 = [-a[alpha - 1], r[sigma - 1]]
                    else:
                        C1 = [-a[alpha - 1], -b[beta - 1], r[sigma - 1]]

                    if sigma == m:
                        C2 = None
                    elif alpha == m1:
                        C2 = [b[beta], -r[sigma]]
                    elif beta == m2:
                        C2 = [a[alpha], -r[sigma]]
                    else:
                        C2 = [a[alpha], b[beta], -r[sigma]]

                    if C1 is not None:
                        self.add_clause(*C1)
                    if C2 is not None:
                        self.add_clause(*C2)

            q.append(r)

        _S = q.pop()  # set of output variables
        assert len(_E) == len(_S)

        self.maybe_close_stream(self.get_filename_totalizer())

        self.reduction = self.reduction._replace(totalizer=_S)  # Note: totalizer is 0-based!

        log_debug(f'Done generating totalizer ({self.number_of_variables-_nv} variables, {self.number_of_clauses-_nc} clauses) in {time.time() - time_start_totalizer:.2f} s')

    def generate_comparator(self):
        log_debug(f'Generating comparator...')
        time_start_cardinality = time.time()
        _nc = self.number_of_clauses

        self.maybe_new_stream(self.get_filename_comparator())

        if self.is_incremental and self.N_defined is not None:
            N_max = self.N_defined
        else:
            N_max = self.C * self.K * self.P

        # sum(E) <= N   <=>   sum(E) < N + 1
        for i in reversed(closed_range(self.N + 1, N_max)):
            self.add_clause(-self.reduction.totalizer[i - 1])  # Note: totalizer is 0-based!

        if self.is_incremental:
            self.N_defined = self.N

        self.maybe_close_stream(self.get_filename_comparator())

        log_debug(f'Done generating comparator ({self.number_of_clauses-_nc} clauses) in {time.time() - time_start_cardinality:.2f} s')

    def solve(self):
        log_info('Solving...')
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

        nodetype = wrapper_int(self.reduction.nodetype)
        assignment = self.Assignment(
            color=wrapper_int(self.reduction.color),
            transition=wrapper_int(self.reduction.transition),
            trans_event=wrapper_int(self.reduction.trans_event),
            output_event=wrapper_int(self.reduction.output_event),
            algorithm_0=wrapper_algo(self.reduction.algorithm_0),
            algorithm_1=wrapper_algo(self.reduction.algorithm_1),
            nodetype=nodetype,
            terminal=wrapper_int(self.reduction.terminal),
            child_left=wrapper_int(self.reduction.child_left),
            child_right=wrapper_int(self.reduction.child_right),
            parent=wrapper_int(self.reduction.parent),
            value=wrapper_bool(self.reduction.value),
            child_value_left=wrapper_bool(self.reduction.child_value_left),
            child_value_right=wrapper_bool(self.reduction.child_value_right),
            fired_only=wrapper_bool(self.reduction.fired_only),
            not_fired=wrapper_bool(self.reduction.not_fired),
            number_of_nodes=sum(nodetype[c][k][p] != 4
                                for c in closed_range(1, self.C)
                                for k in closed_range(1, self.K)
                                for p in closed_range(1, self.P))
        )

        log_debug(f'Done building assignment in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def maybe_n(self):
        return f'_N{self.N}' if self.N else ''

    def get_filename_prefix(self):
        return f'{self.filename_prefix}_C{self.C}_K{self.K}_P{self.P}'

    def get_filename_base(self):
        return f'{self.get_filename_prefix()}_base.dimacs'

    def get_filename_totalizer(self):
        return f'{self.get_filename_prefix()}_totalizer.dimacs'

    def get_filename_comparator(self):
        return f'{self.get_filename_prefix()}{self.maybe_n()}_comparator.dimacs'

    def get_filename_header(self):
        return f'{self.get_filename_prefix()}{self.maybe_n()}_header.dimacs'

    def get_filename_merged(self):
        return f'{self.get_filename_prefix()}{self.maybe_n()}_merged.dimacs'

    def get_filenames(self):
        if self.N:
            return ' '.join((self.get_filename_header(),
                             self.get_filename_base(),
                             self.get_filename_totalizer(),
                             self.get_filename_comparator()))
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
