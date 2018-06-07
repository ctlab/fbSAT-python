__all__ = ('Instance', )

import os
import time
import regex
import shutil
import pickle
import tempfile
import itertools
import subprocess
from io import StringIO
from collections import deque, namedtuple

from .efsm import *
from .utils import *
from .printers import *

VARIABLES = 'color transition output_event algorithm_0 algorithm_1'


class Instance:

    Reduction = namedtuple('Reduction', VARIABLES + ' tr totalizers')
    Assignment = namedtuple('Assignment', VARIABLES + ' C K')

    def __init__(self, *, scenario_tree, C=None, K=None, C_max=None, is_minimize=True, is_incremental=False, sat_solver=None, sat_isolver=None, filename_prefix='', write_strategy='StringIO', is_reuse=False):
        assert write_strategy in ('direct', 'tempfile', 'StringIO')

        if is_incremental:
            assert sat_isolver is not None, "You need to specify incremental sat-solver using `--sat-isolver` option"
        else:
            assert sat_solver is not None, "You need to specify sat-solver using `--sat-solver` option"

        if is_minimize and K is not None:
            log_warn(f'Ignoring K={K} because of minimization')

        if not is_minimize:
            assert C is not None, "You need to specify C if not minimizing"

        self.C_given = C
        self.K_given = K
        self.C_max = C_max
        self.scenario_tree = scenario_tree
        self.is_minimize = is_minimize
        self.is_incremental = is_incremental
        self.sat_solver = sat_solver
        self.sat_isolver = sat_isolver
        self.filename_prefix = filename_prefix
        self.write_strategy = write_strategy
        self.is_reuse = is_reuse
        self.stream = None

    def run(self):
        if self.is_minimize:
            self.run_minimize()
        else:
            self.run_once()

        if self.best:
            efsm = self.build_efsm(self.best)

            filename_automaton = f'{self.filename_prefix}_automaton'
            with open(filename_automaton, 'wb') as f:
                pickle.dump(efsm, f, pickle.HIGHEST_PROTOCOL)

            filename_gv = f'{self.filename_prefix}_C{self.best.C}{self.maybe_k(self.best.K)}_efsm.gv'
            os.makedirs(os.path.dirname(filename_gv), exist_ok=True)
            efsm.write_gv(filename_gv)

            output_format = 'svg'
            cmd = f'dot -T{output_format} {filename_gv} -O'
            log_debug(cmd, symbol='$')
            os.system(cmd)

            efsm.verify(self.scenario_tree)

    def run_minimize(self):
        if self.C_given is None:  # C unspecified -> iterate over C
            if self.C_max is not None:
                C_iter = closed_range(1, self.C_max)
            else:
                C_iter = itertools.count(1)
                C_iter = itertools.islice(C_iter, 20)  # FIXME: stub

            for C in C_iter:
                log_info(f'Trying C = {C}')
                self.run_minimize_with(C)
                if self.best:
                    log_success(f'Best: C={self.best.C}, K={self.best.K}')
                    log_br()
                    break
            else:  # C_iter exhausted
                log_error('CAN\'T FIND ANYTHING :C')

        else:  # C specified -> use it
            C = self.C_given
            log_info(f'Using specified C = {C}')
            self.run_minimize_with(C)
            if self.best:
                log_success(f'Best: C={self.best.C}, K={self.best.K}')
            else:
                log_error(f'UNSAT for C = {C}')
            log_br()

    def run_minimize_with(self, C):
        self.C = C
        self.K = None
        self.best = None
        self.number_of_variables = 0
        self.number_of_clauses = 0
        self.generate_base()

        assignment = self.solve()
        log_br()

        if assignment:  # SAT, start minimizing K
            self.best = assignment

            self.generate_pre()
            _nv = self.number_of_variables  # base+pre variables
            _nc = self.number_of_clauses    # base+pre clauses

            if self.is_incremental:
                self.isolver_process = subprocess.Popen(self.sat_isolver, shell=True, universal_newlines=True,
                                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                self.feed_isolver(self.get_filename_base())
                self.feed_isolver(self.get_filename_pre())

                self.K_defined = None

                # Note: K=C-1 == unconstrained base reduction
                for K in reversed(closed_range(0, C - 2)):
                    log_info(f'Trying K = {K}')
                    self.K = K
                    self.generate_cardinality_isolver()

                    assignment = self.solve_incremental()
                    log_br()

                    if assignment is None:  # UNSAT, the answer is last solution (self.best)
                        break

                    self.best = assignment  # update best found solution
                else:
                    log_warn('Reached K=0, weird...')

                self.isolver_process.kill()
            else:  # non-incrementaly
                # Note: K=C-1 == unconstrained base reduction
                for K in closed_range(0, C - 2):
                    log_info(f'Trying K = {K}')
                    self.K = K
                    self.number_of_variables = _nv
                    self.number_of_clauses = _nc
                    self.generate_cardinality()

                    assignment = self.solve()
                    log_br()

                    if assignment:  # SAT, this is the answer
                        self.best = assignment
                        break
                else:
                    log_warn(f'Reached K=C-2={C-2} and did not find any solution, weird...')

        return self.best

    def run_once(self):
        self.C = self.C_given
        self.K = self.K_given

        self.number_of_variables = 0
        self.number_of_clauses = 0
        self.generate_base()

        if self.K is not None:
            self.generate_pre()
            self.generate_cardinality()

        assignment = self.solve()
        log_br()
        self.best = assignment

        if assignment:  # SAT
            if self.K is None:
                log_success(f'SAT for C={self.C}')
            else:
                log_success(f'SAT for C={self.C}, K={self.K}')
        else:  # UNSAT
            if self.K is None:
                log_error(f'UNSAT for C={self.C}')
            else:
                log_error(f'UNSAT for C={self.C}, K={self.K}')
        log_br()

    def maybe_new_stream(self, filename):
        assert self.stream is None, "Please, be careful creating new stream without closing previous one"
        if self.is_reuse and os.path.exists(filename):
            log_debug(f'Reusing <{filename}>')
            self.stream = None
        elif self.write_strategy == 'direct':
            self.stream = open(filename, 'w')
        elif self.write_strategy == 'tempfile':
            self.stream = tempfile.NamedTemporaryFile('w', delete=False)
        elif self.write_strategy == 'StringIO':
            self.stream = StringIO()

    def maybe_close_stream(self, filename):
        if self.stream is not None:
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
            self.stream = None

    def feed_isolver(self, filename):
        assert self.is_incremental, "Do not feed isolver when not solving incrementaly"
        log_debug(f'Feeding isolver from <{filename}>...')
        with open(filename) as f:
            shutil.copyfileobj(f, self.isolver_process.stdin)

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

    def get_totalizer(self, _E):
        # _E is a set of input variables
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

        return _S

    def generate_base(self):
        C = self.C
        assert self.number_of_variables == 0
        assert self.number_of_clauses == 0

        log_debug(f'Generating base reduction for C = {C}...')
        time_start_base = time.time()
        filename = self.get_filename_base()
        self.maybe_new_stream(filename)

        # =-=-=-=-=-=
        #  CONSTANTS
        # =-=-=-=-=-=

        tree = self.scenario_tree
        V = tree.V
        E = tree.E
        O = tree.O
        X = tree.X  #
        Z = tree.Z
        U = tree.U
        Y = tree.Y  #

        log_debug(f'V = {V}')
        log_debug(f'E = {E}')
        log_debug(f'O = {O}')
        log_debug(f'X = {X}')
        log_debug(f'Z = {Z}')
        log_debug(f'U = {U}')
        log_debug(f'Y = {Y}')

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
        # 1.0. ALO/AMO(color)
        for v in closed_range(1, V):
            ALO(color[v])
            AMO(color[v])

        # 1.1. Root corresponds to start state
        self.add_clause(color[1][1])

        # 1.2. Color definition
        for v in closed_range(2, V):
            for c in closed_range(1, C):
                if tree.output_event[v] == 0:
                    # IF toe[v]=0 THEN color[v,c] <=> color[parent[v],c]
                    self.add_clause(-color[v][c], color[tree.parent[v]][c])
                    self.add_clause(color[v][c], -color[tree.parent[v]][c])
                else:
                    # IF toe[v]!=0 THEN not (color[v,c] and color[parent[v],c])
                    self.add_clause(-color[v][c], -color[tree.parent[v]][c])

        log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

        # 2. Transition constraints
        # 2.0. ALO/AMO(transition)
        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    ALO(transition[c][e][u])
                    AMO(transition[c][e][u])

        # 2.1. Transition definition
        for i in closed_range(1, C):
            for j in closed_range(1, C):
                if i == j:
                    continue
                for e in closed_range(1, E):
                    for u in closed_range(1, U):
                        # transition[i,e,u,j] <=> OR_{v|...}( color[parent[v],i] & color[v,j] )
                        t = transition[i][e][u][j]
                        rhs = []
                        for v in closed_range(2, V):
                            if tree.input_event[v] == e and tree.input_number[v] == u and tree.output_event[v] != 0:
                                # aux <=> color[parent[v],i] & color[v,j]
                                aux = self.new_variable()
                                self.add_clause(aux, -color[tree.parent[v]][i], -color[v][j])
                                self.add_clause(color[tree.parent[v]][i], -aux)
                                self.add_clause(color[v][j], -aux)

                                self.add_clause(t, -aux)
                                rhs.append(aux)
                        self.add_clause(-t, *rhs)

        # 2.2. Self-loops
        for v in closed_range(2, V):
            for c in closed_range(1, C):
                # IF toe[v]=0 THEN color[v,c] => transition[c,tie[v],tin[v],c]
                if tree.output_event[v] == 0:
                    self.add_clause(-color[v][c], transition[c][tree.input_event[v]][tree.input_number[v]][c])

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
        # 4.0. ALO/AMO(output_event)
        for c in closed_range(1, C):
            ALO(output_event[c])
            AMO(output_event[c])

        # 4.1. Start state does INITO (root's output event)
        self.add_clause(output_event[1][tree.output_event[1]])

        # 4.2. Output event is the same as in the tree
        for v in closed_range(2, V):
            o = tree.output_event[v]
            for c in closed_range(1, C):
                if o == 0:
                    # IF toe[v]=0 THEN color[v,c] => output_event[c,toe[tpa[v]]]
                    self.add_clause(-color[v][c], output_event[c][tree.output_event[tree.previous_active[v]]])
                else:
                    # IF toe[v]!=0 THEN color[v,c] => output_event[c,toe[v]]
                    self.add_clause(-color[v][c], output_event[c][o])

        log_debug(f'4. Clauses: {so_far()}', symbol='STAT')

        # TODO: 5. BFS constraints

        # log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.maybe_close_stream(filename)
        self.reduction = self.Reduction(
            color=color,
            transition=transition,
            output_event=output_event,
            algorithm_0=algorithm_0,
            algorithm_1=algorithm_1,
            tr=None,
            totalizers=None
        )

        log_debug(f'Done generating base reduction ({self.number_of_variables} variables, {self.number_of_clauses} clauses) in {time.time() - time_start_base:.2f} s')

    def generate_pre(self):
        C = self.C

        log_debug(f'Generating transitions existance and pre-cardinality constraints for C = {C}...')
        time_start_pre = time.time()
        _nv = self.number_of_variables
        _nc = self.number_of_clauses
        filename = self.get_filename_pre()
        self.maybe_new_stream(filename)

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
        # nut = self.declare_array(C, C, C, with_zero=True)

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

        # Transitions existance
        for i in closed_range(1, C):
            for j in closed_range(1, C):
                if i == j:
                    continue
                # tr_{i,j} <=> OR_{e,u}(transition_{i,e,u,j})
                rhs = []
                for e in closed_range(1, E):
                    for u in closed_range(1, U):
                        t = transition[i][e][u][j]
                        self.add_clause(-t, tr[i][j])
                        rhs.append(t)
                self.add_clause(*rhs, -tr[i][j])

        # Pretend that self-loops do not exist
        for c in closed_range(1, C):
            self.add_clause(-tr[c][c])

        totalizers = [None] + [self.get_totalizer(tr[i][1:]) for i in closed_range(1, C)]

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.maybe_close_stream(filename)
        self.reduction = self.reduction._replace(tr=tr, totalizers=totalizers)

        log_debug(f'Done generating pre ({self.number_of_variables-_nv} variables, {self.number_of_clauses-_nc} clauses) in {time.time() - time_start_pre:.2f} s')

    def generate_cardinality(self):
        C = self.C
        K = self.K

        log_debug(f'Generating transitions cardinality for K={K}...')
        time_start_cardinality = time.time()
        _nc = self.number_of_clauses
        filename = self.get_filename_cardinality()
        self.maybe_new_stream(filename)

        K_max = C

        # Transitions cardinality (leq K): "forbid all states to have K+1 or more transitions"
        for c in closed_range(1, C):
            for k in reversed(closed_range(K + 1, K_max)):
                self.add_clause(-self.reduction.totalizers[c][k - 1])  # Note: each totalizer is 0-based!

        self.maybe_close_stream(filename)

        log_debug(f'Done generating transitions cardinality ({self.number_of_clauses-_nc} clauses) in {time.time() - time_start_cardinality:.2f} s')

    def generate_cardinality_isolver(self):
        C = self.C
        K = self.K

        log_debug(f'Generating transitions cardinality for K={K} and feeding it to isolver...')
        time_start_cardinality = time.time()
        _nc = self.number_of_clauses
        self.stream = self.isolver_process.stdin

        if self.K_defined is not None:
            K_max = self.K_defined
        else:
            K_max = C

        for c in closed_range(1, C):
            for k in reversed(closed_range(K + 1, K_max)):
                self.add_clause(-self.reduction.totalizers[c][k - 1])  # Note: each totalizer is 0-based!

        self.K_defined = K
        self.stream = None

        log_debug(f'Done feeding cardinality ({self.number_of_clauses-_nc} clauses) to isolver in {time.time() - time_start_cardinality:.2f} s')

    def solve(self):
        log_info(f'Solving...')
        time_start_solve = time.time()

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

    def solve_incremental(self):
        log_info(f'Solving incrementaly...')
        time_start_solve = time.time()

        p = self.isolver_process
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
            C=self.C,
            K=self.K,
        )

        # ===========
        # log_debug('color toe')
        # for v in closed_range(1, len(self.scenario_tree)):
        #     log_debug(f'{assignment.color[v]: ^5} {self.scenario_tree.output_event[v]: ^3}')
        # ===========

        # ===============
        if self.reduction.tr is not None:
            C = self.C
            tr = wrapper_bool(self.reduction.tr)
            log_debug('transition existance:')
            for i in closed_range(1, C):
                log_debug(f' >  {i} -> {[j for j in closed_range(1, C) if tr[i][j]]}', symbol=None)

            max_K = max(len([j for j in closed_range(1, C) if tr[i][j]]) for i in closed_range(1, C))
            log_debug(f'Max K: {max_K}')
            if max_K < self.K:
                log_warn(f'max_K({max_K}) < self.K({self.K}), consider using K = {max_K}')
            assert max_K <= self.K
        # ===============

        log_debug(f'Done building assignment in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def maybe_k(self, K='default'):
        if K == 'default':
            K = self.K
        return f'_K{K}' if K is not None else ''

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
            return (self.get_filename_header(),
                    self.get_filename_base(),
                    self.get_filename_pre(),
                    self.get_filename_cardinality())
        else:
            return (self.get_filename_header(),
                    self.get_filename_base())

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
        cmd_cat = f'cat {" ".join(self.get_filenames())} > {filename}'
        log_debug(cmd_cat, symbol='$')
        subprocess.run(cmd_cat, shell=True)
        # Replace cat with shutil module:
        # with open(filename, 'w') as merged:
        #     for fn in self.get_filenames():
        #         with open(fn) as f:
        #             shutil.copyfileobj(f, merged)

    def build_efsm(self, assignment):
        log_debug('Building EFSM...')
        time_start_build = time.time()

        tree = self.scenario_tree
        C = assignment.C
        E = tree.E
        U = tree.U
        input_events = tree.input_events  # [1..E]:InputEvent::str
        output_events = tree.output_events  # [1..O]:OutputEvent::str
        unique_input = tree.unique_input  # [1..U, 1..X]:Bool

        efsm = EFSM()
        for c in closed_range(1, C):
            efsm.add_state(c,
                           output_events[assignment.output_event[c] - 1],
                           assignment.algorithm_0[c],
                           assignment.algorithm_1[c])
        efsm.initial_state = 1

        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for u in closed_range(1, U):
                    dest = assignment.transition[c][e][u]
                    if dest != c:
                        guard = Guard.from_input(unique_input[u])
                        # log_debug(f'GUARD from {c} to {dest} by e={e}, u={u}: {guard}')
                        efsm.add_transition(c, dest, input_events[e - 1], guard)
                    # else:
                    #     log_debug(f'No transition from {c} for e={e}, u={u}')

        # =======================
        efsm.pprint()
        # =======================

        log_debug(f'Done building EFSM with {efsm.number_of_states} states, {efsm.number_of_transitions} transitions and {efsm.number_of_nodes} nodes in {time.time() - time_start_build:.2f} s')
        return efsm
