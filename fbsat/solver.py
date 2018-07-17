from io import StringIO
import time
import regex as re
import subprocess
import shutil
from collections import deque

from .utils import *
from .printers import *

__all__ = ['Solver']


class Solver:

    def __init__(self, cmd, write_strategy='StringIO', filename_prefix=None):
        assert cmd is not None
        self.cmd = cmd
        self.write_strategy = write_strategy
        if write_strategy == 'direct':
            raise NotImplementedError
            assert filename_prefix is not None
            self.filename = f'{filename_prefix}.cnf'
            self.stream = open(self.filename, 'w+')
        elif write_strategy == 'StringIO':
            self.stream = StringIO()
        self.number_of_variables = 0
        self.number_of_clauses = 0

    def new_variable(self):
        self.number_of_variables += 1
        return self.number_of_variables

    def add_clause(self, *vs):
        self.number_of_clauses += 1
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
        elif n == 5:
            return [None] + [[None] + [[None] + [[None] + [last()
                                                           for _ in closed_range(1, dims[3])]
                                                 for _ in closed_range(1, dims[2])]
                                       for _ in closed_range(1, dims[1])]
                             for _ in closed_range(1, dims[0])]
        else:
            raise ValueError(f'unsupported number of dimensions ({n})')

    def ALO(self, data):
        lower = 1 if data[0] is None else 0
        self.add_clause(*data[lower:])

    def AMO(self, data):
        lower = 1 if data[0] is None else 0
        upper = len(data) - 1
        for a in range(lower, upper):
            for b in closed_range(a + 1, upper):
                self.add_clause(-data[a], -data[b])

    def imply(self, lhs, rhs):
        """lhs => rhs"""
        self.add_clause(-lhs, rhs)

    # TODO: imply_and, imply_or

    def iff(self, lhs, rhs):
        """lhs <=> rhs"""
        self.imply(lhs, rhs)
        self.imply(rhs, lhs)

    def iff_and(self, lhs, rhs):
        """lhs <=> AND(rhs)"""
        rhs = tuple(rhs)
        for x in rhs:
            self.add_clause(x, -lhs)
        self.add_clause(lhs, *(-x for x in rhs))

    def iff_or(self, lhs, rhs):
        """lhs <=> OR(rhs)"""
        rhs = tuple(rhs)
        for x in rhs:
            self.add_clause(-x, lhs)
        self.add_clause(-lhs, *rhs)

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

    def solve(self):
        log_debug(f'Solving with "{self.cmd}"...')
        time_start_solve = time.time()
        with subprocess.Popen(self.cmd, shell=True, universal_newlines=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p:
            self.stream.seek(0)
            shutil.copyfileobj(self.stream, p.stdin)
            p.stdin.close()

            is_sat = None
            raw_assignment = [None]  # 1-based
            pattern = re.compile(r'-?\d+')
            for line in map(str.rstrip, p.stdout):
                if line == 's SATISFIABLE':
                    is_sat = True
                elif line.startswith('v'):
                    for m in pattern.finditer(line):
                        value = int(m.group(0))
                        if value == 0:
                            break
                        assert abs(value) == len(raw_assignment)
                        raw_assignment.append(value)
                elif line == 's UNSATISFIABLE':
                    is_sat = False

        if is_sat is None:
            log_error(f'UNSAT or ERROR in {time.time() - time_start_solve:.2f} s')
        elif is_sat:
            log_success(f'SAT in {time.time() - time_start_solve:.2f} s')
            return raw_assignment
        else:
            log_error(f'UNSAT in {time.time() - time_start_solve:.2f} s')
