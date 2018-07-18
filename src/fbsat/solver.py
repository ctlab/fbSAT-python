import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections import deque
from io import StringIO

from .printers import log_debug, log_error, log_success
from .utils import closed_range

__all__ = ['StreamSolver']


class Solver(ABC):

    @abstractmethod
    def new_variable(self):
        pass

    @abstractmethod
    def add_clause(self, *vs):
        pass

    @abstractmethod
    def solve(self):
        pass

    def declare_array(self, *dims, with_zero=False):
        if len(dims) == 1:
            if with_zero:
                return [self.new_variable() for _ in closed_range(0, dims[0])]
            else:
                return [None] + [self.new_variable() for _ in closed_range(1, dims[0])]
        return [None] + [self.declare_array(*dims[1:], with_zero=with_zero) for _ in range(dims[0])]

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


class StreamSolver(Solver):

    def __init__(self, cmd, filename_prefix=None):
        self.cmd = cmd
        self.filename_prefix = filename_prefix
        self.stream = StringIO()
        self.number_of_variables = 0
        self.number_of_clauses = 0

    def new_variable(self):
        self.number_of_variables += 1
        return self.number_of_variables

    def add_clause(self, *vs):
        self.number_of_clauses += 1
        self.stream.write(' '.join(map(str, vs)) + ' 0\n')

    def solve(self):
        log_debug(f'Solving with "{self.cmd}"...')
        time_start_solve = time.time()
        is_sat = None
        raw_assignment = [None]  # 1-based

        with subprocess.Popen(self.cmd, shell=True, universal_newlines=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p:
            self.stream.seek(0)
            shutil.copyfileobj(self.stream, p.stdin)
            p.stdin.close()

            for line in map(str.rstrip, p.stdout):
                if line == 's SATISFIABLE':
                    is_sat = True
                elif line.startswith('v '):
                    for value in map(int, line[2:].split()):
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