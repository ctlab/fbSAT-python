import os
import time
from collections import namedtuple
from functools import partial

from . import Task
from ..efsm import ParseTreeGuard
from ..printers import log_br, log_debug, log_error, log_info, log_success
from ..solver import FileSolver, IncrementalSolver, StreamSolver
from ..utils import NotBool, closed_range, parse_raw_assignment_bool, parse_raw_assignment_int, s2b

__all__ = ['MinimizeGuardTask']

VARIABLES = 'nodetype terminal parent child_left child_right value child_value_left child_value_right'


class MinimizeGuardTask(Task):

    Reduction = namedtuple('Reduction', VARIABLES)
    Assignment = namedtuple('Assignment', VARIABLES + ' P')

    def __init__(self, scenario_tree, *, guard, solver_cmd=None, is_incremental=False, is_filesolver=False, outdir=''):
        self.scenario_tree = scenario_tree
        self.guard = guard

        self.outdir = outdir
        self.is_incremental = is_incremental
        self.is_filesolver = is_filesolver
        self.solver_config = dict(cmd=solver_cmd)

    def _new_solver(self):
        self._is_reduction_declared = False
        if self.is_incremental:
            self.solver = IncrementalSolver(**self.solver_config)
        elif self.is_filesolver:
            self.solver = FileSolver(**self.solver_config, filename_prefix=self.get_filename_prefix())
        else:
            self.solver = StreamSolver(**self.solver_config)

    def get_stem(self):
        return f'minimize_{self.scenario_tree.scenarios_stem}_P{self.P}'

    def get_filename_prefix(self):
        return os.path.join(self.outdir, self.get_stem())

    @property
    def number_of_variables(self):
        return self.solver.number_of_variables

    @property
    def number_of_clauses(self):
        return self.solver.number_of_clauses

    def run(self):
        # TODO: 'fast' argument
        log_debug(f'MinimizeGuardTask: running...')
        time_start_run = time.time()
        minimized_guard = None

        tree = self.scenario_tree
        X = tree.X
        U = tree.U
        unique_inputs = tree.unique_inputs
        guard = self.guard

        inputs = [None]  # [[bool]] of size (U',X), 1-based
        roots = [NotBool]  # [bool] of size U', 1-based
        for u in closed_range(1, U):
            q = guard.truth_table[u - 1]
            iv = unique_inputs[u - 1]
            if q == '0':
                inputs.append(s2b(iv))
                roots.append(False)
            elif q == '1':
                inputs.append(s2b(iv))
                roots.append(True)
            # else:
            #     log_debug(f'Don\'t care for input {iv}')
        U_ = len(inputs)  # may be less than the real tree.U
        log_debug(f'X={X}, U={U}, U_={U_}')

        self.inputs = inputs
        self.roots = roots

        for P in closed_range(1, 12):
            log_info(f'Trying P={P}...')

            self.P = P
            self._new_solver()
            self._declare_reduction()

            raw_assignment = self.solver.solve()
            self.finalize()
            assignment = self.parse_raw_assignment(raw_assignment)
            minimized_guard = self.build_guard(assignment)

            if minimized_guard:
                break
        else:
            log_br()
            log_error('Guard was not minimized')

        log_debug(f'MinimizeGuardTask: done in {time.time() - time_start_run:.2f} s')
        return minimized_guard

    def finalize(self):
        log_debug('MinimizeGuardTask: finalizing...')
        if self.is_incremental:
            self.solver.process.kill()

    def _declare_reduction(self):
        if self._is_reduction_declared:
            return
        self._is_reduction_declared = True

        P = self.P
        log_debug(f'Declaring reduction for P={P}...')
        time_start_reduction = time.time()

        # =-=-=-=-=-=
        #  CONSTANTS
        # =-=-=-=-=-=

        X = self.scenario_tree.X
        U = len(self.inputs) - 1  # Actually, this is U_, dont be confused with the real tree.U!

        comment = self.solver.comment
        add_clause = self.solver.add_clause
        declare_array = self.solver.declare_array
        ALO = self.solver.ALO
        AMO = self.solver.AMO
        imply = self.solver.imply
        iff = self.solver.iff

        # =-=-=-=-=-=
        #  VARIABLES
        # =-=-=-=-=-=

        # guards variables
        nodetype = declare_array(P, 3, with_zero=True)
        terminal = declare_array(P, X, with_zero=True)
        parent = declare_array(P, P, with_zero=True)
        child_left = declare_array(P, P, with_zero=True)
        child_right = declare_array(P, P, with_zero=True)
        value = declare_array(P, U)
        child_value_left = declare_array(P, U)
        child_value_right = declare_array(P, U)

        # =-=-=-=-=-=-=
        #  CONSTRAINTS
        # =-=-=-=-=-=-=

        so_far_state = [self.number_of_clauses]

        def so_far():
            now = self.number_of_clauses
            ans = now - so_far_state[0]
            so_far_state[0] = now
            return ans

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('2. ALO/AMO(nodetype)')
        for p in closed_range(1, P):
            ALO(nodetype[p])
            AMO(nodetype[p])

        log_debug(f'2. Clauses: {so_far()}', symbol='STAT')

        comment('3. Root value')
        for u in closed_range(1, U):
            if self.roots[u]:
                add_clause(value[1][u])
            else:
                add_clause(-value[1][u])

        log_debug(f'3. Clauses: {so_far()}', symbol='STAT')

        comment('7. Parent and children constraints')
        comment('7.0a. ALO/AMO(parent)')
        for p in closed_range(1, P):
            ALO(parent[p])
            AMO(parent[p])

        comment('7.0b. ALO/AMO(child_left)')
        for p in closed_range(1, P):
            ALO(child_left[p])
            AMO(child_left[p])

        comment('7.0c. ALO/AMO(child_right)')
        for p in closed_range(1, P):
            ALO(child_right[p])
            AMO(child_right[p])

        comment('7.1. Root has no parent')
        add_clause(parent[1][0])

        comment('7.2. All nodes (except root) have parent with lesser number')
        for p in closed_range(2, P):
            # OR_{par from 1 to p-1}( parent[p,par] )
            rhs = []
            for par in closed_range(1, p - 1):
                rhs.append(parent[p][par])
            add_clause(*rhs)

            # AND_{par=0, par from p+1 to P}( parent[p,par] )
            add_clause(-parent[p][0])
            for par in closed_range(p + 1, P):
                add_clause(-parent[p][par])

        comment('7.3. parent<->child relation')
        for p in closed_range(1, P - 1):
            for ch in closed_range(p + 1, P):
                # parent[ch,p] => (child_left[p,ch] | child_right[p,ch])
                add_clause(-parent[ch][p],
                           child_left[p][ch],
                           child_right[p][ch])

        log_debug(f'7. Clauses: {so_far()}', symbol='STAT')

        comment('9. Terminals constraints')
        comment('9.0. ALO/AMO(terminal)')
        for p in closed_range(1, P):
            ALO(terminal[p])
            AMO(terminal[p])

        comment('9.1. Only terminals have associated terminal variables')
        for p in closed_range(1, P):
            iff(nodetype[p][0], -terminal[p][0])

        comment('9.2. Terminals have no children')
        for p in closed_range(1, P):
            imply(nodetype[p][0], child_left[p][0])
            imply(nodetype[p][0], child_right[p][0])

        comment('9.3. Terminal: child_value_left and child_value_right are False')
        for p in closed_range(1, P):
            for u in closed_range(1, U):
                imply(nodetype[p][0], -child_value_left[p][u])
                imply(nodetype[p][0], -child_value_right[p][u])

        comment('9.4. Terminals have value from associated input variable')
        for p in closed_range(1, P):
            for x in closed_range(1, X):
                # terminal[p,x] -> AND_u( value[p,u] <-> inputs[u,x] )
                for u in closed_range(1, U):
                    if self.inputs[u][x]:
                        imply(terminal[p][x], value[p][u])
                    else:
                        imply(terminal[p][x], -value[p][u])

        log_debug(f'9. Clauses: {so_far()}', symbol='STAT')

        comment('10. AND/OR nodes constraints')
        comment('10.0. AND/OR nodes cannot have numbers P-1 or P')
        if P >= 1:
            add_clause(-nodetype[P][1])
            add_clause(-nodetype[P][2])
        if P >= 2:
            add_clause(-nodetype[P - 1][1])
            add_clause(-nodetype[P - 1][2])

        comment('10.1. AND/OR: left child has greater number')
        for p in closed_range(1, P - 2):
            # nodetype[p,1or2] => OR_{ch from p+1 to P-1}( child_left[p,ch] )
            rhs = []
            for ch in closed_range(p + 1, P - 1):
                rhs.append(child_left[p][ch])
            add_clause(-nodetype[p][1], *rhs)
            add_clause(-nodetype[p][2], *rhs)

            # nodetype[p,1or2] => AND_{ch from 0 to p; ch=P}( ~child_left[p,ch] )
            for ch in closed_range(0, p):
                imply(nodetype[p][1], -child_left[p][ch])
                imply(nodetype[p][2], -child_left[p][ch])
            imply(nodetype[p][1], -child_left[p][P])
            imply(nodetype[p][2], -child_left[p][P])

        comment('10.2. AND/OR: right child is adjacent (+1) to left')
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 1, P - 1):
                for nt in [1, 2]:
                    # (nodetype[p,1or2] & child_left[p,ch]) => child_right[p,ch+1]
                    add_clause(-nodetype[p][nt],
                               -child_left[p][ch],
                               child_right[p][ch + 1])

        comment('10.3. AND/OR: children`s parents')
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 1, P - 1):
                for nt in [1, 2]:
                    # (nodetype[p,1or2] & child_left[p,ch]) => (parent[ch,p] & parent[ch+1,p])
                    x1 = nodetype[p][nt]
                    x2 = child_left[p][ch]
                    x3 = parent[ch][p]
                    x4 = parent[ch + 1][p]
                    add_clause(-x1, -x2, x3)
                    add_clause(-x1, -x2, x4)

        comment('10.4a. AND/OR: child_value_left is a value of left child')
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 1, P - 1):
                for u in closed_range(1, U):
                    for nt in [1, 2]:
                        # (nodetype[p,1or2] & child_left[p,ch]) => (child_value_left[p,u] <=> value[ch,u])
                        x1 = nodetype[p][nt]
                        x2 = child_left[p][ch]
                        x3 = child_value_left[p][u]
                        x4 = value[ch][u]
                        add_clause(-x1, -x2, -x3, x4)
                        add_clause(-x1, -x2, x3, -x4)

        comment('10.4b. AND/OR: child_value_right is a value of right child')
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 2, P):
                for u in closed_range(1, U):
                    for nt in [1, 2]:
                        # (nodetype[p,1or2] & child_right[p,ch]) => (child_value_right[p,u] <=> value[ch,u])
                        x1 = nodetype[p][nt]
                        x2 = child_right[p][ch]
                        x3 = child_value_right[p][u]
                        x4 = value[ch][u]
                        add_clause(-x1, -x2, -x3, x4)
                        add_clause(-x1, -x2, x3, -x4)

        comment('10.5a. AND: value is calculated as a conjunction of children')
        for p in closed_range(1, P - 2):
            for u in closed_range(1, U):
                # nodetype[p,1] => (value[p,u] <=> cvl[p,u] & cvr[p,u])
                x1 = nodetype[p][1]
                x2 = value[p][u]
                x3 = child_value_left[p][u]
                x4 = child_value_right[p][u]
                add_clause(-x1, x2, -x3, -x4)
                add_clause(-x1, -x2, x3)
                add_clause(-x1, -x2, x4)

        comment('10.5b. OR: value is calculated as a disjunction of children')
        for p in closed_range(1, P - 2):
            for u in closed_range(1, U):
                # nodetype[p,2] => (value[p,u] <=> cvl[p,u] | cvr[p,u])
                x1 = nodetype[p][2]
                x2 = value[p][u]
                x3 = child_value_left[p][u]
                x4 = child_value_right[p][u]
                add_clause(-x1, -x2, x3, x4)
                add_clause(-x1, x2, -x3)
                add_clause(-x1, x2, -x4)

        log_debug(f'10. Clauses: {so_far()}', symbol='STAT')

        comment('11. NOT nodes constraints')
        comment('11.0. NOT nodes cannot have number P')
        add_clause(-nodetype[P][3])

        comment('11.1. NOT: left child has greater number')
        for p in closed_range(1, P - 1):
            # nodetype[p,3] => OR_{ch from p+1 to P}( child_left[p,ch] )
            rhs = []
            for ch in closed_range(p + 1, P):
                rhs.append(child_left[p][ch])
            add_clause(-nodetype[p][3], *rhs)

            # nodetype[p,3] => AND_{ch from 0 to p}( ~child_left[p,ch] )
            for ch in closed_range(0, p):
                imply(nodetype[p][3], -child_left[p][ch])

        comment('11.2. NOT: child`s parents')
        for p in closed_range(1, P - 1):
            for ch in closed_range(p + 1, P):
                # (nodetype[p]=3 & child_left[p]=ch) => parent[ch] = p
                add_clause(-nodetype[p][3],
                           -child_left[p][ch],
                           parent[ch][p])

        comment('11.3. NOT: no right child')
        for p in closed_range(1, P - 1):
            imply(nodetype[p][3], child_right[p][0])

        comment('11.4a. NOT: child_value_left is a value of left child')
        for p in closed_range(1, P - 1):
            for ch in closed_range(p + 1, P):
                for u in closed_range(1, U):
                    # (nodetype[p,3] & child_left[p,ch]) => (cvl[p,u] <=> value[ch,u])
                    x1 = nodetype[p][3]
                    x2 = child_left[p][ch]
                    x3 = value[ch][u]
                    x4 = child_value_left[p][u]
                    add_clause(-x1, -x2, -x3, x4)
                    add_clause(-x1, -x2, x3, -x4)

        comment('11.4b. NOT: child_value_right is False')
        for p in closed_range(1, P - 1):
            for u in closed_range(1, U):
                imply(nodetype[p][3], -child_value_right[p][u])

        comment('11.5. NOT: value is calculated as a negation of child')
        for p in closed_range(1, P - 1):
            for u in closed_range(1, U):
                # nodetype[p,3] => (value[p,u] <=> ~cvl[p,u])
                x1 = nodetype[p][3]
                x2 = value[p][u]
                x3 = -child_value_left[p][u]
                add_clause(-x1, -x2, x3)
                add_clause(-x1, x2, -x3)

        log_debug(f'11. Clauses: {so_far()}', symbol='STAT')

        # TODO: ?. Tree constraints
        #       ?.1. Edges
        #       constraint E = sum (p in 1..P) (bool2int(parent[p] != 0));
        #       ?.2. Vertices
        #       constraint V = P;
        #       ?.3. Tree equality
        #       constraint E = V - 1;

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.reduction = self.Reduction(
            nodetype=nodetype,
            terminal=terminal,
            parent=parent,
            child_left=child_left,
            child_right=child_right,
            value=value,
            child_value_left=child_value_left,
            child_value_right=child_value_right,
        )

        log_debug(f'Done declaring reduction ({self.number_of_variables} variables, {self.number_of_clauses} clauses) in {time.time() - time_start_reduction:.2f} s')

    def parse_raw_assignment(self, raw_assignment):
        if raw_assignment is None:
            return None

        log_debug('Building assignment...')
        time_start_assignment = time.time()

        wrapper_int = partial(parse_raw_assignment_int, raw_assignment)
        wrapper_bool = partial(parse_raw_assignment_bool, raw_assignment)

        assignment = self.Assignment(
            nodetype=wrapper_int(self.reduction.nodetype),
            terminal=wrapper_int(self.reduction.terminal),
            parent=wrapper_int(self.reduction.parent),
            child_left=wrapper_int(self.reduction.child_left),
            child_right=wrapper_int(self.reduction.child_right),
            value=wrapper_bool(self.reduction.value),
            child_value_left=wrapper_bool(self.reduction.child_value_left),
            child_value_right=wrapper_bool(self.reduction.child_value_right),
            P=self.P,
        )

        # log_debug(f'assignment = {assignment}')

        log_debug(f'Done building assignment in {time.time() - time_start_assignment:.2f} s')
        return assignment

    def build_guard(self, assignment):
        # TODO: 'dump' argument
        if assignment is None:
            return None

        log_info('MinimizeGuardTask: building guard...')
        guard = ParseTreeGuard(assignment.nodetype,
                               assignment.terminal,
                               assignment.parent,
                               assignment.child_left,
                               assignment.child_right,
                               names=self.scenario_tree.predicate_names)

        log_success(f'Minimized guard: {guard}')
        return guard
