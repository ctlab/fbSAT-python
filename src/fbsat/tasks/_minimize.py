import os
import time
from collections import namedtuple
from functools import partial

from ..efsm import ParseTreeGuard
from ..printers import log_br, log_debug, log_error, log_info, log_success, log_warn
from ..solver import StreamSolver
from ..utils import closed_range, s2b, parse_raw_assignment_int, parse_raw_assignment_bool, NotBool

__all__ = ['MinimizeGuardTask']

VARIABLES = 'nodetype terminal child_left child_right parent value child_value_left child_value_right'


class MinimizeGuardTask:

    Reduction = namedtuple('Reduction', VARIABLES)
    Assignment = namedtuple('Assignment', VARIABLES + ' P')

    def __init__(self, scenario_tree, *, guard, solver_cmd=None, outdir=''):
        self.scenario_tree = scenario_tree
        self.guard = guard

        self.outdir = outdir
        self.solver_config = dict(cmd=solver_cmd)

    def _new_solver(self):
        self._is_reduction_declared = False
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
        U_ = len(inputs)  # maybe less than the real tree.U
        log_debug(f'X={X}, U={U}, U_={U_}')

        self.inputs = inputs
        self.roots = roots

        for P in closed_range(1, 15):
            log_info(f'Trying P={P}...')

            self.P = P
            self._new_solver()
            self._declare_reduction()

            raw_assignment = self.solver.solve()
            assignment = self.parse_raw_assignment(raw_assignment)
            minimized_guard = self.build_guard(assignment)

            if minimized_guard:
                break
        else:
            log_br()
            log_error('Guard was not minimized')

        log_debug(f'MinimizeGuardTask: done in {time.time() - time_start_run:.2f} s')
        return minimized_guard

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

        new_variable = self.solver.new_variable
        add_clause = self.solver.add_clause
        declare_array = self.solver.declare_array
        ALO = self.solver.ALO
        AMO = self.solver.AMO
        imply = self.solver.imply
        iff = self.solver.iff
        iff_and = self.solver.iff_and
        iff_or = self.solver.iff_or

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

        # constraints

        # 1. Nodetype constraints
        # 1.0. ALO/AMO(nodetype)
        for p in closed_range(1, P):
            ALO(nodetype[p])
            AMO(nodetype[p])

        # 1.1. AND/OR nodes cannot have numbers P-1 or P
        if P >= 1:
            add_clause(-nodetype[P][1])
            add_clause(-nodetype[P][2])
        if P >= 2:
            add_clause(-nodetype[P - 1][1])
            add_clause(-nodetype[P - 1][2])

        # 1.2. NOT nodes cannot have number P
        add_clause(-nodetype[P][3])

        log_debug(f'1. Clauses: {so_far()}', symbol='STAT')

        # 2. Terminals constraints
        # 2.0. ALO/AMO(terminal)
        for p in closed_range(1, P):
            ALO(terminal[p])
            AMO(terminal[p])

        # 2.1. Only terminals have associated terminal variables
        for p in closed_range(1, P):
            iff(nodetype[p][0], -terminal[p][0])

        # 2.2. Terminals have no children
        for p in closed_range(1, P):
            imply(nodetype[p][0], child_left[p][0])
            imply(nodetype[p][0], child_right[p][0])

        # 2.3. Terminals have value from associated input variable
        for p in closed_range(1, P):
            for x in closed_range(1, X):
                # terminal[p,x] -> AND_u( value[p,u] <-> inputs[u,x] )
                for u in closed_range(1, U):
                    try:
                        if self.inputs[u][x]:
                            imply(terminal[p][x], value[p][u])
                        else:
                            imply(terminal[p][x], -value[p][u])
                    except:
                        log_error(f'Out of range for u={u}, x={x}')
                        log_debug(f'inputs[u={u}] = {self.inputs[u]}')
                        raise

        log_debug(f'2. Clauses: {so_far()}', symbol='STAT')

        # 3. Parent and children constraints
        # 3.0. ALO/AMO(parent,child_left,child_right)
        for p in closed_range(1, P):
            ALO(parent[p])
            AMO(parent[p])
        for p in closed_range(1, P):
            ALO(child_left[p])
            AMO(child_left[p])
        for p in closed_range(1, P):
            ALO(child_right[p])
            AMO(child_right[p])

        # 3.1. Root has no parent
        add_clause(parent[1][0])

        # 3.2. BFS: typed nodes (except root) have parent with lesser number
        for p in closed_range(2, P):
            add_clause(-parent[p][0])
            for p_ in closed_range(p + 1, P):
                add_clause(-parent[p][p_])

        # 3.3. parent<->child relation
        for p in closed_range(1, P - 1):
            for ch in closed_range(p + 1, P):
                # parent[ch,p] -> child_left[p,ch] | child_right[p,ch]
                add_clause(-parent[ch][p], child_left[p][ch], child_right[p][ch])

        # 3.4. Node with number P have no children; P-1 -- no right child
        add_clause(child_left[P][0])
        add_clause(child_right[P][0])
        for u in closed_range(1, U):
            add_clause(-child_value_left[P][u])
            add_clause(-child_value_right[P][u])
        if P > 1:
            add_clause(child_right[P - 1][0])
            for u in closed_range(1, U):
                add_clause(-child_value_right[P - 1][u])

        log_debug(f'3. Clauses: {so_far()}', symbol='STAT')

        # 4. AND/OR nodes constraints
        # 4.1. AND/OR: left child has greater number
        for p in closed_range(1, P - 2):
            for p_ in closed_range(0, p):
                imply(nodetype[p][1], -child_left[p][p_])
                imply(nodetype[p][2], -child_left[p][p_])
            imply(nodetype[p][1], -child_left[p][P])
            imply(nodetype[p][2], -child_left[p][P])

        # 4.2. AND/OR: right child is adjacent (+1) to left
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 1, P - 1):
                # (nodetype[p,1or2] & child_left[p][ch]) -> child_right[p][ch+1]
                for nt in [1, 2]:
                    add_clause(-nodetype[p][nt], -child_left[p][ch], child_right[p][ch + 1])

        # 4.3. AND/OR: children's parents
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 1, P - 1):
                # (nodetype[p,1or2] & child_left[p,ch]) -> (parent[ch,p] & parent[ch+1,p])
                for nt in [1, 2]:
                    add_clause(-nodetype[p][nt], -child_left[p][ch], parent[ch][p])
                    add_clause(-nodetype[p][nt], -child_left[p][ch], parent[ch + 1][p])

        # 4.4a AND/OR: child_value_left is a value of left child
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 1, P - 1):
                for u in closed_range(1, U):
                    # (nodetype[p,1or2] & child_left[p,ch]) -> (child_value_left[p,u] <-> value[ch,u])
                    for nt in [1, 2]:
                        add_clause(-nodetype[p][nt], -child_left[p][ch], -child_value_left[p][u], value[ch][u])
                        add_clause(-nodetype[p][nt], -child_left[p][ch], child_value_left[p][u], -value[ch][u])

        # 4.4b AND/OR: child_value_right is a value of right child
        for p in closed_range(1, P - 2):
            for ch in closed_range(p + 2, P):
                for u in closed_range(1, U):
                    # (nodetype[p,1or2] & child_left[p,ch]) -> (child_value_left[p,u] <-> value[ch,u])
                    for nt in [1, 2]:
                        add_clause(-nodetype[p][nt], -child_right[p][ch], -child_value_right[p][u], value[ch][u])
                        add_clause(-nodetype[p][nt], -child_right[p][ch], child_value_right[p][u], -value[ch][u])

        # 4.5a AND: value is calculated as a conjunction of children
        for p in closed_range(1, P - 2):
            for u in closed_range(1, U):
                # nodetype[p,1] -> (value[p,u] <-> child_value_left[p,u] & child_value_right[p,u])
                add_clause(-nodetype[p][1], value[p][u], -child_value_left[p][u], -child_value_right[p][u])
                add_clause(-nodetype[p][1], -value[p][u], child_value_left[p][u])
                add_clause(-nodetype[p][1], -value[p][u], child_value_right[p][u])

        # 4.5b OR: value is calculated as a disjunction of children
        for p in closed_range(1, P - 2):
            for u in closed_range(1, U):
                # nodetype[p,2] -> (value[p,u] <-> child_value_left[p,u] & child_value_right[p,u])
                add_clause(-nodetype[p][2], -value[p][u], child_value_left[p][u], child_value_right[p][u])
                add_clause(-nodetype[p][2], value[p][u], -child_value_left[p][u])
                add_clause(-nodetype[p][2], value[p][u], -child_value_right[p][u])

        log_debug(f'4. Clauses: {so_far()}', symbol='STAT')

        # 5. NOT nodes constraints
        # 5.1. NOT: left child has greater number
        for p in closed_range(1, P - 1):
            for p_ in closed_range(0, p):
                imply(nodetype[p][3], -child_left[p][p_])

        # 5.2. NOT: no right child
        for p in closed_range(1, P - 1):
            imply(nodetype[p][3], child_right[p][0])

        # 5.3. NOT: child's parents
        for p in closed_range(1, P - 1):
            for ch in closed_range(p + 1, P):
                add_clause(-nodetype[p][3], -child_left[p][ch], parent[ch][p])

        # 5.4a NOT: child_value_left is a value of left child
        for p in closed_range(1, P - 1):
            for ch in closed_range(p + 1, P):
                for u in closed_range(1, U):
                    # (nodetype[p,3] & child_left[p,ch]) -> (child_value_left[p,u] <-> value[ch,u])
                    add_clause(-nodetype[p][3], -child_left[p][ch], -child_value_left[p][u], value[ch][u])
                    add_clause(-nodetype[p][3], -child_left[p][ch], child_value_left[p][u], -value[ch][u])

        # 5.4b NOT: child_value_right is False
        for p in closed_range(1, P - 1):
            for u in closed_range(1, U):
                # nodetype[p,3] -> ~child_value_right[p,u]
                imply(nodetype[p][3], -child_value_right[p][u])

        # 5.5. NOT: value is calculated as a negation of child
        for p in closed_range(1, P - 1):
            for u in closed_range(1, U):
                # nodetype[p,3] -> (value[p,u] <-> ~child_value_left[p,u])
                add_clause(-nodetype[p][3], -value[p][u], -child_value_left[p][u])
                add_clause(-nodetype[p][3], value[p][u], child_value_left[p][u])

        log_debug(f'5. Clauses: {so_far()}', symbol='STAT')

        # 6. Root value
        for u in closed_range(1, U):
            if self.roots[u]:
                add_clause(value[1][u])
            else:
                add_clause(-value[1][u])

        log_debug(f'6. Clauses: {so_far()}', symbol='STAT')

        # TODO: 7. Tree constraints
        #       7.1. Edges
        #       constraint E = sum (p in 1..P) (bool2int(parent[p] != 0));
        #       7.2. Vertices
        #       constraint V = P;
        #       7.3. Tree equality
        #       constraint E = V - 1;

        # =-=-=-=-=
        #   FINISH
        # =-=-=-=-=

        self.reduction = self.Reduction(
            nodetype=nodetype,
            terminal=terminal,
            child_left=child_left,
            child_right=child_right,
            parent=parent,
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
            child_left=wrapper_int(self.reduction.child_left),
            child_right=wrapper_int(self.reduction.child_right),
            parent=wrapper_int(self.reduction.parent),
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
