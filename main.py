import os
import time
import regex
import shutil
import tempfile
import subprocess
from itertools import count, product
from collections import deque, namedtuple, OrderedDict

import click
import treelib  # pip
import numpy as np

GlobalState = {}

NAN = 2**31 - 1

Reduction = namedtuple('Reduction', 'C K P N number_of_base_variables number_of_base_clauses number_of_objective_variables number_of_objective_clauses number_of_cardinality_clauses color transition trans_event output_event algorithm_0 algorithm_1 nodetype terminal child_left child_right parent value child_value_left child_value_right fired_only not_fired objective')
Assignment = namedtuple('Assignment', 'color transition trans_event output_event algorithm_0 algorithm_1 nodetype terminal child_left child_right parent value child_value_left child_value_right fired_only not_fired')


def closed_range(start, stop=None, step=1):
    if stop is None:
        start += 1
    else:
        stop += 1
    return range(start, stop, step)


def log(text, symbol, *, fg=None, bg=None, bold=None, nl=True):
    if symbol is None:
        pre = ''
    else:
        pre = '[{: >1}] '.format(symbol)
    click.secho('{}{}'.format(pre, text), fg=fg, bg=bg, bold=bold, nl=nl)


def log_debug(text, symbol='.', *, fg='white', bg=None, bold=None, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_info(text, symbol='*', *, fg='blue', bg=None, bold=True, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_success(text, symbol='+', *, fg='green', bg=None, bold=True, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_warn(text, symbol='!', *, fg='magenta', bg=None, bold=True, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_error(text, symbol='!', *, fg='red', bg=None, bold=True, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_br(*, fg='white', bg=None, bold=False, nl=True):
    log(' '.join('=' * 30), symbol=None, fg=fg, bg=None, bold=bold, nl=nl)


def open_maybe_gzip(filename):
    if os.path.splitext(filename)[1] == '.gz':
        import gzip
        return gzip.open(filename, 'rt')
    else:
        return click.open_file(filename)


class OutputAction:
    def __init__(self, output_event, output_values):
        self.output_event = output_event  # Event :: str
        self.output_values = output_values  # Algorithm  :: str

    def __eq__(self, other):
        if isinstance(other, OutputAction):
            return self.output_event == other.output_event and \
                self.output_values == other.output_values
        return NotImplemented

    def __str__(self):
        return f'{self.output_event}({self.output_values})'

    def __repr__(self):
        return f'{self.__name__}(output_event={self.output_event}, output_values={self.output_values})'


class ScenarioElement:
    def __init__(self, input_event, input_values, output_actions):
        self.input_event = input_event  # Event :: str
        self.input_values = input_values  # Values :: str  # TODO: what about [bool] ?
        self.output_actions = output_actions  # [OutputAction]

    def __eq__(self, other):
        if isinstance(other, ScenarioElement):
            return self.input_event == other.input_event and \
                self.input_values == other.input_values and \
                self.output_actions == other.output_actions
        return NotImplemented

    @property
    def output_event(self):
        '''First output event'''
        return self.output_actions[0].output_event

    @property
    def output_values(self):
        '''First output values'''
        return self.output_actions[0].output_values

    @property
    def all(self):
        return str(self)

    def __str__(self):
        return f'{self.input_event}({self.input_values})->{self.output_actions[0]}'

    def __repr__(self):
        return f'{self.__name__}(input_event={self.input_event}, input_values={self.input_values}, output_actions={self.output_actions})'


class Scenario:
    def __init__(self):
        self.elements = []  # [ScenarioElement]

    def add_element(self, input_event, input_values, output_actions):
        self.elements.append(ScenarioElement(input_event, input_values, output_actions))

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        yield from self.elements

    def __repr__(self):
        return f'<Scenario {self.elements}>'


class ScenarioTree(treelib.Tree):

    def __init__(self, scenarios):
        super().__init__()

        clock = count(start=1)
        root = self.create_node(identifier=next(clock), data=ScenarioElement(None, None, [OutputAction('INITO', '0' * len(GlobalState['last_output_values']))]))
        for scenario in scenarios:
            current = root
            for element in scenario.elements:
                current = self.create_node(identifier=next(clock),
                                           parent=current,
                                           data=element)

    @property
    def predicate_names(self):
        if hasattr(self, '_predicate_names'):
            return self._predicate_names
        else:
            return list(map(lambda i: f'x{i+1}', range(len(next(iter(self.unique_inputs))))))

    @predicate_names.setter
    def predicate_names(self, value):
        self._predicate_names = value

    @property
    def output_variable_names(self):
        if hasattr(self, '_output_variable_names'):
            return self._output_variable_names
        else:
            return list(map(lambda i: f'z{i+1}', range(len(next(iter(self.unique_outputs))))))

    @output_variable_names.setter
    def output_variable_names(self, value):
        self._output_variable_names = value

    @property
    def V(self):
        return self.size()

    @property
    def E(self):
        return len(self.input_events)

    @property
    def O(self):
        return len(self.output_events)

    @property
    def X(self):
        return len(self.predicate_names)

    @property
    def Z(self):
        return len(self.output_variable_names)

    @property
    def U(self):
        return len(self.unique_inputs)

    @property
    def Y(self):
        return len(self.unique_outputs)

    @property
    def input_events(self):
        '''Unique input events'''
        return set(filter(None, (node.data.input_event for node in self.all_nodes_itr())))

    @property
    def output_events(self):
        '''Unique output events'''
        return set(filter(None, (node.data.output_event for node in self.all_nodes_itr())))

    @property
    def unique_inputs(self):
        '''Unique input values'''
        return set(filter(None, (node.data.input_values for node in self.all_nodes_itr())))

    @property
    def unique_outputs(self):
        '''Unique output values'''
        return set(filter(None, (node.data.output_values for node in self.all_nodes_itr())))


class Solution:

    def __init__(self, reduction, assignment):
        self.reduction = reduction
        self.assignment = assignment

    def from_raw_assignment(raw_assignment, reduction):
        log_info('Building assignment...')
        time_start_assignment = time.time()

        def wrapper_int(data):
            new_data = np.full(data.shape[:-1], NAN, dtype=int)
            for magic in product(*map(lambda n: range(1, n), data.shape[:-1])):
                for i, x in enumerate(data[magic]):
                    if x != NAN and raw_assignment[x] > 0:
                        new_data[magic] = i
                        break
                else:
                    log_warn(f'data[{magic}] is unknown')
            return new_data

        def wrapper_bool(data):
            # Assumption: lower bound is 1 on all levels
            new_data = np.full(data.shape, False, dtype=bool)
            for magic in product(*map(lambda n: range(1, n), data.shape)):
                new_data[magic] = raw_assignment[data[magic]] > 0
            return new_data

        def wrapper_algo(data):
            return [None] + [''.join('1' if raw_assignment[item] > 0 else '0'
                                     for item in subdata[1:])
                             for subdata in data[1:]]

        assignment = Assignment(
            color=wrapper_int(reduction.color),
            transition=wrapper_int(reduction.transition),
            trans_event=wrapper_int(reduction.trans_event),
            output_event=wrapper_int(reduction.output_event),
            algorithm_0=wrapper_algo(reduction.algorithm_0),
            algorithm_1=wrapper_algo(reduction.algorithm_1),
            nodetype=wrapper_int(reduction.nodetype),
            terminal=wrapper_int(reduction.terminal),
            child_left=wrapper_int(reduction.child_left),
            child_right=wrapper_int(reduction.child_right),
            parent=wrapper_int(reduction.parent),
            value=wrapper_bool(reduction.value),
            child_value_left=wrapper_bool(reduction.child_value_left),
            child_value_right=wrapper_bool(reduction.child_value_right),
            fired_only=wrapper_bool(reduction.fired_only),
            not_fired=wrapper_bool(reduction.not_fired),
        )

        solution = Solution(reduction, assignment)

        def traverse_generator(name):
            data = getattr(assignment, name)
            for magic in product(*map(lambda n: range(1, n), data.shape)):
                yield magic, data[magic]

        def show_int(name):
            t = [value for _, value in traverse_generator(name)]
            if len(t) > 50:
                log_debug(f'{name} = {t[:20]} ... and more')
            else:
                log_debug(f'{name} = {t}')

        show_int('color')
        show_int('transition')
        show_int('trans_event')
        show_int('output_event')
        show_int('nodetype')
        show_int('terminal')
        show_int('child_left')
        show_int('child_right')
        show_int('parent')
        # show_int('value')
        log_debug(f'Number of nodes: {solution.number_of_nodes}')

        log_success(f'Done building assignment in {time.time() - time_start_assignment:.2f} s')
        return solution

    @property
    def number_of_nodes(self):
        return sum(self.assignment.nodetype[c, k, p] != 4
                   for c in closed_range(1, self.reduction.C)
                   for k in closed_range(1, self.reduction.K)
                   for p in closed_range(1, self.reduction.P))


class Guard:

    class Node:

        def __init__(self, nodetype, terminal):
            assert 0 <= nodetype <= 4
            self.nodetype = nodetype
            self.terminal = terminal
            self.parent = self.child_left = self.child_right = None

        def eval(self, input_values):
            # input_values :: [bool] :: 1-based
            if self.nodetype == 0:  # Terminal
                assert self.terminal != 0
                return input_values[self.terminal]

            elif self.nodetype == 1:  # AND
                assert self.child_left is not None
                assert self.child_right is not None
                return self.child_left.eval(input_values) and self.child_right.eval(input_values)

            elif self.nodetype == 2:  # OR
                assert self.child_left is not None
                assert self.child_right is not None
                return self.child_left.eval(input_values) or self.child_right.eval(input_values)

            elif self.nodetype == 3:  # NOT
                assert self.child_left is not None
                return not self.child_left.eval(input_values)

            elif self.nodetype == 4:  # None
                # return False
                raise ValueError('Maybe think again?')

        def size(self):
            if self.nodetype == 0:
                return 1
            elif self.nodetype == 1 or self.nodetype == 2:
                return 1 + self.child_left.size() + self.child_right.size()
            elif self.nodetype == 3:
                return 1 + self.child_left.size()
            elif self.nodetype == 4:
                # return 0
                raise ValueError('Maybe think again?')

        def __str__(self):
            if self.nodetype == 0:  # Terminal
                return GlobalState['predicate_names'][self.terminal - 1]

            elif self.nodetype == 1:  # AND
                return f'({self.child_left} & {self.child_right})'

            elif self.nodetype == 2:  # OR
                return f'({self.child_left} | {self.child_right})'

            elif self.nodetype == 3:  # NOT
                if self.child_left.nodetype == 0:
                    return f'~{self.child_left}'
                else:
                    return f'~({self.child_left})'

            elif self.nodetype == 4:  # None
                raise ValueError(f'why are you trying to display None-typed node?')

    def __init__(self, nodetype, terminal, parent, child_left, child_right):
        assert len(nodetype) == len(terminal) == len(parent) == len(child_left) == len(child_right)
        P = len(nodetype) - 1

        nodes = [None] + [self.Node(nt, tn) for nt, tn in zip(nodetype[1:], terminal[1:])]  # 1-based

        for p in closed_range(1, P):
            nodes[p].parent = nodes[parent[p]]
            nodes[p].child_left = nodes[child_left[p]]
            nodes[p].child_right = nodes[child_right[p]]

        self.root = nodes[1]

    def eval(self, input_values):
        return self.root.eval(input_values)

    def size(self):
        return self.root.size()

    def __str__(self):
        return str(self.root)


class EFSM:

    class State:

        class Transition:

            def __init__(self, source, destination, input_event, guard):
                self.source = source  # State
                self.destination = destination  # State
                self.input_event = input_event  # Event
                self.guard = guard  # Guard

            def eval(self, input_values):
                return self.guard.eval([None] + [{'0': False, '1': True}[v]
                                                 for v in input_values])

            def __str__(self):
                # Example: 2->3 on REQ if (x1 & x2)
                return f'{self.source.id}→{self.destination.id} on {self.input_event} if {self.guard}'

        def __init__(self, id, output_event, algorithm_0, algorithm_1):
            self.id = id
            self.output_event = output_event
            self.algorithm_0 = algorithm_0
            self.algorithm_1 = algorithm_1
            self.transitions = []  # [Transition]

        def add_transition(self, destination, input_event, guard):
            self.transitions.append(self.Transition(self, destination, input_event, guard))

        def go(self, input_event, input_values, values):
            for transition in self.transitions:
                if transition.input_event == input_event and transition.eval(input_values):
                    destination = transition.destination
                    output_event = destination.output_event
                    new_values = destination.eval(values)
                    return destination, output_event, new_values
            return self, None, values

        def eval(self, values):
            return ''.join({'0': a0, '1': a1}[v]
                           for v, a0, a1 in zip(values, self.algorithm_0, self.algorithm_1))

        def __str__(self):
            # Example: 2/CNF(0:10101, 1:01101)
            return f'{self.id}/{self.output_event}(0:{self.algorithm_0}, 1:{self.algorithm_1})'

        def __repr__(self):
            return f'State(id={self.id}, output_event={self.output_event}, algorithm_0={self.algorithm_0}, algorithm_1={self.algorithm_1})'

    def __init__(self):
        self.states = OrderedDict()
        self.initial_state = None

    @property
    def number_of_states(self):
        return len(self.states)

    @property
    def number_of_transitions(self):
        return sum(len(state.transitions) for state in self.states.values())

    @property
    def number_of_nodes(self):
        return sum(sum(transition.guard.size()
                       for transition in state.transitions)
                   for state in self.states.values())

    def add_state(self, id, output_event, algorithm_0, algorithm_1):
        self.states[id] = self.State(id, output_event, algorithm_0, algorithm_1)

    def add_transition(self, src, dest, input_event, guard):
        self.states[src].add_transition(self.states[dest], input_event, guard)

    def go(self, src, input_event, input_values, values):
        source = self.states[src]
        destination, output_event, new_values = source.go(input_event, input_values, values)

        for dest, destination2 in self.states.items():
            if destination is destination2:
                return (dest, output_event, new_values)

    def pprint(self):
        for state in self.states.values():
            log_debug(state)
            for transition in state.transitions[:-1]:
                log_debug(f'├──{transition}')
            log_debug(f'└──{state.transitions[-1]}')


class Instance:

    def __init__(self, *, C, K, P, N, filename_traces, filename_predicate_names,
                 filename_output_variable_names, basename_formula, is_reuse=False, solver):
        # ======== This variables are for testing, before proper optimization is implemented
        self.C = C
        self.K = K
        self.P = P
        self.N = N
        # ========
        self.filename_traces = filename_traces
        self.basename_formula = basename_formula + '_' + os.path.splitext(os.path.basename(filename_traces))[0]
        self.is_reuse = is_reuse
        self.solver = solver

        self.scenarios_full = read_scenarios(self.filename_traces)
        self.scenarios = preprocess_scenarios(self.scenarios_full)

        log_info('Creating scenario tree...')
        self.tree = ScenarioTree(self.scenarios)
        for v in closed_range(1, min(30, self.tree.size())):
            log_debug(self.tree[v])
        self.tree.predicate_names = read_names(filename_predicate_names)
        self.tree.output_variable_names = read_names(filename_output_variable_names)
        log_success(f'Successfully built ScenarioTree of size {self.tree.size()}')
        log_br()

        GlobalState['predicate_names'] = self.tree.predicate_names
        GlobalState['output_variable_names'] = self.tree.output_variable_names

    def run_once(self):
        # C = self.C
        # K = self.K
        # P = self.P
        # N = self.N

        # self.prepare_base(C, K, P)
        # self.prepare_objective(C, K, P, N)
        # self.solve(C, K, P, N)
        # self.build_solution(C, K, P, N)
        # self.verify()
        # self.dump()
        raise NotImplementedError('please, rerun with --min')

    def run(self):
        C = self.C
        K = self.K
        P = self.P
        N = self.N

        reduction_base = self.generate_base_reduction(C, K, P)
        reduction_obj = self.generate_objective_function(reduction_base)

        last_solution = None
        N_try = N
        while True:
            log_info(f'Optimization: trying N = {N_try}...')
            log_br()
            reduction_card = self.generate_cardinality(N_try, reduction_obj)
            solution = self.solve(reduction_card)
            log_br()

            if solution is None:  # UNSAT
                break
            else:  # SAT
                last_solution = solution
                N_try = solution.number_of_nodes - 1

        if last_solution is None:  # completely UNSAT...
            log_error('COMPLETELY UNSAT')
        else:  # last_solution is the final answer
            log_success(f'Optimization: best solution with N = {last_solution.reduction.N}')
            self.build_efsm(last_solution)
            self.verify()
            self.dump()

    def generate_base_reduction(self, C, K, P):
        log_info(f'Generating base reduction for C={C}, K={K}, P={P}...')
        time_start_generate = time.time()

        def make_array(*dims):
            return np.full([d + 1 for d in dims], NAN, dtype=int)

        def make_array_bool(*dims):
            return np.full([d + 1 for d in dims], False, dtype=bool)

        bomba = count(1)

        def declare_array(*dims, with_zero):
            nonlocal bomba
            data = np.full([d + 1 for d in dims], NAN, dtype=int)
            for magic in product(*map(lambda n: closed_range(1, n), dims[:-1])):
                for i in closed_range(0 if with_zero else 1, dims[-1]):
                    data[magic][i] = next(bomba)
            return data

        clauses = []

        def clause(*vs):
            clauses.append(vs)

        def comment(s):
            clauses.append(s)

        def ALO(data):
            lower = 1 if data[0] == NAN else 0
            clause(*data[lower:])

        def AMO(data):
            lower = 1 if data[0] == NAN else 0
            upper = len(data) - 1
            for a in range(lower, upper):
                for b in closed_range(a + 1, upper):
                    clause(-data[a], -data[b])

        def so_far_generator():
            before = 0
            while True:
                now = len(clauses)
                yield now - before
                before = now

        so_far = so_far_generator()

        tree = self.tree

        unique_input_events = sorted(tree.input_events)
        unique_output_events = sorted(tree.output_events)
        unique_inputs = sorted(tree.unique_inputs)
        unique_outputs = sorted(tree.unique_outputs)

        log_debug(f'unique_input_events = {unique_input_events}')
        log_debug(f'unique_output_events = {unique_output_events}')
        # log_debug(f'unique_inputs = {unique_inputs}')
        # log_debug(f'unique_outputs = {unique_outputs}')

        V = tree.V
        E = tree.E
        O = tree.O
        X = tree.X
        Z = tree.Z
        U = tree.U
        Y = tree.Y

        log_debug(f'V = {V}')
        log_debug(f'E = {E}')
        log_debug(f'O = {O}')
        log_debug(f'X = {X}')
        log_debug(f'Z = {Z}')
        log_debug(f'U = {U}')
        log_debug(f'Y = {Y}')

        tree_parent = make_array(V)
        tree_previous_active = make_array(V)
        tree_input_event = make_array(V)
        tree_output_event = make_array(V)
        input_number = make_array(V)
        output_number = make_array(V)
        unique_input = make_array_bool(U, X)
        unique_output = make_array_bool(Y, Z)

        # tree_parent[1] = 0
        # tree_previous_active[1] = 0
        # tree_input_event[1] = 0
        tree_output_event[1] = unique_output_events.index(tree[1].data.output_event) + 1
        # input_number[1] = 0
        output_number[1] = unique_outputs.index(tree[1].data.output_values) + 1

        for v in closed_range(2, V):
            node = tree[v]
            tree_parent[v] = node.bpointer
            tree_input_event[v] = unique_input_events.index(node.data.input_event) + 1
            if node.data.output_event is not None:
                tree_output_event[v] = unique_output_events.index(node.data.output_event) + 1
            else:
                tree_output_event[v] = 0
            input_number[v] = unique_inputs.index(node.data.input_values) + 1
            output_number[v] = unique_outputs.index(node.data.output_values) + 1

        for v in closed_range(2, V):
            parent = tree_parent[v]
            while parent != 1 and tree_output_event[parent] == 0:
                parent = tree_parent[parent]
            tree_previous_active[v] = parent

        for u in closed_range(1, U):
            for x, c in enumerate(unique_inputs[u - 1], start=1):
                unique_input[u][x] = {'0': False, '1': True}[c]
        for y in closed_range(1, Y):
            for z, c in enumerate(unique_outputs[y - 1], start=1):
                unique_output[y][z] = {'0': False, '1': True}[c]

        # log_debug(f'tree_parent:\n{tree_parent}')
        # log_debug(f'tree_previous_active:\n{tree_previous_active}')
        # log_debug(f'tree_input_event:\n{tree_input_event}')
        # log_debug(f'tree_output_event:\n{tree_output_event}')
        # log_debug(f'input_number:\n{input_number}')
        # log_debug(f'output_number:\n{output_number}')
        # log_debug(f'unique_input:\n{unique_input}')
        # log_debug(f'unique_output:\n{unique_output}')

        # Declare variables

        color = declare_array(V, C, with_zero=True)
        transition = declare_array(C, K, C, with_zero=True)
        trans_event = declare_array(C, K, E, with_zero=False)
        output_event = declare_array(C, O, with_zero=False)
        algorithm_0 = declare_array(C, Z, with_zero=False)
        algorithm_1 = declare_array(C, Z, with_zero=False)
        nodetype = declare_array(C, K, P, 4, with_zero=True)
        terminal = declare_array(C, K, P, X, with_zero=True)
        child_left = declare_array(C, K, P, P, with_zero=True)
        child_right = declare_array(C, K, P, P, with_zero=True)
        parent = declare_array(C, K, P, P, with_zero=True)
        value = declare_array(C, K, P, U, with_zero=False)
        child_value_left = declare_array(C, K, P, U, with_zero=False)
        child_value_right = declare_array(C, K, P, U, with_zero=False)
        fired_only = declare_array(C, K, U, with_zero=False)
        not_fired = declare_array(C, K, U, with_zero=False)

        # log_debug(f'color:\n{color}')
        # log_debug(f'transition:\n{transition}')
        # log_debug(f'trans_event:\n{trans_event}')
        # log_debug(f'output_event:\n{output_event}')
        # log_debug(f'algorithm_0:\n{algorithm_0}')
        # log_debug(f'algorithm_1:\n{algorithm_1}')
        # log_debug(f'nodetype:\n{nodetype}')
        # log_debug(f'terminal:\n{terminal}')
        # log_debug(f'child_left:\n{child_left}')
        # log_debug(f'child_right:\n{child_right}')
        # log_debug(f'parent:\n{parent}')
        # log_debug(f'value:\n{value}')
        # log_debug(f'child_value_left:\n{child_value_left}')
        # log_debug(f'child_value_right:\n{child_value_right}')
        # log_debug(f'fired_only:\n{fired_only}')
        # log_debug(f'not_fired:\n{not_fired}')

        # Declare constraints

        comment('1. Color constraints')
        comment('1.0a ALO(color)')
        for v in closed_range(1, V):
            ALO(color[v])
        comment('1.0b AMO(color)')
        for v in closed_range(1, V):
            AMO(color[v])

        comment('1.1. Start vertex corresponds to start state')
        #   constraint color[1] = 1;
        clause(color[1, 1])

        comment('1.2. Only passive vertices (except root) have aux color)')
        #   constraint forall (v in 2..V where tree_output_event[v] == O+1) (
        #       color[v] = C+1
        #   );
        for v in closed_range(2, V):
            if tree_output_event[v] == 0:
                clause(color[v, 0])
            else:
                clause(-color[v, 0])

        log_debug(f'1. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('2. Transition constraints')
        comment('2.0a ALO(transition)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                ALO(transition[c, k])
        comment('2.0a AMO(transition)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                AMO(transition[c, k])

        comment('2.0b ALO(trans_event)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                ALO(trans_event[c, k])
        comment('2.0b AMO(trans_event)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                AMO(trans_event[c, k])

        comment('2.1. (transition + trans_event + fired_only definitions)')
        #   constraint forall (v in 2..V where tree_output_event[v] != O+1) (
        #       exists (k in 1..K) (
        #           y[color[tree_previous_active[v]], k] = color[v]
        #           /\ w[color[tree_previous_active[v]], k] = tree_input_event[v]
        #           /\ fired_only[color[tree_previous_active[v]], k, input_nums[v]]
        #       )
        #   );
        for v in closed_range(2, V):
            if tree_output_event[v] != 0:
                for cv in closed_range(0, C):
                    for ctpa in closed_range(1, C):
                        constraint = [-color[v, cv], -color[tree_previous_active[v], ctpa]]
                        for k in closed_range(1, K):
                            # aux <-> y[ctpa,k,cv] /\ w[ctpa,k,tie[v]] /\ fired_only[ctpa,k,input_number[v]]
                            # x <-> a /\ b /\ c
                            # CNF: (~a ~b ~c x) & (a ~x) & (b ~x) & (c ~x)
                            x1 = transition[ctpa, k, cv]
                            x2 = trans_event[ctpa, k, tree_input_event[v]]
                            x3 = fired_only[ctpa, k, input_number[v]]
                            aux = next(bomba)
                            clause(-x1, -x2, -x3, aux)
                            clause(-aux, x1)
                            clause(-aux, x2)
                            clause(-aux, x3)
                            constraint.append(aux)
                        clause(*constraint)

        comment('2.2. Forbid transition self-loops')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                clause(-transition[c, k, c])

        log_debug(f'2. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('3. Output event constraints')
        comment('3.0a ALO(output_event)')
        for c in closed_range(1, C):
            ALO(output_event[c])
        comment('3.0b AMO(output_event)')
        for c in closed_range(1, C):
            AMO(output_event[c])

        comment('3.1. Start state does INITO')
        # clause(output_event[1, unique_output_events.index('INITO') + 1])
        clause(output_event[1, tree_output_event[1]])

        comment('3.2. Output event is the same as in the tree')
        for v in closed_range(2, V):
            if tree_output_event[v] != 0:
                for cv in closed_range(1, C):
                    clause(-color[v, cv], output_event[cv, tree_output_event[v]])

        log_debug(f'3. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('4. Algorithm constraints')
        comment('4.1. Start state does nothing')
        #   constraint forall (z in 1..Z) (
        #       d_0[1, z] = 0 /\
        #       d_1[1, z] = 1
        #   );
        for z in closed_range(1, Z):
            clause(-algorithm_0[1, z])
            clause(algorithm_1[1, z])

        comment('4.2. What to do with zero')
        #   constraint forall (v in 2..V, z in 1..Z where tree_output_event[v] != O+1) (
        #       not tree_z[output_nums[tree_previous_active[v]], z] ->
        #           d_0[color[v], z] = tree_z[output_nums[v], z]
        #   );
        for v in closed_range(2, V):
            if tree_output_event[v] != 0:
                for z in closed_range(1, Z):
                    if not unique_output[output_number[tree_previous_active[v]]][z]:
                        for cv in closed_range(1, C):
                            if unique_output[output_number[v]][z]:
                                clause(-color[v, cv], algorithm_0[cv, z])
                            else:
                                clause(-color[v, cv], -algorithm_0[cv, z])

        comment('4.3. What to do with one')
        #   constraint forall (v in 2..V, z in 1..Z where tree_output_event[v] != O+1) (
        #       tree_z[output_nums[tree_previous_active[v]], z] ->
        #           d_1[color[v], z] = tree_z[output_nums[v], z]
        #   );
        for v in closed_range(2, V):
            if tree_output_event[v] != 0:
                for z in closed_range(1, Z):
                    if unique_output[output_number[tree_previous_active[v]]][z]:
                        for cv in closed_range(1, C):
                            if unique_output[output_number[v]][z]:
                                clause(-color[v, cv], algorithm_1[cv, z])
                            else:
                                clause(-color[v, cv], -algorithm_1[cv, z])

        log_debug(f'4. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('5. Firing constraints')
        comment('5.1. (not_fired definition)')
        #   constraint forall (v in 2..V, ctpa in 1..C where tree_output_event[v] == O+1) (
        #       ctpa = color[tree_previous_active[v]] ->
        #           not_fired[ctpa, K, input_nums[v]]
        #   );
        for v in closed_range(2, V):
            if tree_output_event[v] == 0:
                for ctpa in closed_range(1, C):
                    clause(-color[tree_previous_active[v], ctpa],
                           not_fired[ctpa, K, input_number[v]])

        comment('5.2. not fired')
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
                clause(-not_fired[c, 1, u], -value[c, 1, 1, u])
                clause(not_fired[c, 1, u], value[c, 1, 1, u])
                for k in closed_range(2, K):
                    x1 = not_fired[c, k, u]
                    x2 = -value[c, k, 1, u]
                    x3 = not_fired[c, k - 1, u]
                    clause(-x1, x2)
                    clause(-x1, x3)
                    clause(x1, -x2, -x3)

        comment('5.3. fired_only')
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
                clause(-fired_only[c, 1, u], value[c, 1, 1, u])
                clause(fired_only[c, 1, u], -value[c, 1, 1, u])
                for k in closed_range(2, K):
                    x1 = fired_only[c, k, u]
                    x2 = value[c, k, 1, u]
                    x3 = not_fired[c, k - 1, u]
                    clause(-x1, x2)
                    clause(-x1, x3)
                    clause(x1, -x2, -x3)

        log_debug(f'5. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('6. Guard conditions constraints')
        comment('6.1a ALO(nodetype)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(nodetype[c, k, p])
        comment('6.1b AMO(nodetype)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(nodetype[c, k, p])

        comment('6.2a ALO(terminal)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(terminal[c, k, p])
        comment('6.2b AMO(terminal)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(terminal[c, k, p])

        comment('6.3a ALO(child_left)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(child_left[c, k, p])
        comment('6.3b AMO(child_left)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(child_left[c, k, p])

        comment('6.4a ALO(child_right)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(child_right[c, k, p])
        comment('6.4b AMO(child_right)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(child_right[c, k, p])

        comment('6.5a ALO(parent)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    ALO(parent[c, k, p])
        comment('6.5b AMO(parent)')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    AMO(parent[c, k, p])

        log_debug(f'6. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('7. Extra guard conditions constraints')
        comment('7.1. Root has no parent')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                clause(parent[c, k, 1, 0])

        comment('7.2. BFS: typed nodes (except root) have parent with lesser number')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(2, P):
                    clause(nodetype[c, k, p, 4],
                           *[parent[c, k, p, par] for par in range(1, p)])

        comment('7.3. parent of v`s child is *v*')
        comment('=== dropped ===')
        # for c in closed_range(1, C):
        #     for k in closed_range(1, K):
        #         for p in range(1, P):
        #             for ch in closed_range(p + 1, P):
        #                 clause(-child_left[c, k, p, ch], parent[c, k, ch, p])
        #             for ch in closed_range(p + 2, P):
        #                 clause(-child_right[c, k, p, ch], parent[c, k, ch, p])

        log_debug(f'7. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('8. None-type nodes constraints')
        comment('8.1. None-type nodes have largest numbers')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    clause(-nodetype[c, k, p, 4], nodetype[c, k, p + 1, 4])

        comment('8.2. None-type nodes have no parent')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    clause(-nodetype[c, k, p, 4], parent[c, k, p, 0])

        comment('8.3. None-type nodes have no children')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    clause(-nodetype[c, k, p, 4], child_left[c, k, p, 0])
                    clause(-nodetype[c, k, p, 4], child_right[c, k, p, 0])

        comment('8.4. None-type nodes have False value and child_values')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    for u in closed_range(1, U):
                        clause(-nodetype[c, k, p, 4], -value[c, k, p, u])
                        clause(-nodetype[c, k, p, 4], -child_value_left[c, k, p, u])
                        clause(-nodetype[c, k, p, 4], -child_value_right[c, k, p, u])

        log_debug(f'8. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('9. Terminals constraints')
        comment('9.1. Only terminals have associated terminal variables')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    clause(-nodetype[c, k, p, 0], -terminal[c, k, p, 0])
                    clause(nodetype[c, k, p, 0], terminal[c, k, p, 0])

        comment('9.2. Terminals have no children')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    clause(-nodetype[c, k, p, 0], child_left[c, k, p, 0])
                    clause(-nodetype[c, k, p, 0], child_right[c, k, p, 0])

        comment('9.3. Terminals have value from associated input variable')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P):
                    for x in closed_range(1, X):
                        for u in closed_range(1, U):
                            x1 = terminal[c, k, p, x]
                            x2 = value[c, k, p, u]
                            if unique_input[u][x]:
                                clause(-x1, x2)
                            else:
                                clause(-x1, -x2)

        log_debug(f'9. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('10. AND/OR nodes constraints')
        comment('10.0. AND/OR nodes cannot have numbers P-1 or P')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in [P - 1, P]:
                    clause(-nodetype[c, k, p, 1])
                    clause(-nodetype[c, k, p, 2])

        comment('10.1. AND/OR: left child has greater number')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    _cons = [child_left[c, k, p, ch] for ch in closed_range(p + 1, P - 1)]
                    clause(-nodetype[c, k, p, 1], *_cons)
                    clause(-nodetype[c, k, p, 2], *_cons)

        comment('10.2. AND/OR: right child is adjacent (+1) to left')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 1, P - 1):
                        for nt in [1, 2]:
                            clause(-nodetype[c, k, p, nt],
                                   -child_left[c, k, p, ch],
                                   child_right[c, k, p, ch + 1])

        comment('10.3. AND/OR: children`s parents')
        # comment('10.3. === see Extra (7.3) ===')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 1, P - 1):
                        for nt in [1, 2]:
                            _cons = [-nodetype[c, k, p, nt], -child_left[c, k, p, ch]]
                            clause(*_cons, parent[c, k, ch, p])
                            clause(*_cons, parent[c, k, ch + 1, p])

        comment('10.4a AND/OR: child_value_left is a value of left child')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 1, P - 1):
                        for u in closed_range(1, U):
                            for nt in [1, 2]:
                                x1 = nodetype[c, k, p, nt]
                                x2 = child_left[c, k, p, ch]
                                x3 = child_value_left[c, k, p, u]
                                x4 = value[c, k, ch, u]
                                clause(-x1, -x2, -x3, x4)
                                clause(-x1, -x2, x3, -x4)

        comment('10.4b AND/OR: child_value_right is a value of right child')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for ch in closed_range(p + 2, P):
                        for u in closed_range(1, U):
                            for nt in [1, 2]:
                                x1 = nodetype[c, k, p, nt]
                                x2 = child_right[c, k, p, ch]
                                x3 = child_value_right[c, k, p, u]
                                x4 = value[c, k, ch, u]
                                clause(-x1, -x2, -x3, x4)
                                clause(-x1, -x2, x3, -x4)

        comment('10.5a AND: value is calculated as a conjunction of children')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for u in closed_range(1, U):
                        x1 = nodetype[c, k, p, 1]
                        x2 = value[c, k, p, u]
                        x3 = child_value_left[c, k, p, u]
                        x4 = child_value_right[c, k, p, u]
                        clause(-x1, x2, -x3, -x4)
                        clause(-x1, -x2, x3)
                        clause(-x1, -x2, x4)

        comment('10.5b OR: value is calculated as a disjunction of children')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 2):
                    for u in closed_range(1, U):
                        x1 = nodetype[c, k, p, 2]
                        x2 = value[c, k, p, u]
                        x3 = child_value_left[c, k, p, u]
                        x4 = child_value_right[c, k, p, u]
                        clause(-x1, -x2, x3, x4)
                        clause(-x1, x2, -x3)
                        clause(-x1, x2, -x4)

        log_debug(f'10. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        comment('11. NOT nodes constraints')
        comment('11.0. NOT nodes cannot have number P')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                clause(-nodetype[c, k, P, 3])

        comment('11.1. NOT: left child has greater number')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    _cons = [child_left[c, k, p, ch] for ch in closed_range(p + 1, P)]
                    clause(-nodetype[c, k, p, 3], *_cons)

        comment('11.2. NOT: no right child')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    clause(-nodetype[c, k, p, 3], child_right[c, k, p, 0])

        comment('11.3. NOT: child`s parents')
        # comment('11.3. === see Extra (7.3) ===')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for ch in closed_range(p + 1, P):
                        clause(-nodetype[c, k, p, 3],
                               -child_left[c, k, p, ch],
                               parent[c, k, ch, p])

        comment('11.4a NOT: child_value_left is a value of left child')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for ch in closed_range(p + 1, P):
                        for u in closed_range(1, U):
                            x1 = nodetype[c, k, p, 3]
                            x2 = child_left[c, k, p, ch]
                            x3 = child_value_left[c, k, p, u]
                            x4 = value[c, k, ch, u]
                            clause(-x1, -x2, -x3, x4)
                            clause(-x1, -x2, x3, -x4)

        comment('11.4b NOT: child_value_right is False')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for u in closed_range(1, U):
                        clause(-nodetype[c, k, p, 3], -child_value_right[c, k, p, u])

        comment('11.5. NOT: value is calculated as a negation of child')
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for p in closed_range(1, P - 1):
                    for u in closed_range(1, U):
                        x1 = nodetype[c, k, p, 3]
                        x2 = value[c, k, p, u]
                        x3 = child_value_left[c, k, p, u]
                        clause(-x1, -x2, -x3)
                        clause(-x1, x2, x3)

        log_debug(f'11. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # Declare any ad-hoc you like
        comment('AD-HOCs')

        comment('adhoc-1')
        #   constraint forall (c in 1..C, k in 1..K) (
        #       y[c, k] = C+1 <->
        #           nodetype[c, k, 1] = 4
        #   );
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                clause(-transition[c, k, 0], nodetype[c, k, 1, 4])
                clause(transition[c, k, 0], -nodetype[c, k, 1, 4])

        comment('adhoc-2')
        #   constraint forall (c in 1..C, k in 1..K-1) (
        #       y[c, k] = C+1 ->
        #           y[c, k+1] = C+1
        #   );
        for c in closed_range(1, C):
            for k in closed_range(1, K - 1):
                clause(-transition[c, k, 0], transition[c, k + 1, 0])

        comment('adhoc-4')
        #   constraint forall (c in 1..C, k in 1..K, g in 1..U) (
        #       fired_only[c, k, g] ->
        #           nodetype[c, k, 1] != 4
        #   );
        for c in closed_range(1, C):
            for k in closed_range(1, K):
                for u in closed_range(1, U):
                    clause(-fired_only[c, k, u], -nodetype[c, k, 1, 4])

        log_debug(f'A. Clauses: {next(so_far)}', symbol='DEBUG')
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        only_clauses = [clause for clause in clauses if not isinstance(clause, str)]

        number_of_variables = next(bomba) - 1
        number_of_clauses = len(only_clauses)
        # TODO: count unique clauses
        del bomba  # invalidated

        log_debug(f'Base variables: {number_of_variables}')
        log_debug(f'Base clauses: {number_of_clauses}')

        filename_dimacs_base = self.get_filename_dimacs_base(C, K, P)
        if not self.is_reuse or not os.path.exists(filename_dimacs_base):
            self.write_dimacs(clauses, filename_dimacs_base)

        log_success(f'Done generating base reduction in {time.time() - time_start_generate:.2f} s')
        log_br()

        return Reduction(C=C,
                         K=K,
                         P=P,
                         N=None,
                         number_of_base_variables=number_of_variables,
                         number_of_base_clauses=number_of_clauses,
                         number_of_objective_variables=0,
                         number_of_objective_clauses=0,
                         number_of_cardinality_clauses=0,
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
                         objective=None)

    def generate_objective_function(self, reduction):
        log_info(f'Generating objective function...')
        time_start_objective = time.time()

        clauses = []

        def clause(*vs):
            clauses.append(vs)

        def comment(s):
            clauses.append(s)

        C = reduction.C
        K = reduction.K
        P = reduction.P
        bomba = count(reduction.number_of_base_variables + 1)

        comment('Preparing objective function')
        _E = [-reduction.nodetype[c, k, p, 4]
              for c in closed_range(1, C)
              for k in closed_range(1, K)
              for p in closed_range(1, P)]  # set of input variables
        _L = []  # set of linking variables

        q = deque([e] for e in _E)
        while len(q) != 1:
            a = q.popleft()  # 0-based
            b = q.popleft()  # 0-based

            m1 = len(a)
            m2 = len(b)
            m = m1 + m2

            r = [next(bomba) for _ in range(m)]  # 0-based

            if len(q) != 0:
                _L.extend(r)

            for alpha in range(m1 + 1):
                for beta in range(m2 + 1):
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
                        clause(*C1)
                    if C2 is not None:
                        clause(*C2)

            q.append(r)

        _S = q.pop()  # set of output variables
        assert len(_E) == len(_S)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        only_clauses = [clause for clause in clauses if not isinstance(clause, str)]

        number_of_variables = len(_S) + len(_L)
        number_of_clauses = len(only_clauses)
        # TODO: count unique clauses
        # number_of_unique_constraints = len(set(only_constraints))

        # ============
        _base = reduction.number_of_base_variables
        _obj = number_of_variables
        _total = next(bomba) - 1
        del bomba  # invalidated
        if _base + _obj != _total:
            log_error(f'base+obj != total :: {_base}+{_obj} != {_total}')
        # else:
        #     log_success(f'base+obj == total :: {_base}+{_obj} == {_total}', bold=False)
        assert _base + _obj == _total
        # ============

        log_debug(f'Objective variables: {number_of_variables}')
        log_debug(f'Objective clauses: {number_of_clauses}')
        # log_debug(f'Objective unique constraints: {number_of_unique_constraints}')
        # if number_of_constraints != number_of_unique_constraints:
        #     log_warn('Some constraints are duplicated')

        filename_dimacs_objective = self.get_filename_dimacs_objective(C, K, P)
        if not self.is_reuse or not os.path.exists(filename_dimacs_objective):
            self.write_dimacs(clauses, filename_dimacs_objective)

        log_success(f'Done generating objective function in {time.time() - time_start_objective:.2f} s')
        log_br()

        return reduction._replace(
            number_of_objective_variables=number_of_variables,
            number_of_objective_clauses=number_of_clauses,
            objective=_S
        )

    def generate_cardinality(self, N, reduction):
        log_info(f'Generating cardinality function...')
        time_start_cardinality = time.time()

        clauses = []

        def clause(*vs):
            clauses.append(vs)

        def comment(s):
            clauses.append(s)

        C = reduction.C
        K = reduction.K
        P = reduction.P

        comment('Objective function: sum(E) <= N')
        for i in closed_range(N + 1, C * K * P):
            clause(-reduction.objective[i - 1])

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        only_clauses = [clause for clause in clauses if not isinstance(clause, str)]

        number_of_clauses = len(only_clauses)
        # TODO: count unique clauses
        # number_of_unique_clauses = len(set(only_clauses))

        log_debug(f'Cardinality clauses: {number_of_clauses}')
        # log_debug(f'Cardinality unique clauses: {number_of_unique_clauses}')
        # if number_of_clauses != number_of_unique_clauses:
        #     log_warn('Some clauses are duplicated')

        filename_dimacs = self.get_filename_dimacs_cardinality(C, K, P, N)
        self.write_dimacs(clauses, filename_dimacs)

        log_success(f'Done generating cardinality in {time.time() - time_start_cardinality:.2f} s')

        return reduction._replace(
            N=N,
            number_of_cardinality_clauses=number_of_clauses
        )

    def solve(self, reduction):
        log_info(f'Solving...')
        time_start_solve = time.time()

        C = reduction.C
        K = reduction.K
        P = reduction.P
        N = reduction.N

        self.write_header(reduction)

        cmd = f'cat {self.get_filenames(C,K,P,N)} | {self.solver}'
        log_debug(cmd)
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)

        if p.returncode == 10:
            log_success(f'SAT in {time.time() - time_start_solve:.2f} s')

            raw_assignment = [None]  # now 1-based
            for line in p.stdout.split('\n'):
                if line.startswith('v'):
                    for value in map(int, regex.findall(r'-?\d+', line)):
                        if value == 0:
                            break
                        assert abs(value) == len(raw_assignment)
                        raw_assignment.append(value)

            return Solution.from_raw_assignment(raw_assignment, reduction)

        elif p.returncode == 20:
            log_error(f'UNSAT in {time.time() - time_start_solve:.2f} s')
        else:
            log_error(f'returncode {p.returncode} in {time.time() - time_start_solve:.2f} s')

    def get_filename_dimacs_base(self, C, K, P):
        return f'{self.basename_formula}_C{C}_K{K}_P{P}_base.dimacs'

    def get_filename_dimacs_objective(self, C, K, P):
        return f'{self.basename_formula}_C{C}_K{K}_P{P}_objective.dimacs'

    def get_filename_dimacs_cardinality(self, C, K, P, N):
        return f'{self.basename_formula}_C{C}_K{K}_P{P}_N{N}_cardinality.dimacs'

    def get_filename_header(self, C, K, P, N):
        return f'{self.basename_formula}_C{C}_K{K}_P{P}_N{N}_header.dimacs'

    def get_filenames(self, C, K, P, N):
        return f'{self.basename_formula}_C{C}_K{K}_P{P}_{{N{N}_header,base,objective,N{N}_cardinality}}.dimacs'

    def write_dimacs(self, clauses, filename):
        # log_info(f'Writing clauses to <{filename}>...')
        # time_start_write = time.time()
        log_debug(f'Writing clauses to <{filename}>...')

        with tempfile.NamedTemporaryFile('w', delete=False) as f:
            # log_debug(f'Temporarily writing to <{f.name}>...')
            for clause in clauses:
                if isinstance(clause, str):
                    f.write('c ' + clause + '\n')
                else:
                    # assert all(clause)  # no zeros
                    f.write(' '.join(map(str, clause)) + ' 0\n')
        # log_debug(f'Moving temporary file to target')
        shutil.move(f.name, filename)

        # log_success(f'Done writing DIMACS in {time.time() - time_start_write:.2f} s')

    def write_header(self, reduction):
        filename = self.get_filename_header(reduction.C, reduction.K, reduction.P, reduction.N)

        # log_info(f'Writing header to <{filename}>...')
        # time_start_write = time.time()
        log_debug(f'Writing header to <{filename}>...')

        number_of_variables = reduction.number_of_base_variables + reduction.number_of_objective_variables
        number_of_clauses = reduction.number_of_base_clauses + reduction.number_of_objective_clauses + reduction.number_of_cardinality_clauses

        with click.open_file(filename, 'w') as f:
            f.write(f'p cnf {number_of_variables} {number_of_clauses}\n')

        # log_success(f'Done writing header in {time.time() - time_start_write:.2f} s')

    def build_efsm(self, solution):
        log_info('Building EFSM...')
        time_start_build = time.time()

        C = solution.reduction.C
        K = solution.reduction.K
        # P = solution.reduction.P
        # N = solution.reduction.N
        assignment = solution.assignment

        unique_input_events = sorted(self.tree.input_events)
        unique_output_events = sorted(self.tree.output_events)

        self.efsm = EFSM()

        for c in closed_range(1, C):
            self.efsm.add_state(c,
                                unique_output_events[assignment.output_event[c] - 1],
                                assignment.algorithm_0[c],
                                assignment.algorithm_1[c])
        self.efsm.initial_state = 1

        for c in closed_range(1, C):
            for k in closed_range(1, K):
                dest = assignment.transition[c, k]
                if dest != 0:
                    input_event = unique_input_events[assignment.trans_event[c, k] - 1]
                    guard = Guard(assignment.nodetype[c, k],
                                  assignment.terminal[c, k],
                                  assignment.parent[c, k],
                                  assignment.child_left[c, k],
                                  assignment.child_right[c, k])
                    self.efsm.add_transition(c, dest, input_event, guard)

        # =======================
        self.efsm.pprint()
        # =======================

        log_success(f'Done building EFSM with {self.efsm.number_of_states} states, {self.efsm.number_of_transitions} transitions and {self.efsm.number_of_nodes} nodes in {time.time() - time_start_build:.2f} s')
        log_br()

    def verify(self):
        log_info('Verifying...')
        time_start_verify = time.time()

        for i, scenario in enumerate(self.scenarios):
            log_info(f'Processing scenario {i+1}/{len(self.scenarios)}...')

            current_state = self.efsm.initial_state
            current_values = self.tree[1].data.output_values

            for j, element in enumerate(scenario):
                # log_debug(f'element: {element}', symbol=f'{j+1}/{len(scenario)}')
                input_event = element.input_event
                input_values = element.input_values
                new_state, output_event, new_values = self.efsm.go(current_state, input_event, input_values, current_values)

                if output_event != element.output_event and new_values != element.output_values:
                    log_error('incorrect output_event and output_values', symbol=f'{j+1}/{len(scenario)}')
                elif output_event != element.output_event:
                    log_error('incorrect output_event', symbol=f'{j+1}/{len(scenario)}')
                elif new_values != element.output_values:
                    log_error('incorrect output_values', symbol=f'{j+1}/{len(scenario)}')
                # assert output_event == element.output_event
                # assert new_values == element.output_values

                current_state = new_state
                current_values = new_values

        log_success(f'Done verifying in {time.time() - time_start_verify:.2f} s')
        log_br()

    def dump(self):
        log_br()


def read_scenarios(filename):
    log_info(f'Reading scenarios from <{click.format_filename(filename)}>...')
    time_start_reading = time.time()

    scenarios = []  # [Scenario]

    with open_maybe_gzip(filename) as f:
        # number_of_scenarios = int(f.readline())
        f.readline()

        for line in map(str.strip, f):
            scenario = Scenario()

            GlobalState['last_output_values'] = '0' * len(regex.search(r'out=.*?\[([10]*)\]', line).group(1))

            for m in regex.finditer(r'(?:in=(?P<input_event>.*?)\[(?P<input_values>[10]*)\];\s*)+(?:out=(?P<output_event>.*?)\[(?P<output_values>[10]*)\];\s*)*', line):
                t = list(zip(*m.captures('input_event', 'input_values')))

                for input_event, input_values in t[:-1]:
                    scenario.add_element(input_event, input_values, [OutputAction(None, GlobalState['last_output_values'])])

                if len(m.captures('output_event')) == 0:
                    continue

                input_event, input_values = t[-1]
                output_actions = []
                for output_event, output_values in zip(*m.captures('output_event', 'output_values')):
                    output_actions.append(OutputAction(output_event, output_values))
                GlobalState['last_output_values'] = output_values
                scenario.add_element(input_event, input_values, output_actions)

            scenarios.append(scenario)

    log_success(f'Done reading {len(scenarios)} scenarios with total {sum(map(len, scenarios))} elements in {time.time() - time_start_reading:.2f} s')
    return scenarios


def preprocess_scenarios(scenarios_full):
    log_info(f'Preprocessing scenarios...')
    time_start_preprocessing = time.time()

    scenarios = []

    for scenario in scenarios_full:
        processed = Scenario()  # new Scenario
        last = scenario.elements[0]
        processed.elements.append(last)

        for element in scenario.elements[1:]:
            if last is None or element != last:
                processed.elements.append(element)
            # else:
            #     print(f'skipping because elem = {element}, last = {last}')
            last = element

        scenarios.append(processed)

    log_success(f'Done preprocessing {len(scenarios)} scenarios from {sum(map(len, scenarios_full))} down to {sum(map(len, scenarios))} elements in {time.time() - time_start_preprocessing:.2} s')
    return scenarios


def read_names(filename):
    log_info(f'Reading names from <{click.format_filename(filename)}>...')
    with open_maybe_gzip(filename) as f:
        names = f.read().strip().split('\n')
    log_success(f'Done reading names: {", ".join(names)}')
    return names


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help'],
))
@click.option('-t', '--traces', 'filename_traces', metavar='<path/->',
              type=click.Path(exists=True, allow_dash=True),
              help='File with traces')
@click.option('--predicate-names', 'filename_predicate_names', metavar='<path>',
              type=click.Path(exists=True),
              default='predicate-names', show_default=True,
              help='File with precidate names')
@click.option('--output-variable-names', 'filename_output_variable_names', metavar='<path>',
              type=click.Path(exists=True),
              default='output-variables', show_default=True,
              help='File with output variables names')
@click.option('-C', 'C', type=int, metavar='<int>', required=True,
              help='Number of colors (automata states)')
@click.option('-K', 'K', type=int, metavar='<int>', required=True,
              help='Maximum number of transitions from each state')
@click.option('-P', 'P', type=int, metavar='<int>', required=True,
              help='Maximum number of nodes in guard\'s boolean formula\'s parse tree')
@click.option('-N', 'N', type=int, metavar='<int>', default=0,
              help='Initial upper bound on total number of nodes in all guard-trees')
@click.option('--basename', metavar='<basename>',
              type=click.Path(writable=True),
              default='cnf', show_default=True,
              help='Basename of file with generated formula')
@click.option('--solver', metavar='<sat-solver-cmd>',
              default='glucose -model -verb=0', show_default=True,
              # default='cryptominisat5 --verb=0', show_default=True,
              # default='cadical -q', show_default=True,
              help='SAT-solver')
@click.option('--min', 'is_minimize', is_flag=True,
              help='Do minimize')
@click.option('--reuse', 'is_reuse', is_flag=True,
              help='Reuse generated base reduction and objective function')
def cli(filename_traces, filename_predicate_names, filename_output_variable_names, C, K, P, N, basename, solver, is_minimize, is_reuse):
    log_info('Welcome!')
    log_br()
    time_start = time.time()

    if N == 0:
        N = C * K * P
        log_warn(f'Using maximum N=C*K*P = {N}')

    instance = Instance(C=C, K=K, P=P, N=N,
                        filename_traces=filename_traces,
                        filename_predicate_names=filename_predicate_names,
                        filename_output_variable_names=filename_output_variable_names,
                        basename_formula=basename,
                        is_reuse=is_reuse,
                        solver=solver)
    if is_minimize:
        instance.run()
    else:
        instance.run_once()

    log_success(f'All done in {time.time() - time_start:.2f} s')


if __name__ == '__main__':
    cli()
