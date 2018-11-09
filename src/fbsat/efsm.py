import os
import pickle
import random
import string
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from io import StringIO

from lxml import etree

from .printers import *
from .scenario import OutputAction, Scenario
from .utils import *

__all__ = ['FullGuard', 'ParseTreeGuard', 'TruthTableGuard', 'EFSM']


class Guard(ABC):
    @abstractmethod
    def eval(self, input_values):
        # input_values :: [1..X]:Bool
        pass


class FullGuard(Guard):

    def __init__(self, input_values, *, names=None):
        self.input_values = input_values  # [1..X]:Bool
        X = len(input_values) - 1
        if names is not None:
            if len(names) != len(input_values) - 1:
                names = [f'x{x}' for x in closed_range(1, X)]
                log_warn('names are fixed due to mismatch: ' + '.'.join(names))
        else:
            names = [f'x{x}' for x in closed_range(1, X)]
            log_warn('using default names: ' + '.'.join(names))
        self.names = names

    def eval(self, input_values):
        # input_values :: [1..X]:Bool
        return self.input_values[1:] == input_values[1:]

    def __str__(self):
        return b2s(self.input_values[1:])

    def __str_gv__(self):
        # return '&'.join({True: '', False: '~'}[value] + name
        #                 for name, value in zip(self.names, self.input_values[1:]))
        return str(self)

    def __str_fbt__(self):
        # FIXME: maybe use 'NOT' for boolean negation?
        return '&'.join({True: '', False: '~'}[value] + name
                        for name, value in zip(self.names, self.input_values[1:]))

    def __str_smv__(self):
        return ' & '.join({True: '', False: '!'}[value] + name
                          for name, value in zip(self.names, self.input_values[1:]))


class TruthTableGuard(Guard):
    def __init__(self, truth_table, unique_inputs):
        self.truth_table = truth_table  # str from {'01x'} of length U
        self.unique_inputs = unique_inputs  # [input_values::str] of length U

    def eval(self, input_values):
        # input_values :: [1..X]:Bool
        s = b2s(input_values[1:])
        return self.truth_table[self.unique_inputs.index(s)] in '1x'

    def __str__(self):
        return f'[{self.truth_table}]'

    def __str_gv__(self):
        return '[...]'

    def __str_fbt__(self):
        return f'TruthTable({self.truth_table})'

    def __str_smv__(self):
        return self.__str_fbt__()


class ParseTreeGuard(Guard):

    class Node:

        names = None

        def __init__(self, nodetype, terminal):
            assert 0 <= nodetype <= 4
            self.nodetype = nodetype
            self.terminal = terminal
            self.parent = self.child_left = self.child_right = None

        def eval(self, input_values):
            # input_values :: [1..X]:Bool
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
                if self.names is not None:
                    return self.names[self.terminal - 1]
                else:
                    return f'x{self.terminal}'
            elif self.nodetype == 1:  # AND
                left = str(self.child_left)
                right = str(self.child_right)
                if self.child_left.nodetype == 2:  # Left child is OR
                    left = f'({left})'
                if self.child_right.nodetype == 2:  # Right child is OR
                    right = f'({right})'
                return f'{left} & {right}'
            elif self.nodetype == 2:  # OR
                left = str(self.child_left)
                right = str(self.child_right)
                if self.child_left.nodetype == 1:  # Left child is AND
                    left = f'({left})'
                if self.child_right.nodetype == 1:  # Right child is AND
                    right = f'({right})'
                return f'{left} | {right}'
            elif self.nodetype == 3:  # NOT
                if self.child_left.nodetype in [0, 3]:
                    return f'~{self.child_left}'
                else:
                    return f'~({self.child_left})'
            elif self.nodetype == 4:  # None
                raise ValueError(f'why are you trying to display None-typed node?')

        def __str_gv__(self):
            return str(self)

        def __str_fbt__(self):
            if self.nodetype == 0:  # Terminal
                return str(self)
            elif self.nodetype == 1:  # AND
                left = self.child_left.__str_fbt__()
                right = self.child_right.__str_fbt__()
                if self.child_left.nodetype == 2:  # Left child is OR
                    left = f'({left})'
                if self.child_right.nodetype == 2:  # Right child is OR
                    right = f'({right})'
                return f'{left} AND {right}'
            elif self.nodetype == 2:  # OR
                left = self.child_left.__str_fbt__()
                right = self.child_right.__str_fbt__()
                # if self.child_left.nodetype == 1:  # Left child is AND
                #     left = f'({left})'
                # if self.child_right.nodetype == 1:  # Right child is AND
                #     right = f'({right})'
                return f'{left} OR {right}'
            elif self.nodetype == 3:  # NOT
                # FIXME: maybe use '~' for boolean negation?
                if self.child_left.nodetype in [0, 3]:
                    return f'NOT {self.child_left.__str_fbt__()}'
                else:
                    return f'NOT({self.child_left.__str_fbt__()})'
            elif self.nodetype == 4:  # None
                raise ValueError(f'why are you trying to display None-typed node?')

        def __str_smv__(self):
            if self.nodetype == 0:  # Terminal
                return str(self)
            elif self.nodetype == 1:  # AND
                left = self.child_left.__str_smv__()
                right = self.child_right.__str_smv__()
                if self.child_left.nodetype == 2:  # Left child is OR
                    left = f'({left})'
                if self.child_right.nodetype == 2:  # Right child is OR
                    right = f'({right})'
                return f'{left} & {right}'
            elif self.nodetype == 2:  # OR
                left = self.child_left.__str_smv__()
                right = self.child_right.__str_smv__()
                if self.child_left.nodetype == 1:  # Left child is AND
                    left = f'({left})'
                if self.child_right.nodetype == 1:  # Right child is AND
                    right = f'({right})'
                return f'{left} | {right}'
            elif self.nodetype == 3:  # NOT
                if self.child_left.nodetype in [0, 3]:
                    return f'!{self.child_left.__str_smv__()}'
                else:
                    return f'!({self.child_left.__str_smv__()})'
            elif self.nodetype == 4:  # None
                raise ValueError(f'why are you trying to display None-typed node?')

    def __init__(self, nodetype, terminal, parent, child_left, child_right, *, names=None):
        # Note: all arguments are 1-based
        assert len(nodetype) == len(terminal) == len(parent) == len(child_left) == len(child_right)
        P = len(nodetype) - 1
        self.Node.names = names

        nodes = [None] + [self.Node(nt, tn) for nt, tn in zip(nodetype[1:], terminal[1:])]  # 1-based

        for p in closed_range(1, P):
            nodes[p].parent = nodes[parent[p]]
            nodes[p].child_left = nodes[child_left[p]]
            nodes[p].child_right = nodes[child_right[p]]

        self.root = nodes[1]

    @classmethod
    def from_input(cls, input_values, *, names=None):
        # input_values :: [1..X]:Bool
        self = cls.__new__(cls)
        self.Node.names = names
        X = len(input_values) - 1

        left = self.Node(0, 1)
        if not input_values[1]:
            negation = self.Node(3, 0)
            negation.child_left = left
            left.parent = negation
            left = negation

        for x in closed_range(2, X):
            right = self.Node(0, x)
            if not input_values[x]:
                negation = self.Node(3, 0)
                negation.child_left = right
                right.parent = negation
                right = negation
            new_root = self.Node(1, 0)
            new_root.child_left = left
            new_root.child_right = right
            left.parent = new_root
            right.parent = new_root
            left = new_root

        self.root = left
        return self

    @classmethod
    def from_formula(cls, formula, *, names):
        self = cls.__new__(cls)
        self.Node.names = names

        def process(e):
            if hasattr(e, 'operator'):
                if e.operator == '&':
                    root = self.Node(1, 0)
                    left = process(e.args[0])
                    right = process(e.args[1])
                elif e.operator == '|':
                    root = self.Node(2, 0)
                    left = process(e.args[0])
                    right = process(e.args[1])
                elif e.operator == '~':
                    root = self.Node(3, 0)
                    left = process(e.args[0])
                    right = None
                else:
                    log_warn(f'Unknown operator: {e.operator}')

                if left:
                    left.parent = root
                    root.child_left = left
                if right:
                    right.parent = root
                    root.child_right = right

                return root
            elif hasattr(e, 'obj'):
                return self.Node(0, names.index(e.obj) + 1)
            else:
                raise ValueError('TRUE/FALSE formula')

        # pip install boolean.py
        from boolean import BooleanAlgebra
        self.root = process(BooleanAlgebra().parse(formula))
        return self

    @classmethod
    def new_random(cls, P, *, names):
        import random
        self = cls.__new__(cls)
        self.Node.names = names
        X = len(names)

        def rnd_node(P):
            if P == 1:
                terminal = random.randint(1, X)
                return self.Node(0, terminal)
            elif P == 2:
                nodetype = 3
            else:
                nodetype = random.randint(1, 3)
            root = self.Node(nodetype, 0)
            if nodetype == 3:
                child = rnd_node(P - 1)
                child.parent = root
                root.child_left = child
            else:
                P_ = random.randint(1, P - 2)
                left = rnd_node(P_)
                right = rnd_node(P - P_ - 1)
                left.parent = root
                right.parent = root
                root.child_left = left
                root.child_right = right
            return root

        self.root = rnd_node(P)
        from boolean import BooleanAlgebra
        try:
            return cls.from_formula(str(BooleanAlgebra().parse(str(self)).simplify()), names=names)
        except:
            return cls.new_random(P, names=names)

    def eval(self, input_values):
        # input_values :: [1..X]:Bool
        return self.root.eval(input_values)

    def size(self):
        return self.root.size()

    def __str__(self):
        return str(self.root)

    def __str_gv__(self):
        return self.root.__str_gv__()

    def __str_fbt__(self):
        return self.root.__str_fbt__()

    def __str_smv__(self):
        return self.root.__str_smv__()


class EFSM:

    class State:

        class Transition:

            def __init__(self, source, destination, input_event, guard):
                self.source = source  # State
                self.destination = destination  # State
                self.input_event = input_event  # Event
                self.guard = guard  # Guard

            def eval(self, input_values):
                # input_values :: str of length X
                return self.guard.eval(s2b(input_values))

            def __str__(self):
                # Example: 2->3 on REQ if (x1 & x2)
                return f'{self.source.id} to {self.destination.id} on {self.input_event} if {self.guard}'

            def __str_gv__(self, k):
                return f'{self.source.id} -> {self.destination.id} [label="{k}:{self.input_event}/{self.guard.__str_gv__()}"]'

            def __str_fbt__(self):
                return f'{self.input_event}&{self.guard.__str_fbt__()}'

            def __str_smv_destination__(self):
                return(f'_state={self.source.__str_smv__()}'
                       f' & {self.input_event}'
                       f' & ({self.guard.__str_smv__()})'
                       f' : {self.destination.__str_smv__()};')

            def __str_smv_output_event__(self):
                return (f'_state={self.source.__str_smv__()}'
                        f' & next(_state)={self.destination.__str_smv__()}'
                        # f' & {self.input_event}'
                        # f' & ({self.guard.__str_smv__()})'
                        f' : TRUE;')

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

        def get_suitable_transition_and_index(self, input_event, input_values):
            for k, transition in enumerate(self.transitions, start=1):  # Note: k is 1-based
                if transition.input_event == input_event and transition.eval(input_values):
                    return transition, k
            return None, 0

        def eval(self, values):
            return ''.join({'0': a0, '1': a1}[v]
                           for v, a0, a1 in zip(values, self.algorithm_0, self.algorithm_1))

        def __str__(self):
            # Example: 2/CNF(0:10101, 1:01101)
            return f'{self.id}/{self.output_event}(0:{self.algorithm_0}, 1:{self.algorithm_1})'

        def __str_gv__(self):
            return f'{self.id} [label="<p>{self.id}|{self.output_event}|{{0:{self.algorithm_0}|1:{self.algorithm_1}}}" shape=Mrecord]'
            # return f'{self.id} [label=<\n<TABLE CELLBORDER="1" CELLSPACING="0" STYLE="ROUNDED">\n<TR>\n    <TD WIDTH="30" PORT="p">{self.id}</TD>\n    <TD>{self.output_event}</TD>\n<TD BALIGN="LEFT">0:{self.algorithm_0}<BR/>1:{self.algorithm_1}</TD>\n</TR>\n</TR>\n</TABLE>> shape=plaintext]'
            # return f'{self.id} [label="{self.id}:{self.output_event}({self.algorithm_0}_{self.algorithm_1})"]'

        def __str_fbt__(self):
            return f's{self.id}'

        def __str_smv__(self):
            return f's{self.id}_{self.algorithm_0}_{self.algorithm_1}'

        def __repr__(self):
            return f'State(id={self.id!r}, output_event={self.output_event!r}, algorithm_0={self.algorithm_0!r}, algorithm_1={self.algorithm_1!r})'

    def __init__(self, input_events, output_events, input_names, output_names):
        self.states = OrderedDict()
        self.input_events = input_events
        self.output_events = output_events
        self.input_names = input_names
        self.output_names = output_names

    @classmethod
    def new_from_scenario_tree(cls, scenario_tree):
        return cls(scenario_tree.input_events,
                   scenario_tree.output_events,
                   scenario_tree.input_names,
                   scenario_tree.output_names)

    @classmethod
    def new_with_full_guards(cls, scenario_tree, assignment):
        log_debug('Building EFSM with full guards...')
        time_start_build = time.time()

        tree = scenario_tree
        C = assignment.C
        E = tree.E
        U = tree.U
        input_events = tree.input_events    # [str]^E  0-based
        output_events = tree.output_events  # [str]^O  0-based
        unique_input = tree.unique_input  # [1..U, 1..X]:bool

        efsm = cls.new_from_scenario_tree(scenario_tree)
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
                        guard = FullGuard(unique_input[u], names=tree.input_names)
                        # guard = ParseTreeGuard.from_input(unique_input[u], names=tree.input_names)
                        efsm.add_transition(c, dest, input_events[e - 1], guard)

        if efsm.number_of_states != assignment.C:
            log_error(f'Inequal number of states: efsm has {efsm.number_of_states}, assignment has {assignment.C}')
        if efsm.number_of_transitions != assignment.T:
            log_error(f'Inequal number of transitions: efsm has {efsm.number_of_transitions}, assignment has {assignment.T}')

        log_debug(f'Done building EFSM with {efsm.number_of_states} states and {efsm.number_of_transitions} transitions in {time.time() - time_start_build:.2f} s')
        return efsm

    @classmethod
    def new_with_truth_tables(cls, scenario_tree, assignment):
        log_debug('Building EFSM with truth tables...')
        time_start_build = time.time()

        tree = scenario_tree
        C = assignment.C
        K = assignment.K
        E = tree.E
        U = tree.U
        input_events = tree.input_events    # [str]^E  0-based
        output_events = tree.output_events  # [str]^O  0-based
        unique_inputs = tree.unique_inputs  # [str]^U  0-based

        efsm = cls.new_from_scenario_tree(scenario_tree)
        for c in closed_range(1, C):
            efsm.add_state(c,
                           output_events[assignment.output_event[c] - 1],
                           assignment.algorithm_0[c],
                           assignment.algorithm_1[c])
        efsm.initial_state = 1

        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    dest = assignment.transition[c][e][k]
                    if dest != 0:
                        input_event = input_events[e - 1]
                        truth_table = ''
                        for u in closed_range(1, U):
                            if assignment.not_fired[c][e][u][k]:
                                truth_table += '0'
                            elif assignment.first_fired[c][e][u][k]:
                                truth_table += '1'
                            else:
                                truth_table += 'x'
                        guard = TruthTableGuard(truth_table, unique_inputs)
                        efsm.add_transition(c, dest, input_event, guard)

        if efsm.number_of_states != assignment.C:
            log_error(f'Inequal number of states: efsm has {efsm.number_of_states}, assignment has {assignment.C}')
        if efsm.number_of_transitions != assignment.T:
            log_error(f'Inequal number of transitions: efsm has {efsm.number_of_transitions}, assignment has {assignment.T}')

        log_debug(f'Done building EFSM with {efsm.number_of_states} states and {efsm.number_of_transitions} transitions in {time.time() - time_start_build:.2f} s')
        return efsm

    @classmethod
    def new_with_parse_trees(cls, scenario_tree, assignment):
        log_debug('Building EFSM with parse trees...')
        time_start_build = time.time()

        tree = scenario_tree
        C = assignment.C
        K = assignment.K
        E = tree.E
        input_events = tree.input_events    # [str]^E  0-based
        output_events = tree.output_events  # [str]^O  0-based

        efsm = cls.new_from_scenario_tree(scenario_tree)
        for c in closed_range(1, C):
            efsm.add_state(c,
                           output_events[assignment.output_event[c] - 1],
                           assignment.algorithm_0[c],
                           assignment.algorithm_1[c])
        efsm.initial_state = 1

        for c in closed_range(1, C):
            for e in closed_range(1, E):
                for k in closed_range(1, K):
                    dest = assignment.transition[c][e][k]
                    if dest != 0:
                        input_event = input_events[e - 1]
                        guard = ParseTreeGuard(assignment.nodetype[c][e][k],
                                               assignment.terminal[c][e][k],
                                               assignment.parent[c][e][k],
                                               assignment.child_left[c][e][k],
                                               assignment.child_right[c][e][k],
                                               names=tree.input_names)
                        efsm.add_transition(c, dest, input_event, guard)

        if efsm.number_of_states != assignment.C:
            log_error(f'Inequal number of states: efsm has {efsm.number_of_states}, assignment has {assignment.C}')
        if efsm.number_of_transitions != assignment.T:
            log_error(f'Inequal number of transitions: efsm has {efsm.number_of_transitions}, assignment has {assignment.T}')
        if efsm.number_of_nodes != assignment.N:
            log_error(f'Inequal number of nodes: efsm has {efsm.number_of_nodes}, assignment has {assignment.N}')

        log_debug(f'Done building EFSM with {efsm.number_of_states} states, {efsm.number_of_transitions} transitions and {efsm.number_of_nodes} nodes in {time.time() - time_start_build:.2f} s')
        return efsm

    @classmethod
    def new_random(cls, C, K, P, input_events, output_events, input_names, output_names, algo0_distribution='01', algo1_distribution='01'):
        assert all(c in '01' for c in algo0_distribution)
        assert all(c in '01' for c in algo1_distribution)

        log_debug('Building random EFSM...')
        time_start_build = time.time()

        E = len(input_events)
        O = len(output_events)
        X = len(input_names)
        Z = len(output_names)

        def add_random_state():
            output_event = random.choice(output_events)
            algo0 = ''.join(random.choice(algo0_distribution) for _ in range(Z))
            algo1 = ''.join(random.choice(algo1_distribution) for _ in range(Z))
            efsm.add_state(c, output_event, algo0, algo1)

        def add_random_transition(c1, c2):
            input_event = random.choice(input_events)
            guard = ParseTreeGuard.new_random(P, names=input_names)
            efsm.add_transition(c1, c2, input_event, guard)

        efsm = cls(input_events, output_events, input_names, output_names)
        efsm.add_state(1, 'INITO', '0' * Z, '1' * Z)
        efsm.initial_state = 1
        for c in closed_range(2, C):
            add_random_state()

        # `q` is a random Hamiltonian path
        q = list(closed_range(2, C))
        random.shuffle(q)
        q.append(random.choice(list(set(closed_range(2, C)) - {q[-1]})))
        for c1, c2 in zip([1] + q, q):
            add_random_transition(c1, c2)
        for c1 in closed_range(1, C):
            for _ in range(random.randint(0, K - 1)):
                c2 = random.choice(closed_range(2, C))  # allow loops
                # c2 = random.choice(list(set(closed_range(2, C)) - {c1}))  # forbid loops
                add_random_transition(c1, c2)

        log_debug(f'Done building random EFSM with C={efsm.C}, K={efsm.K}, P={efsm.P}, T={efsm.T}, N={efsm.N} in {time.time() - time_start_build:.2f} s')
        return efsm

    @property
    def initial_state(self):
        if hasattr(self, '_initial_state'):
            return self._initial_state
        else:
            if self.states:
                return next(iter(self.states.keys()))
            else:
                return None

    @initial_state.setter
    def initial_state(self, value):
        self._initial_state = value

    @property
    def number_of_states(self):
        return len(self.states)

    @property
    def maximum_outgoing_transitions(self):
        return max(len(state.transitions) for state in self.states.values())

    @property
    def number_of_transitions(self):
        return sum(len(state.transitions) for state in self.states.values())

    @property
    def number_of_nodes(self):
        return sum(sum(transition.guard.size()
                       for transition in state.transitions)
                   for state in self.states.values())

    @property
    def guard_condition_maxsize(self):
        return max(transition.guard.size()
                   for state in self.states.values()
                   for transition in state.transitions)

    @property
    def C(self):
        return self.number_of_states

    @property
    def K(self):
        return self.maximum_outgoing_transitions

    @property
    def P(self):
        return self.guard_condition_maxsize

    @property
    def T(self):
        return self.number_of_transitions

    @property
    def N(self):
        return self.number_of_nodes

    @property
    def E(self):
        return len(self.input_events)

    @property
    def O(self):
        return len(self.output_events)

    @property
    def X(self):
        return len(self.input_names)

    @property
    def Z(self):
        return len(self.output_names)

    def add_state(self, id, output_event, algorithm_0, algorithm_1):
        self.states[id] = self.State(id, output_event, algorithm_0, algorithm_1)

    def add_transition(self, src, dest, input_event, guard):
        self.states[src].add_transition(self.states[dest], input_event, guard)

    def go(self, src, input_event, input_values, values):
        source = self.states[src]
        destination, output_event, new_values = source.go(input_event, input_values, values)
        return destination.id, output_event, new_values

    def get_suitable_transition_and_index(self, src, input_event, input_values):
        source = self.states[src]
        return source.get_suitable_transition_and_index(input_event, input_values)

    def random_walk(self, scenario_length, *, input_events, X, Z, distribution='01', preprocess=False):
        assert all(c in '01' for c in distribution)

        scenario = Scenario()
        current_state = self.initial_state
        current_values = '0' * Z

        for _ in range(scenario_length):
            input_event = random.choice(input_events)
            input_values = ''.join(random.choice(distribution) for _ in range(X))
            transition, k = self.get_suitable_transition_and_index(current_state, input_event, input_values)  # Note: k is 1-based, 0 if no transition

            if k == 0:
                scenario.add_element(input_event, input_values, [OutputAction(None, current_values)])
            else:
                destination = transition.destination  # :: State
                output_event = destination.output_event
                new_state = destination.id
                new_values = destination.eval(current_values)
                current_state = new_state
                current_values = new_values
                scenario.add_element(input_event, input_values, [OutputAction(output_event, new_values)])

        if preprocess:
            scenario = Scenario.proprocess(scenario)

        return scenario

    def pprint(self):
        for state in self.states.values():
            log_debug(f'  ┌─{state}', symbol=None)
            if state.transitions:
                for transition in state.transitions[:-1]:
                    log_debug(f'  ├──{transition}', symbol=None)
                log_debug(f'  └──{state.transitions[-1]}', symbol=None)

    def get_gv_string(self):
        s = StringIO()
        s.write('digraph {\n')
        for _x in ['graph', 'node', 'edge']:
            s.write(f'    {_x} [fontname="Source Code Pro,monospace" fontsize="12"]\n')

        s.write('    // States\n')
        s.write('    { node [margin="0.05,0.01"]\n')
        # s.write('    rankdir=LR;\n')
        for state in self.states.values():
            for line in state.__str_gv__().split('\n'):
                s.write('      ' + line + '\n')
        s.write('    }\n')

        s.write('    // Transitions\n')
        for state in self.states.values():
            for k, transition in enumerate(state.transitions, start=1):
                s.write('    ' + transition.__str_gv__(k) + '\n')

        s.write('}\n')
        return s.getvalue()

    def get_fbt_string(self):
        def _r():
            return str(random.randint(1, 1000))

        FBType = etree.Element('FBType')

        etree.SubElement(FBType, 'Identification', Standard='61499-2')
        etree.SubElement(FBType, 'VersionInfo', Organization='nxtControl GmbH', Version='0.0', Author='fbSAT', Date='2011-08-30', Remarks='Template', Namespace='Main', Name='CentralController', Comment='Basic Function Block Type')
        InterfaceList = etree.SubElement(FBType, 'InterfaceList')
        BasicFB = etree.SubElement(FBType, 'BasicFB')

        InputVars = etree.SubElement(InterfaceList, 'InputVars')
        OutputVars = etree.SubElement(InterfaceList, 'OutputVars')
        EventInputs = etree.SubElement(InterfaceList, 'EventInputs')
        EventOutputs = etree.SubElement(InterfaceList, 'EventOutputs')

        # InterfaceList::InputVars declaration
        for input_name in self.input_names:
            etree.SubElement(InputVars, 'VarDeclaration', Name=input_name, Type='BOOL')

        # InterfaceList::OutputVars declaration
        for output_name in self.output_names:
            etree.SubElement(OutputVars, 'VarDeclaration', Name=output_name, Type='BOOL')

        # InterfaceList::EventInputs declaration
        for input_event in ('INIT',) + tuple(self.input_events):
            e = etree.SubElement(EventInputs, 'Event', Name=input_event)
            if input_event != 'INIT':
                for input_name in self.input_names:
                    etree.SubElement(e, 'With', Var=input_name)

        # InterfaceList::EventOutputs declaration
        for output_event in self.output_events:
            e = etree.SubElement(EventOutputs, 'Event', Name=output_event)
            if output_event != 'INITO':
                for name in self.output_names:
                    etree.SubElement(e, 'With', Var=name)

        # BasicFB::ECC declaration
        ECC = etree.SubElement(BasicFB, 'ECC')
        etree.SubElement(ECC, 'ECState', Name='START', x=_r(), y=_r())
        for state in self.states.values():
            s = etree.SubElement(ECC, 'ECState', Name=state.__str_fbt__(), x=_r(), y=_r())
            algorithm = f'{state.algorithm_0}_{state.algorithm_1}'
            etree.SubElement(s, 'ECAction', Algorithm=algorithm, Output=state.output_event)

        etree.SubElement(ECC, 'ECTransition', Source='START', Destination=self.states[self.initial_state].__str_fbt__(), Condition='INIT', x=_r(), y=_r())
        for state in self.states.values():
            for transition in state.transitions:
                etree.SubElement(ECC, 'ECTransition', x=_r(), y=_r(),
                                 Source=transition.source.__str_fbt__(),
                                 Destination=transition.destination.__str_fbt__(),
                                 Condition=transition.__str_fbt__())

        # BasicFB::Algorithms declaration
        algorithms = set((state.algorithm_0, state.algorithm_1) for state in self.states.values())
        for algorithm_0, algorithm_1 in algorithms:
            a = etree.SubElement(BasicFB, 'Algorithm', Name=f'{algorithm_0}_{algorithm_1}')
            st = algorithm2st(self.output_names, algorithm_0, algorithm_1)
            etree.SubElement(a, 'ST', Text=st)

        return etree.tostring(FBType, encoding='UTF-8', xml_declaration=True, pretty_print=True).decode()

    def get_smv_string(self):
        s = StringIO()

        s.write('MODULE main()\n')

        state_names = ', '.join(state.__str_smv__() for state in self.states.values())
        s.write(f'\nVAR _state : {{{state_names}}};\n')

        for input_event in self.input_events:
            s.write(f'VAR {input_event} : boolean;\n')
        for input_name in self.input_names:
            s.write(f'VAR {input_name} : boolean;\n')
        for output_event in self.output_events:
            s.write(f'VAR {output_event} : boolean;\n')
        for output_name in self.output_names:
            s.write(f'VAR {output_name} : boolean;\n')

        s.write('\nASSIGN\n')

        s.write(f'\ninit(_state) := {self.states[self.initial_state].__str_smv__()};\n')
        s.write('\nnext(_state) := case\n')
        for state in self.states.values():
            for transition in state.transitions:
                s.write('    ' + transition.__str_smv_destination__() + '\n')
        s.write('    TRUE: _state;\n')
        s.write('esac;\n')

        for output_event in self.output_events:
            if output_event == 'INITO':
                continue
            s.write(f'\ninit({output_event}) := FALSE;\n')
            s.write(f'\nnext({output_event}) := case\n')
            for state in self.states.values():
                for transition in state.transitions:
                    s.write('    ' + transition.__str_smv_output_event__() + '\n')
            s.write('    TRUE: FALSE;\n')
            s.write('esac;\n')

        for z in closed_range(1, self.Z):
            output_name = self.output_names[z - 1]
            s.write(f'\ninit({output_name}) := FALSE;\n')
            s.write(f'\nnext({output_name}) := case\n')
            d = {'00': [], '01': [], '10': [], '11': []}
            for state in self.states.values():
                for transition in state.transitions:
                    new0 = transition.destination.algorithm_0[z - 1]
                    new1 = transition.destination.algorithm_1[z - 1]
                    d[f'0{new0}'].append(f'next(_state) = {transition.destination.__str_smv__()}')
                    d[f'1{new1}'].append(f'next(_state) = {transition.destination.__str_smv__()}')
            if d['00']:
                s.write(f'    !{output_name} & (' + ' | '.join(d['00']) + ') : FALSE;\n')
            if d['01']:
                s.write(f'    !{output_name} & (' + ' | '.join(d['01']) + ') : TRUE;\n')
            if d['10']:
                s.write(f'    {output_name} & (' + ' | '.join(d['10']) + ') : FALSE;\n')
            if d['11']:
                s.write(f'    {output_name} & (' + ' | '.join(d['11']) + ') : TRUE;\n')
            s.write(f'    TRUE : {output_name};\n')
            s.write('esac;\n')

        return s.getvalue()

    def write_pkl(self, filename):
        log_debug(f'Pickling EFSM to <{filename}>...')

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def write_gv(self, filename):
        log_debug(f'Dumping EFSM in GV format to <{filename}>...')

        with open(filename, 'w') as f:
            f.write(self.get_gv_string())

    def write_fbt(self, filename):
        log_debug(f'Dumping EFSM in FBT format to <{filename}>...')

        with open(filename, 'w') as f:
            f.write(self.get_fbt_string())

    def write_smv(self, filename):
        log_debug(f'Dumping EFSM in SMV format to <{filename}>...')

        with open(filename, 'w') as f:
            f.write(self.get_smv_string())

    def dump(self, prefix):
        filename_pkl = prefix + '.pkl'
        self.write_pkl(filename_pkl)

        filename_gv = prefix + '.gv'
        self.write_gv(filename_gv)

        # for output_format in ['svg', 'pdf']:
        for output_format in ['svg']:
            cmd = f'dot -T{output_format} {filename_gv} -O'
            log_debug(cmd, symbol='$')
            os.system(cmd)

        filename_fbt = prefix + '.fbt'
        self.write_fbt(filename_fbt)

        filename_smv = prefix + '.smv'
        self.write_smv(filename_smv)

    def verify(self, scenario_tree):
        log_info('Verifying...')
        time_start_verify = time.time()
        is_ok = True
        root = scenario_tree[1]

        for i, path in enumerate(scenario_tree.paths_to_leaves(), start=1):
            current_state = self.initial_state  # :: int
            current_values = root.data.output_values

            for j, identifier in enumerate(path[1:], start=1):  # exclude root from path
                element = scenario_tree[identifier].data
                input_event = element.input_event
                input_values = element.input_values
                new_state, output_event, new_values = self.go(current_state, input_event, input_values, current_values)
                # log_debug(f'InputEvent: {input_event}, InputValues: {input_values}, State: {current_state} -> {new_state}, OutputEvent: {output_event}, OutputValues: {current_values} -> {new_values}', symbol=f'{j}/{len(path)-1}')

                if output_event != element.output_event and new_values != element.output_values:
                    log_error(f'incorrect output_event and output_values', symbol=f'{i}:{j}/{len(path)-1}')
                    is_ok = False
                elif output_event != element.output_event:
                    log_error(f'incorrect output_event (oe={output_event} != elem.oe={element.output_event})', symbol=f'{i}:{j}/{len(path)-1}')
                    is_ok = False
                elif new_values != element.output_values:
                    log_error(f'incorrect output_values (ov={new_values} != elem.ov={element.output_values})', symbol=f'{i}:{j}/{len(path)-1}')
                    is_ok = False
                # assert output_event == element.output_event
                # assert new_values == element.output_values
                if output_event != element.output_event or new_values != element.output_values:
                    # log_warn(f'current state: {current_state}')
                    # log_warn(f'new state: {new_state}')
                    # log_warn(f'output event: {output_event}')
                    # log_warn(f'new values: {new_values}')
                    # log_warn('failed scenario:')
                    # for _j, _identifier in enumerate(path[1:], start=1):
                    #     log_debug(f'{scenario_tree[_identifier].data}', symbol=f'{_j: >2}/{len(path)-1}')
                    raise AssertionError()

                current_state = new_state
                current_values = new_values

        log_success(f'Done verifying {scenario_tree.number_of_scenarios} scenario(s) in {time.time() - time_start_verify:.2f} s')
        return is_ok
