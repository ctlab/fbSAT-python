__all__ = ('Guard', 'EFSM')

import time
from collections import OrderedDict

from lxml import etree

from .utils import *
from .printers import *


class Guard:

    class Node:

        predicate_names = 'c1Home c1End c2Home c2End vcHome vcEnd pp1 pp2 pp3 vac'.split()
        output_variable_names = 'c1Extend c1Retract c2Extend c2Retract vcExtend vacuum_on vacuum_off'.split()

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
                return self.predicate_names[self.terminal - 1]
            elif self.nodetype == 1:  # AND
                return f'({self.child_left} & {self.child_right})'
            elif self.nodetype == 2:  # OR
                return f'({self.child_left} | {self.child_right})'
            elif self.nodetype == 3:  # NOT
                # FIXME: why do we need this dispatch? Seems like first branch is enough
                if self.child_left.nodetype == 0:
                    return f'~{self.child_left}'
                else:
                    return f'~({self.child_left})'
            elif self.nodetype == 4:  # None
                raise ValueError(f'why are you trying to display None-typed node?')

        def __str_nice__(self):
            if self.nodetype == 0:  # Terminal
                return str(self)
            elif self.nodetype == 1:  # AND
                return f'({self.child_left.__str_nice__()} AND {self.child_right.__str_nice__()})'
            elif self.nodetype == 2:  # OR
                return f'({self.child_left.__str_nice__()} OR {self.child_right.__str_nice__()})'
            elif self.nodetype == 3:  # NOT
                return f'NOT {self.child_left.__str_nice__()}'
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

    @classmethod
    def from_input(cls, input_values):
        # input_values :: [1..X]:Bool
        self = cls.__new__(cls)
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

    def eval(self, input_values):
        # input_values :: [1..X]:Bool
        return self.root.eval(input_values)

    def size(self):
        return self.root.size()

    def __str__(self):
        return str(self.root)

    def __str_nice__(self):
        return self.root.__str_nice__()


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
        return destination.id, output_event, new_values

    def get_suitable_transition_and_index(self, src, input_event, input_values):
        source = self.states[src]
        return source.get_suitable_transition_and_index(input_event, input_values)

    def pprint(self):
        for state in self.states.values():
            log_debug(state)
            if state.transitions:
                for transition in state.transitions[:-1]:
                    log_debug(f'├──{transition}')
                log_debug(f'└──{state.transitions[-1]}')

    def get_gv_string(self):
        state_numbers = {state: i for i, state in enumerate(self.states.values())}
        lines = ['digraph {',
                 '    // States',
                 '    { node []',
                 *(f'      {state_numbers[state]} [label="{state.id}: {state.output_event}({state.algorithm_0}_{state.algorithm_1})"]'
                   for state in self.states.values()),
                 '    }',
                 '    // Transitions',
                 *(f'    {state_numbers[transition.source]} -> {state_numbers[transition.destination]} [label="{transition.input_event} [{transition.guard}]"]'
                     for state in self.states.values() for transition in state.transitions),
                 '}']
        return '\n'.join(lines)

    def get_fbt_string(self, tree):
        FBType = etree.Element('FBType')

        etree.SubElement(FBType, 'Identification', Standard='61499-2')
        etree.SubElement(FBType, 'VersionInfo', Organization='nxtControl GmbH', Version='0.0', Author='fbSAT', Date='2011-08-30', Remarks='Template')
        InterfaceList = etree.SubElement(FBType, 'InterfaceList')
        BasicFB = etree.SubElement(FBType, 'BasicFB')

        InputVars = etree.SubElement(InterfaceList, 'InputVars')
        OutputVars = etree.SubElement(InterfaceList, 'OutputVars')
        EventInputs = etree.SubElement(InterfaceList, 'EventInputs')
        EventOutputs = etree.SubElement(InterfaceList, 'EventOutputs')

        # InterfaceList::InputVars declaration
        for input_variable_name in tree.predicate_names:
            etree.SubElement(InputVars, 'VarDeclaration', Name=input_variable_name, Type='BOOL')

        # InterfaceList::OutputVars declaration
        for output_variable_name in tree.output_variable_names:
            etree.SubElement(OutputVars, 'VarDeclaration', Name=output_variable_name, Type='BOOL')

        # InterfaceList::EventInputs declaration
        for input_event in tree.input_events:
            e = etree.SubElement(EventInputs, 'Event', Name=input_event)
            if input_event != 'INIT':
                for input_variable_name in tree.predicate_names:
                    etree.SubElement(e, 'With', Var=input_variable_name)

        # InterfaceList::EventOutputs declaration
        for output_event in tree.output_events:
            e = etree.SubElement(EventOutputs, 'Event', Name=output_event)
            if output_event != 'INITO':
                for output_variable_name in tree.output_variable_names:
                    etree.SubElement(e, 'With', Var=output_variable_name)

        # BasicFB::ECC declaration
        ECC = etree.SubElement(BasicFB, 'ECC')
        etree.SubElement(ECC, 'ECState', Name='START', Comment='Initial state')
        for state in self.states.values():
            s = etree.SubElement(ECC, 'ECState', Name=f's_{state.id}')
            etree.SubElement(s, 'ECAction', Algorithm=f'{state.output_event}_{state.algorithm_0}_{state.algorithm_1}')

        etree.SubElement(ECC, 'ECTransition', Source='START', Destination=f's_1', Condition='INIT')
        for state in self.states.values():
            for transition in state.transitions:
                etree.SubElement(ECC, 'ECTransition',
                                 Source=f's_{transition.source.id}',
                                 Destination=f's_{transition.destination.id}',
                                 Condition=f'{transition.input_event}&{transition.guard.__str_nice__()}')

        # BasicFB::Algorithms declaration
        algorithms = set()
        for state in self.states.values():
            algorithms.add((state.output_event, state.algorithm_0, state.algorithm_1))
        for output_event, algorithm_0, algorithm_1 in algorithms:
            a = etree.SubElement(BasicFB, 'Algorithm', Name=f'{output_event}_{algorithm_0}_{algorithm_1}')
            etree.SubElement(a, 'ST', Text=f'{output_event} := FALSE;\n{algorithm2st(algorithm_0, algorithm_1)}\n')

        return etree.tostring(FBType, encoding='UTF-8', xml_declaration=True, pretty_print=True).decode()

    def write_gv(self, filename):
        log_debug(f'Dumping EFSM to <{filename}>...')

        with open(filename, 'w') as f:
            f.write(self.get_gv_string() + '\n')

    def write_fbt(self, filename):
        log_debug(f'Dumping EFSM to <{filename}>...')

        with open(filename, 'w') as f:
            f.write(self.get_fbt_string())

    def verify(self, scenario_tree):
        log_info('Verifying...')
        time_start_verify = time.time()

        tree = scenario_tree
        root = tree[1]

        for path in tree.paths_to_leaves():
            current_state = self.initial_state  # :: int
            current_values = root.data.output_values

            for j, identifier in enumerate(path[1:], start=1):  # exclude root from path
                element = tree[identifier].data
                input_event = element.input_event
                input_values = element.input_values
                new_state, output_event, new_values = self.go(current_state, input_event, input_values, current_values)
                # log_debug(f'InputEvent: {input_event}, InputValues: {input_values}, State: {current_state} -> {new_state}, OutputEvent: {output_event}, OutputValues: {current_values} -> {new_values}', symbol=f'{j}/{len(path)-1}')

                if output_event != element.output_event and new_values != element.output_values:
                    log_error(f'incorrect output_event and output_values', symbol=f'{j}/{len(path)-1}')
                elif output_event != element.output_event:
                    log_error(f'incorrect output_event (oe={output_event} != elem.oe={element.output_event})', symbol=f'{j}/{len(path)-1}')
                elif new_values != element.output_values:
                    log_error(f'incorrect output_values (ov={new_values} != elem.ov={element.output_values})', symbol=f'{j}/{len(path)-1}')
                assert output_event == element.output_event
                assert new_values == element.output_values

                current_state = new_state
                current_values = new_values

        log_success(f'Done verifying in {time.time() - time_start_verify:.2f} s')
        log_br()
