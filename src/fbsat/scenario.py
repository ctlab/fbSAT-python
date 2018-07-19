import os
import time
import itertools

import regex
import click
import treelib

from .efsm import ParseTreeGuard
from .utils import *
from .printers import *

__all__ = ['ScenarioTree']


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

        self.output_event = self.output_actions[0].output_event
        self.output_values = self.output_actions[0].output_values

    @property
    def output_event(self):
        return self.output_actions[0].output_event

    @output_event.setter
    def output_event(self, value):
        self.output_actions[0].output_event = value

    @property
    def output_values(self):
        return self.output_actions[0].output_values

    @output_values.setter
    def output_values(self, value):
        self.output_actions[0].output_values = value

    def __eq__(self, other):
        if isinstance(other, ScenarioElement):
            return (self.input_event, self.input_values, self.output_event, self.output_values) == \
                (other.input_event, other.input_values, other.output_event, other.output_values)
        return NotImplemented

    def __str__(self):
        return f'{self.input_event}({self.input_values})->{self.output_actions[0]}'

    def __repr__(self):
        return f'{self.__name__}(input_event={self.input_event}, input_values={self.input_values}, output_actions={self.output_actions})'


class Scenario:
    def __init__(self):
        self.elements = []  # [ScenarioElement]

    def add_element(self, input_event, input_values, output_actions):
        self.elements.append(ScenarioElement(input_event, input_values, output_actions))

    @classmethod
    def preprocess(cls, scenario):
        processed = cls()  # new Scenario
        last = scenario.elements[0]
        processed.elements.append(last)

        for element in scenario.elements[1:]:
            if last is None or element != last:
                processed.elements.append(element)
            # else:
            #     print(f'skipping because elem = {element}, last = {last}')
            last = element

        return processed

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        yield from self.elements

    def __repr__(self):
        return f'<Scenario {self.elements}>'

    @staticmethod
    def read_scenarios(filename):
        log_debug(f'Reading scenarios from <{click.format_filename(filename)}>...')
        time_start_reading = time.time()

        scenarios = []  # [Scenario]

        with open_maybe_gzip(filename) as f:
            # number_of_scenarios = int(f.readline())
            f.readline()

            for line in map(str.strip, f):
                scenario = Scenario()

                LocalState = type('', (object,), dict(last_output_values='0' * len(regex.search(r'out=.*?\[([10]*)\]', line).group(1))))

                for m in regex.finditer(r'(?:in=(?P<input_event>.*?)\[(?P<input_values>[10]*)\];\s*)+(?:out=(?P<output_event>.*?)\[(?P<output_values>[10]*)\];\s*)*', line):
                    t = list(zip(*m.captures('input_event', 'input_values')))

                    for input_event, input_values in t[:-1]:
                        scenario.add_element(input_event, input_values, [OutputAction(None, LocalState.last_output_values)])

                    if len(m.captures('output_event')) == 0:
                        continue

                    input_event, input_values = t[-1]
                    output_actions = []
                    for output_event, output_values in zip(*m.captures('output_event', 'output_values')):
                        output_actions.append(OutputAction(output_event, output_values))
                    LocalState.last_output_values = output_values
                    scenario.add_element(input_event, input_values, output_actions)

                scenarios.append(scenario)

        log_debug(f'Done reading {len(scenarios)} scenarios with total {sum(map(len, scenarios))} elements in {time.time() - time_start_reading:.2f} s')
        return scenarios

    @staticmethod
    def preprocess_scenarios(scenarios_full):
        log_debug(f'Preprocessing scenarios...')
        time_start_preprocessing = time.time()

        scenarios = list(map(Scenario.preprocess, scenarios_full))

        log_debug(f'Done preprocessing {len(scenarios)} scenarios from {sum(map(len, scenarios_full))} down to {sum(map(len, scenarios))} elements in {time.time() - time_start_preprocessing:.2} s')
        return scenarios


class ScenarioTree(treelib.Tree):

    def __init__(self, scenarios):
        super().__init__()

        clock = itertools.count(start=1)
        root = self.create_node(identifier=next(clock), data=ScenarioElement(None, None, [OutputAction('INITO', '')]))
        for scenario in scenarios:
            current = root
            for element in scenario.elements:
                for child in self.children(current.identifier):
                    if (child.data.input_event, child.data.input_values) == (element.input_event, element.input_values):
                        current = child
                        break
                else:
                    current = self.create_node(identifier=next(clock),
                                               parent=current,
                                               data=element)
        root.data.output_values = '0' * len(current.data.output_values)
        assert all(len(node.data.output_values) == len(root.data.output_values) for node in self.all_nodes_itr())

        self.input_events = tuple(sorted(set(filter(None, (node.data.input_event for node in self.all_nodes_itr())))))
        self.output_events = tuple(sorted(set(filter(None, (node.data.output_event for node in self.all_nodes_itr())))))
        self.unique_inputs = tuple(sorted(set(filter(None, (node.data.input_values for node in self.all_nodes_itr())))))
        self.unique_outputs = tuple(sorted(set(filter(None, (node.data.output_values for node in self.all_nodes_itr())))))

        self.V = self.size()
        self.E = len(self.input_events)
        self.O = len(self.output_events)
        self.X = len(self.unique_inputs[0])
        self.Z = len(self.unique_outputs[0])
        self.U = len(self.unique_inputs)
        self.Y = len(self.unique_outputs)

        def make_array(d):
            return [None for _ in closed_range(d)]

        def make_array2(d1, d2):
            return [[None for _ in closed_range(d2)] for _ in closed_range(d1)]

        self.parent = make_array(self.V)  # 0..V
        self.previous_active = make_array(self.V)  # 0..V
        self.input_event = make_array(self.V)  # 1..E
        self.output_event = make_array(self.V)  # 0..O
        self.input_number = make_array(self.V)  # 1..U
        self.output_number = make_array(self.V)  # 1..Y
        self.unique_input = make_array2(self.U, self.X)  # bool
        self.unique_output = make_array2(self.Y, self.Z)  # bool
        self.output_value = make_array2(self.V, self.Z)  # bool

        # self.tree_parent[1] = 0
        # self.tree_previous_active[1] = 0
        # self.tree_input_event[1] = 0
        self.output_event[1] = self.output_events.index(self[1].data.output_event) + 1
        # self.input_number[1] = 0
        self.output_number[1] = self.unique_outputs.index(self[1].data.output_values) + 1

        for v in closed_range(2, self.V):
            node = self[v]
            self.parent[v] = node.bpointer
            self.input_event[v] = self.input_events.index(node.data.input_event) + 1
            if node.data.output_event is not None:
                self.output_event[v] = self.output_events.index(node.data.output_event) + 1
            else:
                self.output_event[v] = 0
            self.input_number[v] = self.unique_inputs.index(node.data.input_values) + 1
            self.output_number[v] = self.unique_outputs.index(node.data.output_values) + 1

        for v in closed_range(2, self.V):
            parent = self.parent[v]
            while parent != 1 and self.output_event[parent] == 0:
                parent = self.parent[parent]
            self.previous_active[v] = parent

        for u in closed_range(1, self.U):
            for x, c in enumerate(self.unique_inputs[u - 1], start=1):
                self.unique_input[u][x] = {'0': False, '1': True}[c]
        for y in closed_range(1, self.Y):
            for z, c in enumerate(self.unique_outputs[y - 1], start=1):
                self.unique_output[y][z] = {'0': False, '1': True}[c]

        for v in closed_range(1, self.V):
            for z, c in enumerate(self[v].data.output_values, start=1):
                self.output_value[v][z] = {'0': False, '1': True}[c]

    @staticmethod
    def from_files(filename_scenarios, filename_predicate_names, filename_output_variable_names, preprocess=True):
        scenarios = Scenario.read_scenarios(filename_scenarios)
        if preprocess:
            scenarios = Scenario.preprocess_scenarios(scenarios)
        tree = ScenarioTree(scenarios)
        tree.predicate_names = read_names(filename_predicate_names)
        tree.output_variable_names = read_names(filename_output_variable_names)
        tree.scenarios_filename = filename_scenarios
        # ===========
        # FIXME: dirty
        ParseTreeGuard.Node.predicate_names = tree.predicate_names
        ParseTreeGuard.Node.output_variable_names = tree.output_variable_names
        # ===========
        # ===========
        # FIXME: adhoc for algorithm2st
        GlobalState['predicate_names'] = tree.predicate_names
        GlobalState['output_variable_names'] = tree.output_variable_names
        # ===========
        return tree

    @property
    def predicate_names(self):
        if hasattr(self, '_predicate_names'):
            return self._predicate_names
        else:
            return list(map(lambda x: f'x{x}', closed_range(1, self.X)))

    @predicate_names.setter
    def predicate_names(self, value):
        self._predicate_names = value

    @property
    def output_variable_names(self):
        if hasattr(self, '_output_variable_names'):
            return self._output_variable_names
        else:
            return list(map(lambda z: f'z{z}', closed_range(1, self.Z)))

    @output_variable_names.setter
    def output_variable_names(self, value):
        self._output_variable_names = value

    @property
    def scenarios_stem(self):
        if hasattr(self, 'scenarios_filename'):
            return os.path.splitext(os.path.basename(self.scenarios_filename))[0]
        else:
            return 'unknown'

    def pprint(self, n=None):
        log_debug('Scenario tree:')
        for v in closed_range(1, self.size() if n is None else min(n, self.size())):
            log_debug(f'    {self[v]}', symbol=None)
