__all__ = ('Instance', )

import os
import time
import regex
import shutil
import tempfile
import pickle
import itertools
import subprocess
from io import StringIO
from collections import namedtuple

from .efsm import *
from .utils import *
from .printers import *


class Instance:

    def __init__(self, *, scenario_tree, efsm, mzn_solver=None, filename_prefix='', write_strategy='StringIO', is_reuse=False):
        self.scenario_tree = scenario_tree
        self.efsm = efsm
        self.mzn_solver = mzn_solver
        self.filename_prefix = filename_prefix
        self.write_strategy = write_strategy
        self.is_reuse = is_reuse

    def run(self):
        self.meat()
        # self.solve()

    def meat(self):
        tree = self.scenario_tree
        efsm = self.efsm
        root = tree[1]
        C = len(efsm.states)
        X = tree.X

        history = [None] + [dict() for _ in range(C)]  # for every c from [1..C]: {input_values: k} of size D_c

        for path in tree.paths_to_leaves():
            current_state = efsm.initial_state  # :: int
            current_values = root.data.output_values

            for element in (tree[i].data for i in path[1:]):
                input_event = element.input_event
                input_values = element.input_values

                transition, k = efsm.get_suitable_transition_and_index(current_state, input_event, input_values)  # Note: 1-based, 0 if no transition
                assert input_values not in history[current_state] or history[current_state][input_values] == k, f'input_values = {input_values}, k = {k}, but history[{current_state}][{input_values}] = {history[current_state][input_values]} != {k}'
                history[current_state][input_values] = k

                if k > 0:
                    destination = transition.destination  # :: State
                    new_state = destination.id
                    new_values = destination.eval(current_values)
                    current_state = new_state
                    current_values = new_values

        # ===========
        log_debug('History:')
        for c in closed_range(1, C):
            log_debug(f'[{c}/{C}] [D={len(history[c])}] {history[c]}', symbol=None)
        # ===========

        ps = []

        for c in closed_range(1, C):
            D = len(history[c])
            K = len(efsm.states[c].transitions)
            for k in closed_range(1, K):
                log_info(f'Solving state {c}/{C}, transition {k}/{K}')
                for P in itertools.islice(itertools.count(1), 10):
                    log_info(f'Trying P = {P}')
                    filename = f'{self.filename_prefix}_C{C}_c{c}_K{K}_k{k}_P{P}.mzn'
                    with open(filename, 'w') as f:
                        f.write(f'% C = {C};\n')
                        f.write(f'% c = {c};\n')
                        f.write(f'K = {K};\n')
                        f.write(f'k = {k};\n')
                        f.write(f'P = {P};\n')
                        f.write(f'X = {X};\n')
                        f.write(f'D = {D};\n')
                        f.write(f'tran_id = [{", ".join(map(str, history[c].values()))}];\n')
                        f.write(f'inputs = [| ')
                        f.write(',\n          | '.join(', '.join({'0': 'false', '1': ' true'}[v] for v in iv) for iv in history[c]))
                        f.write(f' |]\n')

                    cmd = f'{self.mzn_solver} minizinc/one-transition-hybrid.mzn {filename}'
                    # cmd = f'mzn2fzn minizinc/one-transition-hybrid.mzn {filename} -O x.ozn -o x.fzn && fzn-gecode x.fzn | solns2out x.ozn && rm x.fzn x.ozn'
                    log_debug(cmd)
                    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       universal_newlines=True)
                    ok = False
                    for line in p.stdout.split('\n'):
                        if 'UNSAT' in line:
                            log_error('UNSAT!')
                            break
                        elif 'UNKNOWN' in line:
                            if 'TIMEOUT' in p.stdout:
                                log_error('TIMEOUT!')
                            else:
                                log_error('UNKNOWN!')
                            break
                        elif '------' in line:
                            ok = True
                            break
                        elif line.startswith('# nodetype = '):
                            nodetype = [None] + list(map(int, line.split(' = ')[1][1:-1].split(', ')))
                        elif line.startswith('# terminal = '):
                            terminal = [None] + list(map(int, line.split(' = ')[1][1:-1].split(', ')))
                        elif line.startswith('# parent = '):
                            parent = [None] + list(map(int, line.split(' = ')[1][1:-1].split(', ')))
                        elif line.startswith('# left-child = '):
                            child_left = [None] + list(map(int, line.split(' = ')[1][1:-1].split(', ')))
                    else:
                        log_error('Maybe UNSAT!')

                    if ok:
                        ps.append(P)
                        log_success(f'OK for state {c}/{C} and transition {k}/{K} with P = {P}')
                        # log_debug(f'Nodetype: {nodetype}')
                        # log_debug(f'Terminal: {terminal}')
                        # log_debug(f'Parent: {parent}')
                        # log_debug(f'Left child: {child_left}')
                        child_right = [None] + [child_left[p] + 1 if nodetype[p] in (1, 2) else 0 for p in closed_range(1, P)]
                        guard = Guard(nodetype, terminal, parent, child_left, child_right)
                        efsm.states[c].transitions[k - 1].guard = guard
                        break
                else:
                    log_warn('Sorry, can\'t to find P :c')
                log_br()

        # =======================
        efsm.pprint()
        # =======================

        log_debug(f'Ps: {ps}')
        log_debug(f'Max P: {max(ps)}')
        log_debug(f'Sum P: {sum(ps)}')
        log_br()

        log_debug('Dumping minimized efsm...')
        filename_automaton = f'{self.filename_prefix}_automaton_minimized'
        with open(filename_automaton, 'wb') as f:
            pickle.dump(efsm, f, pickle.HIGHEST_PROTOCOL)

        filename_gv = f'{self.filename_prefix}_C{C}_K{max(len(state.transitions) for state in efsm.states.values())}_efsm_minimized.gv'
        os.makedirs(os.path.dirname(filename_gv), exist_ok=True)
        efsm.write_gv(filename_gv)

        output_format = 'svg'
        cmd = f'dot -T{output_format} {filename_gv} -O'
        log_debug(cmd, symbol='$')
        os.system(cmd)

        efsm.verify(self.scenario_tree)

        log_br()
