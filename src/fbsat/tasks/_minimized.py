import os
import subprocess
import time
from collections import namedtuple

import pymzn

from . import BasicAutomatonTask, MinimalBasicAutomatonTask
from ..efsm import ParseTreeGuard
from ..printers import log_br, log_debug, log_error, log_info, log_success, log_warn
from ..utils import closed_range, s2b

__all__ = ['MinimizeAllGuardsTask']


class MyGecode(pymzn.Solver):
    """Interface to the Gecode solver.
    Parameters
    ----------
    mzn_path : str
        The path to the mzn-gecode executable.
    fzn_path : str
        The path to the fzn-gecode executable.
    globals_dir : str
        The path to the directory for global included files.
    """

    def __init__(
        self, mzn_path='mzn-gecode', fzn_path='fzn-gecode', globals_dir='gecode'
    ):
        super().__init__(
            globals_dir, support_mzn=True, support_all=True, support_num=True,
            support_timeout=True, support_stats=True
        )
        self.mzn_cmd = mzn_path
        self.fzn_cmd = fzn_path

    def args(
        self, mzn_file, *dzn_files, data=None, include=None, timeout=None,
        all_solutions=False, num_solutions=None, output_mode='item', parallel=1,
        seed=0, statistics=False, **kwargs
    ):
        mzn = False
        args = []
        if mzn_file.endswith('fzn'):
            args.append(self.fzn_cmd)
        else:
            mzn = True
            # args += [self.mzn_cmd, '-G', self.globals_dir]
            args += [self.mzn_cmd, '--fzn-cmd', self.fzn_cmd]
            if include:
                if isinstance(include, str):
                    include = [include]
                for path in include:
                    args += ['-I', path]
            if data:
                args += ['-D', data]

        fzn_flags = []
        if statistics:
            args.append('-s')
        if all_solutions:
            args.append('-a')
        if num_solutions is not None:
            args += ['-n', str(num_solutions)]
        if parallel != 1:
            fzn_flags += ['-p', str(parallel)]
        if timeout and timeout > 0:
            timeout = timeout * 1000  # Gecode takes milliseconds
            fzn_flags += ['-time', str(timeout)]
        if seed != 0:
            fzn_flags += ['-r', str(seed)]
        if mzn and fzn_flags:
            args += ['--fzn-flags', '{}'.format(' '.join(fzn_flags))]
        else:
            args += fzn_flags

        args.append(mzn_file)
        if mzn and dzn_files:
            for dzn_file in dzn_files:
                args.append(dzn_file)
        return args

    def solve(self, *args, suppress_segfault=False, **kwargs):
        """Solve a MiniZinc/FlatZinc problem with Gecode.
        Parameters
        ----------
        suppress_segfault : bool
            Whether to accept or not a solution returned when a segmentation
            fault has happened (this is unfortunately necessary sometimes due to
            some bugs in Gecode).
        """
        solver_args = self.args(*args, **kwargs)

        try:
            log_debug('Running solver with arguments {}'.format(solver_args))
            process = pymzn.process.Process(solver_args).run()
            out = process.stdout_data
            err = process.stderr_data
        except subprocess.CalledProcessError as err:
            if suppress_segfault and len(err.stdout) > 0 \
                    and err.stderr.startswith('Segmentation fault'):
                log_warn('Gecode returned error code {} (segmentation '
                         'fault) but a solution was found and returned '
                         '(suppress_segfault=True).'.format(err.returncode))
                out = err.stdout
            else:
                log_error(err.stderr)
                raise RuntimeError(err.stderr) from err
        return out, err


class MinimizeGuardTask:

    def __init__(self, scenario_tree, guard):
        self.scenario_tree = scenario_tree
        self.guard = guard

    def run(self):
        log_debug(f'MinimizeGuardTask: running...')
        time_start_run = time.time()
        minimized_guard = None

        tree = self.scenario_tree
        X = tree.X
        U = tree.U
        unique_inputs = tree.unique_inputs
        guard = self.guard

        input_values = []  # [[bool]] of size (U',X)
        root_value = []  # [bool]
        for u in closed_range(1, U):
            q = guard.truth_table[u - 1]
            iv = unique_inputs[u - 1]
            if q == '0':
                input_values.append(s2b(iv, zero_based=True))
                root_value.append(False)
            elif q == '1':
                input_values.append(s2b(iv, zero_based=True))
                root_value.append(True)
            else:
                log_debug(f'Don\'t care for input {iv}')
        U_ = len(input_values)  # maybe lesser than real tree.U
        log_debug(f'X={X}, U={U}, U_={U_}')
        for P in closed_range(1, 7):
            log_debug(f'Trying P={P}...')
            data = dict(P=P, U=U_, X=X, input_values=input_values, root_value=root_value)
            # pymzn.dict2dzn(data, fout=f'{self.filename_prefix}_C{self.C}_T{self.T}_comb3-bf_c{c}_k{k}_P{P}.dzn')
            solver = MyGecode('minizinc', 'fzn-gecode')
            sols = pymzn.minizinc('minizinc/boolean-function.mzn', data=data, solver=solver,
                                  output_vars=['nodetype', 'terminal', 'parent', 'child_left', 'child_right'])
            try:
                sols._fetch_all()
            except pymzn.MiniZincUnsatisfiableError:
                log_error(f'UNSAT for P={P}!')
            else:
                log_success(f'SAT for P={P}!')
                sol = sols[0]
                log_debug(f'nodetype = {sol["nodetype"]}')
                log_debug(f'terminal = {sol["terminal"]}')
                log_debug(f'parent = {sol["parent"]}')
                log_debug(f'child_left = {sol["child_left"]}')
                log_debug(f'child_right = {sol["child_right"]}')
                new_guard = ParseTreeGuard([None] + sol["nodetype"],
                                           [None] + sol["terminal"],
                                           [None] + sol["parent"],
                                           [None] + sol["child_left"],
                                           [None] + sol["child_right"])
                log_debug(f'Minimized guard: {new_guard}')
                minimized_guard = new_guard
                break

        log_debug(f'MinimizeGuardTask: done in {time.time() - time_start_run:.2f} s')
        return minimized_guard


class MinimizeAllGuardsTask:

    def __init__(self, scenario_tree, basic_automaton=None, C=None, K=None, T=None, *, use_bfs=True, solver_cmd=None, write_strategy=None, outdir=''):
        self.scenario_tree = scenario_tree
        self.basic_automaton = basic_automaton
        self.C = C
        self.K = K
        self.T = T
        self.outdir = outdir
        self.basic_config = dict(use_bfs=use_bfs,
                                 solver_cmd=solver_cmd,
                                 write_strategy=write_strategy,
                                 outdir=outdir)

    def get_stem(self, C, K, T):
        return f'minimized_{self.scenario_tree.scenarios_stem}_C{C}_K{K}_T{T}'

    def get_filename_prefix(self, C, K, T):
        return os.path.join(self.outdir, self.get_stem(C, K, T))

    def run(self):
        log_debug('MinimizeAllGuardsTask: running...')
        time_start_run = time.time()

        if self.basic_automaton:
            automaton = self.basic_automaton
        else:
            log_debug('MinimizeAllGuardsTask: building basic automaton...')
            if self.T is not None:
                config = dict(scenario_tree=self.scenario_tree,
                              C=self.C, K=self.K,
                              **self.basic_config)
                task = BasicAutomatonTask(**config)
                automaton = task.run(self.T)
            else:
                if self.K is not None:
                    log_warn(f'Ignoring specified K={self.K}')
                config = dict(scenario_tree=self.scenario_tree,
                              C=self.C,
                              **self.basic_config)
                task = MinimalBasicAutomatonTask(**config)
                automaton = task.run()

        if automaton:
            log_br()
            log_info('MinimizeAllGuardsTask: minimizing guards...')
            C = automaton.number_of_states
            if self.K is None or self.T:
                K = C
            else:
                K = self.K
            T = automaton.number_of_transitions

            for c in closed_range(1, C):
                state = automaton.states[c]
                for k, transition in enumerate(state.transitions, start=1):
                    log_br()
                    log_info(f'MinimizeAllGuardsTask: minimizing guard on transition k={k} from state c={c}...')
                    guard = transition.guard
                    task = MinimizeGuardTask(self.scenario_tree, guard)
                    minimized_guard = task.run()
                    if minimized_guard:
                        log_debug(f'MinimizeAllGuardsTask: guard on transition k={k} from state c={c} was minimized to {minimized_guard}')
                        transition.guard = minimized_guard
                    else:
                        log_debug(f'MinimizeAllGuardsTask: guard on transition k={k} from state c={c} was not minimized')

            automaton.dump(self.get_filename_prefix(C, K, T))

            log_success('Automaton with minimized guards:')
            automaton.pprint()
            automaton.verify()

        log_debug(f'MinimizeAllGuardsTask: done in {time.time() - time_start_run:.2f} s')
        log_br()
        if automaton:
            log_success('')
        else:
            log_error('')
        return automaton
