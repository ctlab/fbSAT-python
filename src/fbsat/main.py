import pickle
import time

import click

from .printers import *
from .scenario import *
from .tasks import *
from .utils import *
from .version import version as __version__


@click.command(context_settings=dict(
        max_content_width=999,
        help_option_names=['-h', '--help']
))
@click.option('-i', '--scenarios', 'filename_scenarios', metavar='<path>', required=True,
              type=click.Path(exists=True),
              help='File with scenarios')
@click.option('-o', '--outdir', 'outdir', metavar='<path>',
              type=click.Path(file_okay=False, writable=True),
              default='out', show_default=True,
              help='Output directory')
@click.option('-m', '--method', type=click.Choice(['full', 'full-min',
                                                   'basic', 'basic-min',
                                                   'extended', 'extended-min', 'extended-min-ub',
                                                   'minimize']), required=True,
              help='Method to use')
@click.option('--input-names', 'input_names', metavar='<x.../path>', callback=parse_names,
              help='Comma-separated list of input variable names, or a filename')
@click.option('--output-names', 'output_names', metavar='<z.../path>', callback=parse_names,
              help='Comma-separated list of output variable names, or a filename')
@click.option('--automaton', metavar='<path>',
              type=click.Path(exists=True),
              help='[minimize] File with pickled automaton')
@click.option('-C', 'C', type=int, metavar='<int>',
              help='Number of automaton states')
@click.option('-K', 'K', type=int, metavar='<int>',
              help='Maximum number of transitions from each state')
@click.option('-P', 'P', type=int, metavar='<int>',
              help='Maximum number of nodes in guard\'s boolean formula\'s parse tree')
@click.option('-T', 'T', type=int, metavar='<int>',
              help='Upper bound on total number of transitions')
@click.option('-N', 'N', type=int, metavar='<int>',
              help='Upper bound on total number of nodes in all guard-trees')
@click.option('-w', 'w', type=int, metavar='<int>',
              help='[ext-min-ub] Maximum width of local minima')
@click.option('--bfs/--no-bfs', 'use_bfs',
              default=True, show_default=True,
              help='Use BFS symmetry-breaking constraints')
@click.option('--distinct/--no-distinct', 'is_distinct',
              default=False, show_default=True,
              help='Distinct transitions')
@click.option('--forbid-or/--no-forbid-or', 'is_forbid_or',
              default=False, show_default=True,
              help='[ext] Forbid OR parse tree nodes')
@click.option('--sat-solver', metavar='<cmd>',
              # default='glucose -model -verb=0', show_default=True,
              default='cryptominisat5 --verb=0', show_default=True,
              # default='cadical -q', show_default=True,
              help='SAT solver')
# TODO: feature switches
# TODO: add '--stream'
@click.option('--incremental', 'is_incremental', is_flag=True,
              help='Use IncrementalSolver backend')
# TODO: replace with '--tempfile'
@click.option('--filesolver', 'is_filesolver', is_flag=True,
              help='Use FileSolver backend')
@click.version_option(__version__)
def cli(filename_scenarios, outdir, method, input_names, output_names, automaton,
        C, K, P, T, N, w,
        use_bfs, is_distinct, is_forbid_or,
        sat_solver, is_incremental, is_filesolver):
    log_info('Welcome!')
    time_start = time.time()
    # =====================
    # import base64
    # import zlib
    # log_debug(zlib.decompress(base64.b64decode(b'eNqVUUEOwCAIu/MKnrvjSMaWLOFzvmRqFgcKixoOWlppFTCvdG5O8eWCIRunubUHajIfiXdV0uOdo3k+3xZpQgEvebbseS4yWXilgpOKR35YGbHRuMEp4MfDSmc5Eyn1z4jXo6A+mX9x0fphn1Zf0/YPKdybbQ==')).decode(), symbol=None)
    # =====================

    log_br()
    log_info('Building scenario tree...')
    time_start_tree = time.time()
    scenarios = Scenario.read_scenarios(filename_scenarios)
    scenario_tree = ScenarioTree(scenarios, input_names=input_names, output_names=output_names)
    log_success(f'Successfully built scenario tree of size {scenario_tree.size()}'
                f' in {time.time() - time_start_tree:.2f} s')

    if method == 'full':
        log_info(f'{method} method')
        if K is not None:
            log_warn(f'Ignoring specified K={K}')
        if P is not None:
            log_warn(f'Ignoring specified P={P}')
        if N is not None:
            log_warn(f'Ignoring specified N={N}')
        config = dict(scenario_tree=scenario_tree,
                      C=C,
                      use_bfs=use_bfs,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        task = FullAutomatonTask(**config)
        efsm = task.run(T)

    elif method == 'full-min':
        log_info(f'{method} method')
        if K is not None:
            log_warn(f'Ignoring specified K={K}')
        if P is not None:
            log_warn(f'Ignoring specified P={P}')
        if N is not None:
            log_warn(f'Ignoring specified N={N}')
        config = dict(scenario_tree=scenario_tree,
                      C=C, T=T,
                      use_bfs=use_bfs,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        task = MinimalFullAutomatonTask(**config)
        efsm = task.run()

    elif method == 'basic':
        log_info(f'{method} method')
        if P is not None:
            log_warn(f'Ignoring specified P={P}')
        if N is not None:
            log_warn(f'Ignoring specified N={N}')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        task = BasicAutomatonTask(**config)
        efsm = task.run(T)

    elif method == 'basic-min':
        log_info(f'{method} method')
        if P is not None:
            log_warn(f'Ignoring specified P={P}')
        if N is not None:
            log_warn(f'Ignoring specified N={N}')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, T_init=T,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        task = MinimalBasicAutomatonTask(**config)
        efsm = task.run()

    elif method == 'extended':
        log_info(f'{method} method')
        if T is not None:
            log_warn(f'Ignoring specified T={T}')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, P=P,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      is_forbid_or=is_forbid_or,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        task = ExtendedAutomatonTask(**config)
        efsm = task.run(N)

    elif method == 'extended-min':
        log_info(f'{method} method')
        if T is not None:
            log_warn(f'Ignoring specified T={T}')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, P=P, N_init=N,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      is_forbid_or=is_forbid_or,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        task = MinimalExtendedAutomatonTask(**config)
        efsm = task.run()

    elif method == 'extended-min-ub':
        log_info(f'{method} method')
        if P is not None:
            log_warn(f'Ignoring specified P={P}')
        if N is not None:
            log_warn(f'Ignoring specified N={N}')
        if T is not None:
            log_warn(f'Ignoring specified T={T}')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, w=w,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      is_forbid_or=is_forbid_or,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        task = MinimalExtendedUBAutomatonTask(**config)
        efsm = task.run()

    elif method == 'minimize':
        log_info(f'{method} method')
        if P is not None:
            log_warn(f'Ignoring specified P={P}')
        if N is not None:
            log_warn(f'Ignoring specified N={N}')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, T=T,
                      use_bfs=use_bfs,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        if automaton:
            with open(automaton, 'rb') as f:
                log_debug(f'Loading pickled automaton from {automaton}...')
                config['basic_automaton'] = pickle.load(f)
        task = MinimizeAllGuardsTask(**config)
        efsm = task.run()

    log_br()
    log_success(f'All done in {time.time() - time_start:.2f} s')
