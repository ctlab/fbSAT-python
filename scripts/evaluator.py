import json
import pathlib
import pickle
import time

import click

from fbsat.printers import *
from fbsat.tasks import *

from pathutils import ensure_dir


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
))
@click.option('-i', '--input', 'indir', metavar='<path>',
              type=click.Path(exists=True, file_okay=False), required=True,
              help='Input folder with scenarios (e.g., simulation/replica0/scenarios')
@click.option('-o', '--output', 'outdir', metavar='<path>',
              type=click.Path(writable=True, file_okay=False), required=True,
              help='Output folder for evaluation results (e.g., simulation/replica0/out/<method>)')
@click.option('--method', metavar='<method>', required=True,
              type=click.Choice(['full', 'full-min',
                                 'basic', 'basic-min',
                                 'extended', 'extended-min', 'extended-min-ub']),
              help='Method')
@click.option('-C', 'C', type=int, metavar='<int>',
              help='Number of colors (automata states)')
@click.option('-K', 'K', type=int, metavar='<int>',
              help='Maximum number of transitions from each state')
@click.option('-P', 'P', type=int, metavar='<int>',
              help='Maximum number of nodes in guard\'s boolean formula\'s parse tree')
@click.option('-N', 'N', type=int, metavar='<int>',
              help='Upper bound on total number of nodes in all guard-trees')
@click.option('-T', 'T', type=int, metavar='<int>',
              help='Upper bound on total number of transitions')
@click.option('-w', 'w', type=int, metavar='<int>',
              help='[complete-min-ub] Maximum width of local minimum')
@click.option('--bfs/--no-bfs', 'use_bfs',
              default=True, show_default=True,
              help='Use BFS symmetry-breaking constraints')
@click.option('--distinct/--no-distinct', 'is_distinct',
              default=False, show_default=True,
              help='Force distinct transitions')
@click.option('--forbid-or/--no-forbid-or', 'is_forbid_or',
              default=False, show_default=True,
              help='[extended] Forbid OR')
@click.option('--sat-solver', metavar='<cmd>',
              default='cryptominisat5 --verb=0', show_default=True,
              help='SAT solver')
@click.option('--incremental', 'is_incremental', is_flag=True,
              help='Use IncrementalSolver backend')
@click.option('--filesolver', 'is_filesolver', is_flag=True,
              help='Use FileSolver backend')
@click.option('--exist-err', 'exist', flag_value='err', default=True,
              help='Disallow writing in existing folder')
@click.option('--exist-ok', 'exist', flag_value='ok',
              help='Allow writing in existing folder')
@click.option('--exist-rm', 'exist', flag_value='rm',
              help='Remove everything in existing folder')
@click.option('--exist-rm-files', 'exist', flag_value='rm-files',
              help='Remove all files in existing folder recursively')
@click.option('--exist-re', 'exist', flag_value='re',
              help='Recreate existing folder')
def cli(indir, outdir, method, C, K, P, T, N, w, use_bfs, is_distinct, is_forbid_or, sat_solver, is_incremental, is_filesolver, exist):
    time_start_evaluate = time.time()

    path_input = pathlib.Path(indir)
    assert path_input.is_dir()

    path_output = pathlib.Path(outdir)
    ensure_dir(path_output, exist)

    # Load scenarios info
    path_scenarios_info = path_input / 'info_scenarios.json'
    if not path_scenarios_info.exists():
        raise click.BadParameter(f'folder does not contain {path_scenarios_info.name}', param_hint='input')
    with path_scenarios_info.open() as f:
        log_debug(f'Loading scenarios info from <{path_scenarios_info!s}>...')
        scenarios_info = json.load(f)
        log_debug(f'scenarios_info: {scenarios_info}')

    number_of_scenarios = scenarios_info['number_of_scenarios']
    tree_size = scenarios_info['tree_size']

    # Load scenarios (yet, file with scenarios is empty; actual scenario tree is pickled)
    path_scenarios = path_input / f'scenarios_{number_of_scenarios}x{tree_size}'
    path_scenario_tree = path_scenarios.with_suffix('.pkl')
    with path_scenario_tree.open('rb') as f:
        log_debug(f'Unpickling scenario tree from <{path_scenario_tree!s}>...')
        scenario_tree = pickle.load(f)

    if method == 'basic':
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      path_output=path_output)
        task = PartialAutomatonTask(**config)
        efsm = task.run(T)

    elif method == 'basic-min':
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, T_init=T,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      path_output=path_output)
        task = MinimalPartialAutomatonTask(**config)
        efsm = task.run()

    elif method == 'extended':
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, P=P,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      is_forbid_or=is_forbid_or,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      path_output=path_output)
        task = CompleteAutomatonTask(**config)
        efsm = task.run()

    elif method == 'extended-min':
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, P=P, N_init=N,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      is_forbid_or=is_forbid_or,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      path_output=path_output)
        task = MinimalCompleteAutomatonTask(**config)
        efsm = task.run()

    elif method == 'extended-min-ub':
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, w=w,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      is_forbid_or=is_forbid_or,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      path_output=path_output)
        task = MinimalCompleteUBAutomatonTask(**config)
        efsm = task.run()

    else:
        raise click.BadParameter(f'"{method}" is not yet supported', param_hint='method')

    log_br()
    log_success(f'Done evaluating in {time.time() - time_start_evaluate:.2f} s')


if __name__ == '__main__':
    cli()
