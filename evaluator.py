import json
import pathlib
import pickle
import shutil
import time

import click

from fbsat.printers import *
from fbsat.tasks import *


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
))
@click.option('-i', '--input', '--scenarios', 'indir', metavar='<path>',
              type=click.Path(exists=True, file_okay=False), required=True,
              help='Input folder with scenarios (e.g., simulation/replica0/scenarios')
@click.option('-o', '--output', '--outdir', 'outdir', metavar='<path>',
              type=click.Path(writable=True, file_okay=False), required=True,
              help='Output folder for evaluation results (e.g., similation/replica0/out/<method>)')
@click.option('--method', metavar='<method>', required=True,
              type=click.Choice(['full', 'full-min',
                                 'basic', 'basic-min',
                                 'extended', 'extended-min', 'extended-min-ub']),
              help='Method')
@click.option('-w', 'w', type=int, metavar='<int>',
              help='[complete-min-ub] Maximum width of local minima')
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
@click.option('--force-write', 'is_force_write', is_flag=True,
              help='Write in existing output folder')
@click.option('--force-remove', 'is_force_remove', is_flag=True,
              help='Remove existing output folder')
def cli(indir, outdir, method, w, use_bfs, is_distinct, is_forbid_or, sat_solver, is_incremental, is_filesolver, is_force_write, is_force_remove):
    time_start_evaluate = time.time()

    path_input = pathlib.Path(indir)
    assert path_input.is_dir()

    # Ensure output folder exists and maybe recreate it or throw an error
    path_output = pathlib.Path(outdir)
    if not path_output.exists():
        log_debug(f'Creating output folder <{path_output}>...')
        path_output.mkdir(parents=True)
    else:
        assert path_output.is_dir()
        if is_force_write:
            pass
        elif is_force_remove:
            log_warn(f'Recreating output folder <{path_output}>...')
            for child in path_output.iterdir():
                if child.is_dir():
                    shutil.rmtree(str(child))
                elif child.is_file():
                    child.unlink()
                else:
                    log_warn(f'Neither a directory nor a file: {child}')
        else:
            raise click.BadParameter('folder already exists, consider --force-write or --force-remove', param_hint='output')

    # Load scenarios info
    path_scenarios_info = path_input.joinpath('info_scenarios.json')
    if not path_scenarios_info.exists():
        raise click.BadParameter(f'folder does not contain {path_scenarios_info.name}', param_hint='input')
    with path_scenarios_info.open() as f:
        log_debug(f'Loading scenarios info from <{path_scenarios_info!s}>...')
        scenarios_info = json.load(f)
        log_debug(f'scenarios_info: {scenarios_info}')

    number_of_scenarios = scenarios_info['number_of_scenarios']
    tree_size = scenarios_info['tree_size']

    # Load scenarios (yet, file with scenarios is empty; actual scenario tree is pickled)
    path_scenarios = path_input.joinpath(f'scenarios_{number_of_scenarios}x{tree_size}')
    path_scenario_tree = path_scenarios.with_suffix('.pkl')
    with path_scenario_tree.open('rb') as f:
        log_debug(f'Unpickling scenario tree from <{path_scenario_tree!s}>...')
        scenario_tree = pickle.load(f)

    # EVALUATE:
    # create task
    # run task
    # save results and stats

    if method == 'extended-min-ub':
        config = dict(scenario_tree=scenario_tree,
                      w=w,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      is_forbid_or=is_forbid_or,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      outdir=str(path_output))

        task = MinimalCompleteUBAutomatonTask(**config)
        time_start_task = time.time()
        efsm = task.run()
        time_total_task = time.time() - time_start_task

    log_br()
    log_success(f'Done evaluating in {time.time() - time_start_evaluate:.2f} s')


if __name__ == '__main__':
    cli()
