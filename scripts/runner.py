import os
import pathlib
import time

import click

from fbsat.printers import *
from fbsat.utils import closed_range


def parse_minmax(ctx, param, value):
    try:
        min_, max_ = map(int, value.split('-', 2))
        return (min_, max_)
    except ValueError:
        raise click.BadParameter('needs to be in format <min-max>')


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
))
@click.option('-i', '--input', 'simulation_template', metavar='<path{i}>',
              type=click.Path(writable=True, file_okay=False), required=True,
              default='experiments/random/simulation{i}', show_default=True,
              help='Simulation path template')
@click.option('-s', '--simulations', 'simulations_range', metavar='<min-max>', callback=parse_minmax,
              default='1-9', show_default=True,
              help='Range of replicated simulations')
@click.option('--replica-glob', metavar='<path{i}>',
              default='replica*', show_default=True,
              help='Replicas glob')
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
def cli(simulation_template, simulations_range, replica_glob, method, C, K, P, T, N, w, use_bfs, is_distinct, is_forbid_or, sat_solver, is_incremental, is_filesolver, exist):
    time_start_evaluate = time.time()

    args = ''
    if method is not None:
        args += f' --method {method}'
    if C is not None:
        args += f' -C {C}'
    if K is not None:
        args += f' -K {K}'
    if P is not None:
        args += f' -P {P}'
    if T is not None:
        args += f' -T {T}'
    if N is not None:
        args += f' -N {N}'
    if w is not None:
        args += f' -w {w}'
    args += f' --{"" if use_bfs else "no-"}bfs'
    args += f' --{"" if is_distinct else "no-"}distinct'
    args += f' --{"" if is_forbid_or else "no-"}forbid-or'
    args += f' --sat-solver "{sat_solver}"'
    if is_incremental:
        args += ' --incremental'
    if is_filesolver:
        args += ' --filesolver'
    args += f' --exist-{exist}'

    for simulation_index in closed_range(*simulations_range):
        path_simulation = pathlib.Path(simulation_template.format(i=simulation_index))
        assert path_simulation.exists()

        for path_replica in path_simulation.glob(replica_glob):
            path_scenarios = path_replica / 'scenarios'
            assert path_scenarios.exists()
            if method == 'extended-min-ub' and w is not None:
                path_out = path_replica / 'out' / (method + f'-w{w}')
            else:
                path_out = path_replica / 'out' / method

            cmd = f'python evaluator.py -i {path_scenarios!s} -o {path_out!s} {args}'
            log_debug(cmd, symbol='$')
            os.system(cmd)

    log_br()
    log_success(f'Done evaluating in {time.time() - time_start_evaluate:.2f} s')


if __name__ == '__main__':
    cli()
