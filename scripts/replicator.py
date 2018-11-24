import os
import pathlib
import time

import click

from fbsat.printers import *
from fbsat.utils import closed_range

from pathutils import ensure_dir


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
# @click.option('-i', '--input', 'indir', metavar='<path>',
#               type=click.Path(exists=True, file_okay=False), required=True,
#               help='')
@click.option('-o', '--output', 'simulation_template', metavar='<path{i}>',
              type=click.Path(writable=True, file_okay=False), required=True,
              default='experiments/random/simulation{i}', show_default=True,
              help='Simulation path template (i/C/K/E/O/X/Z/P/ns/sl)')
@click.option('-s', '--simulations', 'simulations_range', metavar='<min-max>', callback=parse_minmax,
              default='1-9', show_default=True,
              help='Range of replicated simulations')
@click.option('-r', '--n-replicas', 'number_of_replicas', metavar='<min-max>',
              default=3, show_default=True,
              help='Number of replicas in each simulation')
@click.option('--replica-template', metavar='<path{i}>',
              default='replica{i}', show_default=True,
              help='Replica folder template')
@click.option('-C', 'C', type=int, metavar='<int>', required=True,
              help='Number of states')
@click.option('-K', 'K', type=int, metavar='<int>',
              help='Maximum number of outgoing transitions from each state')
@click.option('-E', 'E', type=int, metavar='<int>',
              default=1, show_default=True,
              help='Number of input events')
@click.option('-O', 'O', type=int, metavar='<int>',
              default=1, show_default=True,
              help='Number of output events')
@click.option('-X', 'X', type=int, metavar='<int>',
              default=2, show_default=True,
              help='Number of input variables')
@click.option('-Z', 'Z', type=int, metavar='<int>',
              default=2, show_default=True,
              help='Number of output variables')
@click.option('-P', 'P', type=int, metavar='<int>',
              default=5, show_default=True,
              help='Maximum guard size')
@click.option('-n', '--n-scenarios', 'number_of_scenarios', metavar='<int>',
              default=20, show_default=True,
              help='Number of scenarios')
@click.option('-l', '--scenario-len', 'scenario_length', metavar='<int>',
              default=50, show_default=True,
              help='Scenario length')
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
def cli(simulation_template, simulations_range, number_of_replicas, replica_template, C, K, E, O, X, Z, P, number_of_scenarios, scenario_length, exist):
    time_start_replicate = time.time()

    for simulation_index in closed_range(*simulations_range):
        path_simulation = pathlib.Path(simulation_template.format(i=simulation_index, C=C, K=K, E=E, O=O, X=X, Z=Z, P=P, ns=number_of_scenarios, sl=scenario_length))
        ensure_dir(path_simulation, exist)

        path_efsm = path_simulation / 'efsm'

        cmd = f'python generator.py -o {path_efsm!s} -C {C} -E {E} -O {O} -X {X} -Z {Z}'
        log_info(cmd, symbol='$')
        os.system(cmd)

        for replica_index in closed_range(1, number_of_replicas):
            path_replica = path_simulation / replica_template.format(i=replica_index)
            path_replica.mkdir(parents=True)
            path_scenarios = path_replica / 'scenarios'

            cmd = f'python simulator.py -i {path_efsm!s} -o {path_scenarios!s}'
            log_br()
            log_info(cmd, symbol='$')
            os.system(cmd)

    log_br()
    time_total_replicate = time.time() - time_start_replicate
    log_success(f'Done replicating in {time_total_replicate:.2f} s')


if __name__ == '__main__':
    cli()
