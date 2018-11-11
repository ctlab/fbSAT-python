import json
import pathlib
import shutil
import string
import time

import click

from fbsat.efsm import EFSM
from fbsat.printers import *
from fbsat.utils import closed_range


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
))
@click.option('-o', '--output', 'outdir', metavar='<path>',
              type=click.Path(writable=True, file_okay=False), required=True,
              help='Output folder for generated EFSM (e.g. simulation/efsm')
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
@click.option('-X', 'X', type=int, metavar='<int>', required=True,
              default=2, show_default=True,
              help='Number of input variables')
@click.option('-Z', 'Z', type=int, metavar='<int>', required=True,
              default=2, show_default=True,
              help='Number of output variables')
@click.option('-P', 'P', type=int, metavar='<int>',
              default=5, show_default=True,
              help='Maximum guard size')
@click.option('--input-events', metavar='<events...>',
              help='Comma-separated list of input events')
@click.option('--output-events', metavar='<events...>',
              help='Comma-separated list of output events')
@click.option('--input-names', metavar='<vars...>',
              help='Comma-separated list of input variables')
@click.option('--output-names', metavar='<vars...>',
              help='Comma-separated list of output variables')
@click.option('--force-write', 'is_force_write', is_flag=True,
              help='Write in existing output folder')
@click.option('--force-remove', 'is_force_remove', is_flag=True,
              help='Remove existing output folder')
def cli(outdir, C, K, E, O, X, Z, P, input_events, output_events, input_names, output_names,
        is_force_write, is_force_remove):
    time_start_generate = time.time()

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
            log_error('Output folder already exists, use --force-write or --force-remove')
            return

    assert C >= 1

    if K is None:
        log_debug(f'Using K=C={C}')
        K = C
    else:
        assert K >= 1
        assert K <= C

    if input_events:
        input_events = input_events.split(',')
    elif E == 1:
        input_events = ['REQ']
    else:
        input_events = list(string.ascii_uppercase[:E])

    if output_events:
        output_events = output_events.split(',')
    elif O == 1:
        output_events = ['CNF']
    else:
        output_events = list(string.ascii_uppercase[-O:])

    if input_names:
        input_names = input_names.split(',')
    else:
        input_names = [f'x{x}' for x in closed_range(1, X)]

    if output_names:
        output_names = output_names.split(',')
    else:
        output_names = [f'z{z}' for z in closed_range(1, Z)]

    # Generate random efsm
    efsm = EFSM.new_random(C, K, P, input_events, output_events, input_names, output_names)
    log_success('Randomly generated EFSM:')
    efsm.pprint()

    # Dump efsm
    path_efsm = path_output / f'efsm_random_C{efsm.C}_K{efsm.K}_P{efsm.P}_T{efsm.T}_N{efsm.N}'
    efsm.dump(str(path_efsm))

    # Save efsm info
    path_efsm_info = path_output / 'info_efsm.json'
    with path_efsm_info.open('w', encoding='utf8') as f:
        log_info(f'Writing EFSM info into <{path_efsm_info!s}>...')
        efsm_info = dict(C=efsm.C, K=efsm.K, P=efsm.P, T=efsm.T, N=efsm.N,
                         input_events=input_events, output_events=output_events,
                         input_names=input_names, output_names=output_names)
        json.dump(efsm_info, f, ensure_ascii=False, indent=4)

    log_br()
    log_success(f'Done generating a random automaton into '
                f'<{path_output!s}> in {time.time() - time_start_generate:.2f} s')


if __name__ == '__main__':
    cli()
