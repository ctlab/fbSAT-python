import json
import pathlib
import pickle
import time

import click

from fbsat.printers import *
from fbsat.scenario import ScenarioTree

from pathutils import ensure_dir


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
))
@click.option('-i', '--intput', 'indir', metavar='<path>',
              type=click.Path(exists=True, file_okay=False), required=True,
              help='Input folder with EFSM (e.g., simulation/efsm)')
@click.option('-o', '--output', 'outdir', metavar='<path>',
              type=click.Path(writable=True, file_okay=False), required=True,
              help='Output folder for simulated scenarios (e.g., simulation/replica0/scenarios)')
@click.option('--n-scenarios', 'number_of_scenarios', metavar='<int>',
              default=20, show_default=True,
              help='Number of scenarios')
@click.option('--scenario-len', 'scenario_length', metavar='<int>',
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
def cli(indir, outdir, number_of_scenarios, scenario_length, exist):
    time_start_simulate = time.time()

    path_input = pathlib.Path(indir)
    assert path_input.is_dir()

    path_output = pathlib.Path(outdir)
    ensure_dir(path_output, exist)

    # Load automaton info
    path_efsm_info = path_input / 'info_efsm.json'
    if not path_efsm_info.exists():
        raise click.BadParameter(f'folder does not contain {path_efsm_info.name}', param_hint='input')
    with path_efsm_info.open() as f:
        log_debug(f'Loading EFSM info from <{path_efsm_info!s}>...')
        efsm_info = json.load(f)
        log_debug(f'efsm_info: {efsm_info}')

    C = efsm_info['C']
    K = efsm_info['K']
    P = efsm_info['P']
    T = efsm_info['T']
    N = efsm_info['N']
    input_events = efsm_info['input_events']
    output_events = efsm_info['output_events']
    input_names = efsm_info['input_names']
    output_names = efsm_info['output_names']

    # Load pickled automaton
    path_efsm = path_input / f'efsm_random_C{C}_K{K}_P{P}_T{T}_N{N}.pkl'
    with path_efsm.open('rb') as f:
        log_debug(f'Unpickling EFSM from <{path_efsm!s}>...')
        efsm = pickle.load(f)

    # Simulate scenarios
    scenarios = []
    log_debug(f'Generating {number_of_scenarios} scenarios, each of length {scenario_length}...')
    for _ in range(number_of_scenarios):
        scenarios.append(efsm.random_walk(scenario_length, input_events=input_events,
                                          X=len(input_names), Z=len(output_names)))
    _elements_uncompressed = sum(map(len, scenarios))

    # Build scenario tree
    scenario_tree = ScenarioTree(scenarios, input_names, output_names)
    log_debug(f'Scenario tree size: {scenario_tree.size()}')

    # Verify simulated scenarios
    efsm.verify(scenario_tree)

    # Save scenarios (so far, actual saving has not been implemented; just creating an empty file)
    path_scenarios = path_output / f'scenarios_{number_of_scenarios}x{scenario_tree.size()}'
    log_debug(f'Saving scenarios into <{path_scenarios!s}>...')
    path_scenarios.touch()

    # Pickle scenario tree
    path_scenario_tree = path_scenarios.with_suffix('.pkl')
    with path_scenario_tree.open('wb') as f:
        log_debug(f'Pickling scenario tree into <{path_scenario_tree!s}>...')
        pickle.dump(scenario_tree, f)

    # Save scenarios info
    path_scenarios_info = path_output / 'info_scenarios.json'
    with path_scenarios_info.open('w', encoding='utf8') as f:
        log_debug(f'Writing scenarios info into <{path_scenarios_info!s}>...')
        scenarios_info = dict(number_of_scenarios=number_of_scenarios,
                              tree_size=scenario_tree.size(),
                              elements_uncompressed=_elements_uncompressed,
                              input_events=input_events, output_events=output_events,
                              input_names=input_names, output_names=output_names)
        json.dump(scenarios_info, f, ensure_ascii=False, indent=4)

    log_br()
    log_success(f'Done simulating {number_of_scenarios} scenario(s) into <{path_output!s}> in {time.time() - time_start_simulate:.2f} s')


if __name__ == '__main__':
    cli()
