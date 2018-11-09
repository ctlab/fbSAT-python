import os
import json
import string
import time
import pickle
import pathlib

import click

from fbsat.printers import *
from fbsat.scenario import OutputAction, Scenario, ScenarioTree
from fbsat.tasks import *
from fbsat.utils import closed_range


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
))
@click.option('-i', '--input', '--efsm', 'indir', metavar='<path>',
              type=click.Path(exists=True, file_okay=False), required=True,
              # default='simulation/efsm', show_default=True,
              help='Input folder with EFSM (e.g., simulation/efsm')
@click.option('-o', '--output', '--outdir', 'outdir', metavar='<path>',
              type=click.Path(writable=True, file_okay=False), required=True,
              # default='simulation/replica0/scenarios', show_default=True,
              help='Output folder for simulated scenarios (e.g., simulation/replica0/scenarios)')
@click.option('--n-scenarios', 'number_of_scenarios', metavar='<int>',
              default=20, show_default=True,
              help='Number of scenarios')
@click.option('--scenario-len', 'scenario_length', metavar='<int>',
              default=50, show_default=True,
              help='Scenario length')
@click.option('--force-write', 'is_force_write', is_flag=True,
              help='Write in existing output folder')
@click.option('--force-remove', 'is_force_remove', is_flag=True,
              help='Remove existing output folder')
def cli(indir, outdir, number_of_scenarios, scenario_length, is_force_write, is_force_remove):
    time_start_simulate = time.time()

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

    # Load automaton info
    path_efsm_info = path_input.joinpath('info_efsm.json')
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
    path_efsm = path_input.joinpath(f'efsm_random_C{C}_K{K}_P{P}_T{T}_N{N}.pkl')
    with path_efsm.open('rb') as f:
        efsm = pickle.load(f)

    # Simulate scenarios
    scenarios = []
    log_info(f'Generating {number_of_scenarios} scenarios, each of length {scenario_length}...')
    for _ in range(number_of_scenarios):
        scenarios.append(efsm.random_walk(scenario_length, input_events=input_events,
                                          X=len(input_names), Z=len(output_names)))
    _tree_size_uncompressed = sum(map(len, scenarios))

    # Preprocess scenarios
    scenarios = Scenario.preprocess_scenarios(scenarios)

    # Build scenario tree
    scenario_tree = ScenarioTree(scenarios)
    scenario_tree.scenarios_filename = 'simulated'
    scenario_tree.input_names = input_names
    scenario_tree.output_names = output_names
    log_debug(f'ScenarioTree size: {scenario_tree.size()}')
    # scenario_tree.pprint(n=10)

    # Pickle scenario tree
    path_scenarios = path_output.joinpath('scenarios')
    # log_debug(f'Saving scenario tree into <{path_scenarios!s}>...')
    path_scenarios.touch()
    path_scenarios_pkl = path_scenarios.with_suffix('.pkl')
    with path_scenarios_pkl.open('wb') as f:
        log_debug(f'Pickling scenario tree into <{path_scenarios_pkl!s}>...')
        pickle.dump(scenario_tree, f)

    # Save scenarios info
    path_scenarios_info = path_output.joinpath('info_scenarios.json')
    with path_scenarios_info.open('w', encoding='utf8') as f:
        log_info(f'Writing scenarios info into <{path_scenarios_info!s}>...')
        scenarios_info = dict(number_of_scenarios=number_of_scenarios,
                              tree_size_compressed=scenario_tree.size(),
                              tree_size_uncompressed=_tree_size_uncompressed,
                              input_events=input_events, output_events=output_events,
                              input_names=input_names, output_names=output_names)
        json.dump(scenarios_info, f, ensure_ascii=False, indent=4)

    log_br()
    log_success(f'Done simulating {number_of_scenarios} scenario(s) into <{path_output!s}> in {time.time() - time_start_simulate:.2f} s')

    return

    efsm.scenario_tree = scenario_tree
    efsm.dump(f'{outdir}/efsm_random_C{efsm.number_of_states}_T{efsm.number_of_transitions}_N{efsm.number_of_nodes}')
    efsm.verify()

    config = dict(scenario_tree=scenario_tree,
                  # C=C, K=K, P=P, N=N,
                  w=w,
                  # use_bfs=True,
                  # is_distinct=False,
                  # is_forbid_or=False,
                  solver_cmd='incremental-cryptominisat',
                  is_incremental=True,
                  # is_filesolver=False,
                  outdir=outdir)
    # task = MinimalPartialAutomatonTask(**config)
    task = MinimalCompleteUBAutomatonTask(**config)
    time_start_task = time.time()
    minimal_complete_automaton = task.run()
    time_total_task = time.time() - time_start_task

    with open(f'{outdir}/stats_total.csv', 'w') as f:
        f.write(f'scenarios,tree_size,C,E,O,X,Z,P,C_real,P_real,T,N,time\n')
        f.write(f'{len(scenarios)},'
                f'{scenario_tree.size()},'
                f'{C},'
                f'{E},'
                f'{O},'
                f'{X},'
                f'{Z},'
                f'{P},'
                f'{minimal_complete_automaton.number_of_states},'
                f'{minimal_complete_automaton.guard_condition_maxsize},'
                f'{minimal_complete_automaton.number_of_transitions},'
                f'{minimal_complete_automaton.number_of_nodes},'
                f'{time_total_task:.3f}'
                '\n')

    # with open(f'{outdir}/stats_intermediate.csv', 'w') as f:
    #     f.write(f'N\n')
    #     for automaton in task.intermediate:
    #         f.write(f'{automaton.number_of_nodes}\n')

    log_br()
    log_success(f'Done bootstraping into {outdir} in {time.time() - time_start_boot:.2f} s')


if __name__ == '__main__':
    cli()
