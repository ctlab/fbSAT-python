import os
import time
import pickle

import click

from .basic import Instance as InstanceBasic
from .minimize import Instance as InstanceMinimize
from .combined import Instance as InstanceCombined
from .scenario import *
from .efsm import *
from .utils import *
from .printers import *
from version import __version__

CONTEXT_SETTINGS = dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('strategy', type=click.Choice(['basic', 'minimize', 'combined']))
@click.option('-i', '--scenarios', 'filename_scenarios', metavar='<path/->', required=True,
              type=click.Path(exists=True, allow_dash=True),
              help='File with scenarios')
@click.option('--predicate-names', 'filename_predicate_names', metavar='<path>',
              type=click.Path(exists=True),
              default='predicate-names', show_default=True,
              help='File with precidate names')
@click.option('--output-variable-names', 'filename_output_variable_names', metavar='<path>',
              type=click.Path(exists=True),
              default='output-variables', show_default=True,
              help='File with output variables names')
# TODO: replace filename prefix with output directory and maybe filename template
@click.option('--prefix', 'filename_prefix', metavar='<path>',
              type=click.Path(writable=True),
              default='cnf/cnf', show_default=True,
              help='Generated CNF filename prefix')
@click.option('-C', 'C', type=int, metavar='<int>',
              help='Number of colors (automata states)')
@click.option('-K', 'K', type=int, metavar='<int>',
              help='Maximum number of transitions from each state')
@click.option('-P', 'P', type=int, metavar='<int>',
              help='Maximum number of nodes in guard\'s boolean formula\'s parse tree')
@click.option('-N', 'N', type=int, metavar='<int>', default=0,
              help='Initial upper bound on total number of nodes in all guard-trees')
@click.option('-Cmax', 'Cmax', type=int, metavar='<int>',
              help='[Basic] C_max')
@click.option('--min', 'is_minimize', is_flag=True,
              help='Do minimize')
@click.option('--incremental', 'is_incremental', is_flag=True,
              help='Use incremental solver')
@click.option('--reuse', 'is_reuse', is_flag=True,
              help='Reuse generated base reduction and objective function')
@click.option('--sat-solver', metavar='<cmd>',
              default='glucose -model -verb=0', show_default=True,
              # default='cryptominisat5 --verb=0', show_default=True,
              # default='cadical -q', show_default=True,
              help='SAT solver')
@click.option('--sat-isolver', metavar='<cmd>',
              default='incremental-lingeling', show_default=True,
              help='Incremental SAT solver')
@click.option('--mzn-solver', metavar='<cmd>',
              default='mzn-fzn --solver fz', show_default=True,
              help='Minizinc solver')
@click.option('--write-strategy', type=click.Choice(['direct', 'tempfile', 'StringIO']),
              default='StringIO', show_default=True,
              help='Which file-write strategy to use')
@click.version_option(__version__)
def cli(strategy, filename_scenarios, filename_predicate_names, filename_output_variable_names, filename_prefix, C, K, P, N, Cmax, is_minimize, is_incremental, is_reuse, sat_solver, sat_isolver, mzn_solver, write_strategy):
    if strategy == 'basic':
        if is_incremental and sat_isolver is None:
            raise click.BadParameter('missing incremental solver', param_hint='sat_isolver')
        if not is_minimize and C is None:
            raise click.BadParameter('missing value', param_hint='C')
    elif strategy == 'combined':
        if C is None:
            raise click.BadParameter('missing value', param_hint='C')
        if K is None:
            raise click.BadParameter('missing value', param_hint='K')

    log_info('Welcome!')
    log_br()
    time_start = time.time()

    log_info('Building scenario tree...')
    time_start_tree = time.time()
    scenario_tree = ScenarioTree.from_files(filename_scenarios, filename_predicate_names, filename_output_variable_names)
    # scenario_tree.pprint(n=30)
    log_success(f'Successfully built scenario tree of size {scenario_tree.size()} in {time.time() - time_start_tree:.2f} s')
    log_br()

    if not os.path.exists(os.path.dirname(filename_prefix)):
        log_warn('Ensuring folder for CNFs exists')
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

    filename_prefix += '_' + os.path.splitext(os.path.basename(filename_scenarios))[0]

    if strategy == 'basic':
        log_info('Basic strategy')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, C_max=Cmax,
                      is_minimize=is_minimize,
                      is_incremental=is_incremental,
                      sat_solver=sat_solver,
                      sat_isolver=sat_isolver,
                      filename_prefix=filename_prefix,
                      write_strategy=write_strategy,
                      is_reuse=is_reuse)
        InstanceBasic(**config).run()
    elif strategy == 'minimize':
        log_info('Minimize strategy')
        with open(filename_prefix + '_automaton', 'rb') as f:
            efsm = pickle.load(f)
        config = dict(scenario_tree=scenario_tree,
                      efsm=efsm,
                      mzn_solver=mzn_solver,
                      filename_prefix=filename_prefix,
                      write_strategy=write_strategy,
                      is_reuse=is_reuse)
        InstanceMinimize(**config).run()
    elif strategy == 'combined':
        log_info('Combined strategy')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, P=P, N=N,
                      is_minimize=is_minimize,
                      is_incremental=is_incremental,
                      sat_solver=sat_solver,
                      sat_isolver=sat_isolver,
                      filename_prefix=filename_prefix,
                      write_strategy=write_strategy,
                      is_reuse=is_reuse)
        InstanceCombined(**config).run()

    log_success(f'All done in {time.time() - time_start:.2f} s')
