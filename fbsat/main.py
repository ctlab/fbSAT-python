import time

import click

from .basic import Instance as InstanceBasic
# from .minimize import Instance as InstanceMinimize
from .combined import Instance as InstanceCombined
from .scenario import *
from .utils import *
from .printers import *
from version import __version__

CONTEXT_SETTINGS = dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('strategy')
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
@click.option('--prefix', 'filename_prefix', metavar='<path>',
              type=click.Path(writable=True),
              default='cnf', show_default=True,
              help='Generated CNF filename prefix')
@click.option('-C', 'C', type=int, metavar='<int>',
              help='Number of colors (automata states)')
@click.option('-K', 'K', type=int, metavar='<int>',
              help='Maximum number of transitions from each state')
@click.option('-P', 'P', type=int, metavar='<int>',
              help='Maximum number of nodes in guard\'s boolean formula\'s parse tree')
@click.option('-N', 'N', type=int, metavar='<int>', default=0,
              help='Initial upper bound on total number of nodes in all guard-trees')
@click.option('--min', 'is_minimize', is_flag=True,
              help='Do minimize')
@click.option('--sat-solver', metavar='<sat-solver-cmd>',
              default='glucose -model -verb=0', show_default=True,
              # default='cryptominisat5 --verb=0', show_default=True,
              # default='cadical -q', show_default=True,
              help='SAT-solver')
@click.version_option(__version__)
def cli(strategy, filename_scenarios, filename_predicate_names, filename_output_variable_names, filename_prefix, C, K, P, N, is_minimize, sat_solver):
    if strategy == 'combined':
        if C is None:
            raise click.BadParameter('missing value', param_hint='C')
        if K is None:
            raise click.BadParameter('missing value', param_hint='K')
        if P is None:
            raise click.BadParameter('missing value', param_hint='P')

    log_info('Welcome!')
    log_br()
    time_start = time.time()

    log_info('Building scenario tree...')
    scenario_tree = ScenarioTree.from_files(filename_scenarios, filename_predicate_names, filename_output_variable_names)
    scenario_tree.pprint(n=30)
    log_success(f'Successfully built scenario tree of size {scenario_tree.size()}')
    log_br()

    if strategy == 'basic':
        log_info('Basic strategy')
        InstanceBasic(scenario_tree=scenario_tree,
                      filename_prefix=filename_prefix,
                      sat_solver=sat_solver).run()
    elif strategy == 'combined':
        log_info('Combined strategy')
        InstanceCombined(scenario_tree=scenario_tree,
                         C=C, K=K, P=P, N=N,
                         filename_prefix=filename_prefix,
                         is_minimize=is_minimize,
                         sat_solver=sat_solver).run()

    log_success(f'All done in {time.time() - time_start:.2f} s')
