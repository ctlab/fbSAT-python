import os
import pickle
import time

import click

from .basic import Instance as InstanceBasic
from .basic2 import Instance as InstanceBasic2
from .combined import Instance as InstanceCombined
from .combined2 import Instance as InstanceCombined2
from .combined3 import Instance as InstanceCombined3
from .efsm import *
from .minimize import Instance as InstanceMinimize
from .printers import *
from .scenario import *
from .solver import *
from .tasks import *
from .utils import *
from .version import version as __version__

CONTEXT_SETTINGS = dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('strategy', type=click.Choice(['old-basic', 'old-basic2', 'old-minimize',
                                               'old-combined', 'old-combined2', 'old-combined3',
                                               'full', 'full-min',
                                               'partial', 'partial-min',
                                               'complete', 'complete-min',
                                               'minimize']))
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
@click.option('-o', '--out', '--dir', '--outdir', 'outdir', metavar='<path>',
              type=click.Path(file_okay=False, writable=True),
              default='out', show_default=True,
              help='Output directory')
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
@click.option('-Cmax', 'Cmax', type=int, metavar='<int>',
              help='[old-basic] C_max')
@click.option('--min', 'is_minimize', is_flag=True,
              help='[old] Do minimize')
@click.option('--incremental', 'is_incremental', is_flag=True,
              help='Use IncrementalSolver backend')
@click.option('--filesolver', 'is_filesolver', is_flag=True,
              help='Use FileSolver backend')
@click.option('--reuse', 'is_reuse', is_flag=True,
              help='Reuse generated base reduction and objective function')
@click.option('--bfs/--no-bfs', 'use_bfs', is_flag=True,
              default=True, show_default=True,
              help='Use BFS symmetry-breaking constraints')
@click.option('--distinct/--no-distinct', 'is_distinct', is_flag=True,
              default=False, show_default=True,
              help='[complete] Distinct transitions')
@click.option('--forbid-or/--no-forbid-or', 'is_forbid_or', is_flag=True,
              default=False, show_default=True,
              help='[complete] Distinct transitions')
@click.option('--sat-solver', metavar='<cmd>',
              default='glucose -model -verb=0', show_default=True,
              # default='cryptominisat5 --verb=0', show_default=True,
              # default='cadical -q', show_default=True,
              help='SAT solver')
@click.option('--sat-isolver', metavar='<cmd>',
              default='incremental-lingeling', show_default=True,
              help='[old] Incremental SAT solver')
@click.option('--mzn-solver', metavar='<cmd>',
              default='minizinc --fzn-cmd fzn-gecode', show_default=True,
              # default='mzn-fzn --solver fz', show_default=True,
              help='[old] Minizinc solver')
@click.option('--write-strategy', type=click.Choice(['direct', 'tempfile', 'StringIO', 'pysat']),
              default='StringIO', show_default=True,
              help='[old] Which file-write strategy to use')
@click.option('--automaton', help='File with pickled automaton')
@click.version_option(__version__)
def cli(strategy, filename_scenarios, filename_predicate_names, filename_output_variable_names,
        filename_prefix, outdir, C, K, P, N, T, Cmax, is_minimize, is_incremental, is_filesolver,
        is_reuse, use_bfs, is_distinct, is_forbid_or, sat_solver, sat_isolver, mzn_solver,
        write_strategy, automaton):
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
    scenario_tree = ScenarioTree.from_files(filename_scenarios, filename_predicate_names, filename_output_variable_names)
    # scenario_tree.pprint(n=30)
    log_success(f'Successfully built scenario tree of size {scenario_tree.size()} in {time.time() - time_start_tree:.2f} s')

    if not os.path.exists(os.path.dirname(filename_prefix)):
        log_br()
        log_warn('Ensuring folder for CNFs exists')
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

    if not os.path.exists(outdir):
        log_br()
        log_warn('Ensuring output directory exists')
        os.makedirs(outdir, exist_ok=True)

    filename_prefix += '_' + os.path.splitext(os.path.basename(filename_scenarios))[0]

    # =================
    with open(f'{filename_prefix}_scenario_tree', 'wb') as f:
        pickle.dump(scenario_tree, f, pickle.HIGHEST_PROTOCOL)
    # =================

    log_br()
    if strategy == 'old-basic':
        log_info('Old Basic strategy')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, C_max=Cmax,
                      is_minimize=is_minimize,
                      is_incremental=is_incremental,
                      sat_solver=sat_solver,
                      sat_isolver=sat_isolver,
                      filename_prefix=filename_prefix,
                      write_strategy=write_strategy,
                      is_reuse=is_reuse,
                      use_bfs=use_bfs)
        InstanceBasic(**config).run()
    elif strategy == 'old-basic2':
        log_info('Old Basic2 strategy')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, C_max=Cmax,
                      is_minimize=is_minimize,
                      is_incremental=is_incremental,
                      sat_solver=sat_solver,
                      sat_isolver=sat_isolver,
                      filename_prefix=filename_prefix,
                      write_strategy=write_strategy,
                      is_reuse=is_reuse)
        InstanceBasic2(**config).run()
    elif strategy == 'old-minimize':
        log_info('Old Minimize strategy')
        if automaton:
            filename_automaton = automaton
        else:
            filename_automaton = filename_prefix + '_automaton_basic'
        with open(filename_automaton, 'rb') as f:
            efsm = pickle.load(f)
        config = dict(scenario_tree=scenario_tree,
                      efsm=efsm,
                      mzn_solver=mzn_solver,
                      filename_prefix=filename_prefix,
                      write_strategy=write_strategy,
                      is_reuse=is_reuse)
        InstanceMinimize(**config).run()
    elif strategy == 'old-combined':
        log_info('Old Combined strategy')
        assert C is not None
        assert K is not None
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
    elif strategy == 'old-combined2':
        log_info('Old Combined2 strategy')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, P=P, N=N,
                      is_minimize=is_minimize,
                      is_incremental=is_incremental,
                      sat_solver=sat_solver,
                      sat_isolver=sat_isolver,
                      filename_prefix=filename_prefix,
                      write_strategy=write_strategy,
                      is_reuse=is_reuse)
        InstanceCombined2(**config).run()
    elif strategy == 'old-combined3':
        log_info('Old Combined3 strategy')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, T=T,
                      is_minimize=is_minimize,
                      sat_solver=sat_solver,
                      filename_prefix=filename_prefix,
                      write_strategy=write_strategy,
                      is_reuse=is_reuse)
        InstanceCombined3(**config).run()

    # ===== NEW STRATEGIES =====

    elif strategy == 'full':
        log_info('Full strategy')
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
        full_automaton = task.run(T)  # noqa
        task.finalize()

    elif strategy == 'full-min':
        log_info('MinimalFull strategy')
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
        minimal_full_automaton = task.run()  # noqa

    elif strategy == 'partial':
        log_info('Partial strategy')
        if P is not None:
            log_warn(f'Ignoring specified P={P}')
        if N is not None:
            log_warn(f'Ignoring specified N={N}')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K,
                      use_bfs=use_bfs,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        task = PartialAutomatonTask(**config)
        partial_automaton = task.run(T)  # noqa
        task.finalize()

    elif strategy == 'partial-min':
        log_info('MinimalPartial strategy')
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
        task = MinimalPartialAutomatonTask(**config)
        minimal_partial_automaton = task.run()  # noqa

    elif strategy == 'complete':
        log_info('Complete strategy')
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
        task = CompleteAutomatonTask(**config)
        complete_automaton = task.run(N)  # noqa
        task.finalize()

    elif strategy == 'complete-min':
        log_info('MinimalComplete strategy')
        if T is not None:
            log_warn(f'Ignoring specified T={T}')
        config = dict(scenario_tree=scenario_tree,
                      C=C, K=K, P=P, N=N,
                      use_bfs=use_bfs,
                      is_distinct=is_distinct,
                      is_forbid_or=is_forbid_or,
                      solver_cmd=sat_solver,
                      is_incremental=is_incremental,
                      is_filesolver=is_filesolver,
                      outdir=outdir)
        task = MinimalCompleteAutomatonTask(**config)
        minimal_complete_automaton = task.run()  # noqa

    elif strategy == 'minimize':
        log_info('MinimizeAllGuards strategy')
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
                config['partial_automaton'] = pickle.load(f)
        task = MinimizeAllGuardsTask(**config)
        minimized_automaton = task.run()  # noqa

    log_br()
    log_success(f'All done in {time.time() - time_start:.2f} s')
