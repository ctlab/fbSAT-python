import shutil

import click

from fbsat.printers import *


def ensure_dir(path, exist):
    '''Ensure folder exists and maybe clear/recreate it or throw an error'''
    if not path.exists():
        log_debug(f'Creating folder <{path}>...')
        path.mkdir(parents=True)
    else:
        assert path.is_dir()
        if exist == 'ok':
            pass
        elif exist == 'remove-files':
            for child in path.rglob('*'):
                if child.is_file():
                    log_warn(f'Removing file <{child!s}>')
                    child.unlink()
        elif exist == 'remove-all':
            for child in path.iterdir():
                if child.is_dir():
                    log_warn(f'Removing directory <{child!s}>')
                    shutil.rmtree(str(child))
                elif child.is_file():
                    log_warn(f'Removing file <{child!s}>')
                    child.unlink()
                else:
                    log_warn(f'Neither a directory nor a file: {child}')
        elif exist == 'recreate':
            log_warn(f'Recreating folder <{path}>...')
            shutil.rmtree(str(path))
            path.mkdir()
        else:
            raise click.BadParameter('folder already exists, consider --exist-ok and similar options')
