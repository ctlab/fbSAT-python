__all__ = ('NotBool', 'closed_range', 'open_maybe_gzip', 'read_names')

import os

import click

from .printers import log_debug

GlobalState = {}


class NotBoolType:
    def __bool__(self):
        raise RuntimeError('this is not bool')

    def __repr__(self):
        return 'NotBool'


NotBool = NotBoolType()


def closed_range(start, stop=None, step=1):
    if stop is None:
        return range(start + 1)
    else:
        return range(start, stop + 1, step)


def open_maybe_gzip(filename):
    if os.path.splitext(filename)[1] == '.gz':
        import gzip
        return gzip.open(filename, 'rt')
    else:
        return click.open_file(filename)


def read_names(filename):
    log_debug(f'Reading names from <{click.format_filename(filename)}>...')
    with open_maybe_gzip(filename) as f:
        names = f.read().strip().split('\n')
    log_debug(f'Done reading names: {", ".join(names)}')
    return names
