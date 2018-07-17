import os

import click

from .printers import log_debug, log_warn

__all__ = ['NotBool', 'closed_range', 'open_maybe_gzip', 'read_names', 'b2s', 's2b']

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


def b2s(data):
    '''Converts 0-based bool array to string'''
    return ''.join('1' if x else '0' for x in data)


def s2b(s, zero_based=False):
    '''Converts string to bool array'''
    ans = [c != '0' for c in s]
    if zero_based:
        return ans
    else:
        return [NotBool] + ans


def parse_raw_assignment_int(raw_assignment, data):
    if isinstance(data[1], (list, tuple)):
        return [None] + [parse_raw_assignment_int(raw_assignment, x) for x in data[1:]]
    else:
        for i, x in enumerate(data):
            if x is not None and raw_assignment[x] > 0:
                return i
        log_warn('data[...] is unknown')


def parse_raw_assignment_bool(raw_assignment, data):
    if isinstance(data[1], (list, tuple)):
        return [None] + [parse_raw_assignment_bool(raw_assignment, x) for x in data[1:]]
    else:
        if data[0] is None:
            return [NotBool] + [raw_assignment[x] > 0 for x in data[1:]]
        else:
            return [raw_assignment[x] > 0 for x in data]


def parse_raw_assignment_algo(raw_assignment, data):
    return [None] + [b2s(raw_assignment[item] > 0 for item in subdata[1:])
                     for subdata in data[1:]]
