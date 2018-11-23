import json
import gzip
import pathlib
from functools import wraps

import click

from .printers import log_debug, log_warn

__all__ = ['NotBool', 'closed_range', 'open_maybe_gzip', 'parse_names',
           'algorithm2st', 'b2s', 's2b',
           'parse_raw_assignment_algo', 'parse_raw_assignment_bool', 'parse_raw_assignment_int',
           'auto_finalize', 'json_dump']


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
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    else:
        return click.open_file(filename)


def parse_names(ctx, param, value):
    if value:
        path = pathlib.Path(value)
        if path.is_file():
            log_debug(f'Reading names from <{path}>...')

            with open_maybe_gzip(str(path)) as f:
                names = f.read().strip().split()

            log_debug(f'Done reading names: {", ".join(names)}')
            return names
        else:
            return value.split(',')


def algorithm2st(output_variable_names, algorithm_0, algorithm_1):
    assert len(output_variable_names) == len(algorithm_0) == len(algorithm_1)
    st = ''
    for name, a0, a1 in zip(output_variable_names, algorithm_0, algorithm_1):
        if a0 == a1:
            st += f'{name}:={ {"0":"FALSE", "1":"TRUE"}[a0] };'
        elif a0 == '0':
            st += f'{name}:=~{name};'
    return st


def b2s(data, *, s_True='1', s_False='0'):
    '''Converts 0-based bool array to string'''
    return ''.join(s_True if x else s_False for x in data)


def s2b(s, zero_based=False):
    '''Converts string to bool array'''
    ans = [{'1': True, '0': False}[c] for c in s]
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


def auto_finalize(func):
    @wraps(func)
    def wrapped(self, *args, finalize=True, **kwargs):
        result = func(self, *args, **kwargs)
        if finalize:
            self.finalize()
        return result
    return wrapped


def json_dump(obj, path):
    with path.open('w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
