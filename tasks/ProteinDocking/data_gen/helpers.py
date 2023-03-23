"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


from argparse import ArgumentError
import os
import multiprocessing as mp
from functools import partial, partialmethod
from pathlib import Path, PosixPath
from tqdm import tqdm
from logging import Logger
from datetime import datetime, timezone
from subprocess import Popen, PIPE


def title_str(title):
    """Format a string title"""
    return "#" * 80 + "\n" + \
           "#" + ' ' * 5 + title + "\n" + \
           "#" + ' ' * 5 + datetime.now(timezone.utc).strftime('UTC Time: %Y-%m-%d %H:%M:%S') + "\n" + \
           "#" * 80


def args_str(args, indent=2, top_ruler=False, bottom_ruler=True):
    """Format argument str, one argument per line"""
    str_formatted = ''
    if top_ruler:
        str_formatted += '=' * 50 + '\n'
    str_formatted += 'Args:\n' + ' ' * indent \
        + ('\n' + ' ' * indent).join(f"{key}: {val}" for key, val in args.__dict__.items())
    if bottom_ruler:
        str_formatted += "\n" + '=' * 50
    return str_formatted


def log_info(msg, logger):
    """Log info message.
    logger can be 'print' (function), logging.Logger, path to logging file, or a list of them
    """
    if isinstance(logger, (list, tuple)):
        for l in logger:
            log_info(msg, l)
    elif logger is print:
        print(msg, flush=True)
    elif isinstance(logger, Logger):
        logger.info(msg)
    elif isinstance(logger, (str, PosixPath)):
        # append to a file
        assert Path(logger).parent.exists(), f"{Path(logger).parent} doesn't exist"
        with open(logger, 'a' if Path(logger).exists() else 'w') as f:
            f.write(msg + '\n')
    elif callable(logger):
        logger(msg)
    else:
        raise TypeError("'logger' must be print, a Logger, a file path, a callable, or a list of them")


def catch_error(*args, func=None, logger=None, return_val=None, **kwargs):
    """Capture error and return a placeholder result rather than interruption. Optionally log error message"""
    if func is None:
        raise ArgumentError("Please provide 'func' as a keyword argument")
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger is not None:
            log_info(
                msg=type(e).__name__ + ": " + str(e),
                logger=logger
            )
        return return_val


def optim_chuncksize(itr, j):
    """Compute an optimal chuncksize as min(len(itr) // (j * 20), 1)"""
    return max(len(itr) // j // 20, 1)


def mp_run(func, itr,
           j=None, chunksize=None, mute_tqdm=False,
           catch_err=False, err_logger=None, err_return_val="error",
           **kwargs):
    """Helper function for multiprocessing run. The each item in the iterable should contain only one argument"""

    if kwargs != {}:
        func = partial(func, **kwargs)
    if mute_tqdm:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    if j is None:
        j = min(len(itr), os.cpu_count() // 2)
    if chunksize is None:
        chunksize = optim_chuncksize(itr, j)
    
    if catch_err is True:
        func = partial(catch_error, func=func, logger=err_logger, return_val=err_return_val)

    if j > 1:
        pool = mp.Pool(processes=j)
        res = list(tqdm(pool.imap(func, itr, chunksize=chunksize), total=len(itr)))
        pool.close()
        pool.terminate()
    else:
        res = [func(x) for x in tqdm(itr)]
    if catch_err:
        err = sum([r == err_return_val for r in res])
        if err_logger is not None:
            log_info(f"{err}/{len(res)} error encountered", err_logger)
        else:
            print(f"{err}/{len(res)} error encountered")
    return res


def mp_run_starmap(func, itr,
                   j=None, chunksize=None, mute_tqdm=False,
                   catch_err=False, err_logger=None, err_return_val=None,
                   **kwargs):
    """Helper function for multiprocessing run using starmap. Each item in iterable should be an iterable of arguments"""

    if kwargs != {}:
        func = partial(func, **kwargs)
    if mute_tqdm:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    if j is None:
        j = min(len(itr), os.cpu_count() // 2)
    if chunksize is None:
        chunksize = optim_chuncksize(itr, j)
    if catch_err is True:
        func = partial(catch_error, func=func, logger=err_logger, return_val=err_return_val)
    
    if j > 1:
        pool = mp.Pool(processes=j)
        res = pool.starmap(func, tqdm(itr), chunksize=chunksize)
        pool.terminate()
    else:
        res = [func(x) for x in tqdm(itr)]
    
    if catch_err:
        err = sum([r == err_return_val for r in res])
        if err_logger is not None:
            log_info(f"{err} / {len(res)} error encountered", err_logger)
        else:
            print(f"{err}/{len(res)} error encountered")
    return res


def subprocess_run(cmd):
    """Run shell subprocess"""
    if isinstance(cmd, str):
        import shlex
        cmd = shlex.split(cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out = ''
    for line in iter(proc.stdout.readline, b''):
        out += line.decode('utf-8')
        print('>>> {}'.format(line.decode('utf-8').rstrip()))

    stdout, stderr = proc.communicate()
    # out = stdout.decode('utf-8').strip('\n')
    err = stderr.decode('utf-8').strip('\n')
    return out, err


