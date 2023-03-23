"""Helper functions

-------
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import sys
import torch
import random
import logging
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from subprocess import Popen, PIPE, SubprocessError
from functools import partial


def subprocess_run(cmd, print_out=True, out_log=None, err_ignore=False):
    """Run a shell subprocess. 
    Input cmd can be a list or command string

    Args:
        print_out (bool): print output to screen
        out_log (path): also log output to a file
        err_ignore (bool): don't raise error message. default False.

    Returns:
        out (str): standard output, utf-8 decoded
        err (str): standard error, utf-8 decoded
    """
    if isinstance(cmd, str):
        import shlex
        cmd = shlex.split(cmd)
    
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    
    # print output
    if print_out:
        out = ''
        for line in iter(proc.stdout.readline, b''):
            out += line.decode('utf-8')
            print('>>> {}'.format(line.decode('utf-8').rstrip()), flush=True)
        _, stderr = proc.communicate()
    else:
        stdout, stderr = proc.communicate()
        out = stdout.decode('utf-8').strip('\n')
    err = stderr.decode('utf-8').strip('\n')

    # log output to file
    if out_log is not None:
        with open(out_log, 'w') as handle:
            handle.write(out)
    
    if not err_ignore and err != '':
        raise SubprocessError(f"Error encountered: {' '.join(cmd)}\n{err}")

    return out, err


def optim_chuncksize(itr, j):
    """Compute an optimal chuncksize for multiprocess: min(len(itr) // (j * 20), 1)"""
    return max(len(itr) // j // 20, 1)


def mp_run(
    func, itr,
    j=None, chunksize=None, mute_tqdm=False,
    check_err=False, err_code=1,
    **kwargs
):
    """Multiprocessing run using Pool.imap, each item in the iterable should have only one argument

    Args:
        j (int): number of processes. Default: 0.5 * cpu_count
        chunksize (int): chunksize to split processes. Default: min(len(itr) // (j * 20), 1)
        mute_tqdm (bool): suppress tqdm progress bar. Default False
        check_err (bool): if report how many error encountered
        err_code (int or str): code to identify errors
        **kwargs: other keyword argument to pass to func
    """

    if kwargs != {}:
        func = partial(func, **kwargs)
    if j is None:
        j = min(len(itr), os.cpu_count() // 2)
    if chunksize is None:
        chunksize = optim_chuncksize(itr, j)

    if j > 1:
        pool = mp.Pool(processes=j)
        res = list(tqdm(pool.imap(func, itr, chunksize=chunksize), total=len(itr), disable=mute_tqdm))
        pool.close()
        pool.terminate()
    else:
        res = [func(x) for x in tqdm(itr, disable=mute_tqdm)]
    
    if check_err:
        err = sum([r == err_code for r in res])
        print(f"{err:,}/{len(res):,} error encountered")
    return res


def set_logger(log_fpath):
    """Set file logger at log_fpath"""
    Path(log_fpath).parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s", 
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_fpath), 'a'),
        ]
    )


def set_seed(seed):
    """Set all random seeds"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operations have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False