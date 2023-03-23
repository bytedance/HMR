"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import time
import shutil
import argparse
from pathlib import Path
from HMR.data_gen.input_output import convert_pdb_to_pqr, convert_pqr_to_xyzrn
from HMR.utils.helpers import mp_run, subprocess_run


default_pdb2pqr = os.environ.get('PDB2PQR_BIN', subprocess_run('which pdb2pqr30'))


def pdb_to_xyzrn(pdb_fpath, output_root, pdb2pqr_bin):
    """Convert .pdb file to .xyzrn file for MSMS calculation

    Return:
        0 if success, else 1
    """
    pdb_fpath = Path(pdb_fpath)
    pid = pdb_fpath.stem
    output_root = Path(output_root)/pid
    output_root.mkdir(parents=True, exist_ok=True)
    pqr_fpath = output_root/f'{pid}.pqr'
    pqr_log_fpath = output_root/f'{pid}_pdb2pqr.log'
    xyzrn_fpath = output_root/f'{pid}.xyzrn'

    out = convert_pdb_to_pqr(pdb_fpath, pqr_fpath, pdb2pqr_bin, pqr_log_fpath)
    if out == 1:
        return out
    
    convert_pqr_to_xyzrn(pqr_fpath, xyzrn_fpath)
    return 0


def main():
    pdb_list = [str(f) for f in Path(args.MasifLigand_source).glob('*.pdb')]
    mp_run(
        pdb_to_xyzrn,
        pdb_list,
        output_root=args.out_root,
        pdb2pqr_bin=args.pdb2pqr_bin,
        j=args.j, mute_tqdm=args.mute_tqdm,
        check_err=True, err_code=1
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MasifLigand-source', default='.')
    parser.add_argument('--out-root', default='./MasifLigand_mesh')
    parser.add_argument('--pdb2pqr-bin', type=str, default=default_pdb2pqr)
    parser.add_argument('--mute-tqdm', action='store_true')
    parser.add_argument('-j', type=int, default=None)

    args = parser.parse_args()

    assert os.path.exists(args.pdb2pqr_bin)
    assert os.path.exists(args.MasifLigand_source)

    # overwrite output 
    # if os.path.exists(args.out_root):
        # shutil.rmtree(args.out_root)
    os.makedirs(args.out_root, exist_ok=True)

    start = time.time()
    main()
    print(f'MaSIF-ligand data prep step1 time: {(time.time()-start):.1f}s\n')
