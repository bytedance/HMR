"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import time
import argparse
from pathlib import Path
from HMR.data_gen.mol_surface import compute_ses
from HMR.utils.helpers import mp_run

default_msms = os.environ.get('MSMS_BIN', '')

def msms(xyzrn_fpath, probe_radius, density, msms_bin):
    xyzrn_fpath = Path(xyzrn_fpath)
    mesh_prefix = str(xyzrn_fpath.parent/xyzrn_fpath.stem)
    return compute_ses(str(xyzrn_fpath), mesh_prefix, probe_radius, density, msms_bin, print_out=False)


def main():
    # available .xyzrn files
    xyzrn_list = [str(f) for f in Path(args.data_root).rglob('*.xyzrn')]

    mp_run(
        msms, xyzrn_list,
        probe_radius=args.probe_radius, density=args.density, msms_bin=args.msms_bin,
        j=args.j, mute_tqdm=args.mute_tqdm,
        check_err=True, err_code=1
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./MasifLigand_mesh')
    parser.add_argument('--msms-bin', type=str, default=default_msms)
    parser.add_argument('--probe-radius', type=float, default=1.0)
    parser.add_argument('--density', type=float, default=2.0)
    parser.add_argument('--mute-tqdm', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    
    args = parser.parse_args()

    assert os.path.exists(args.msms_bin)
    assert os.path.exists(args.data_root)

    start = time.time()
    main()
    print(f'MaSIF-ligand step2 time: {(time.time()-start):.1f}s\n')
