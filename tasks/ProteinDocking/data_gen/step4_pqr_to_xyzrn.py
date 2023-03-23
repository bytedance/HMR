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
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partialmethod
from HMR.data_gen.chemistry import atom_type_dict, res_type_dict

def parse_pqr_file(pqr_fpath):
    with open(pqr_fpath, 'r') as f:
        f_read = f.readlines()
    xyz_list = [] # atomic coordinates
    rn_list = [] # radius and descriptions
    for line in f_read:
        if line[:4] == 'ATOM':
            assert (len(line) == 70) and (line[69] == '\n')
            atom_id = int(line[6:11]) # 1-based indexing
            assert line[11] == ' '
            atom_name = line[12:16].strip()
            assert atom_name[0] in atom_type_dict
            assert line[16] == ' '
            res_name = line[17:20]
            if not res_name in res_type_dict:
                res_name = 'UNK'
            res_id = int(line[22:26].strip()) # 1-based indexing
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            assert line[54] == ' '
            charge = float(line[55:62])
            assert line[62] == ' '
            radius = float(line[63:69])
            xyz_list.append([x, y, z])
            full_id = f'{res_name}_{res_id:d}_{atom_name}_{atom_id:d}_{charge:.4f}_{radius:.4f}'
            rn_list.append(str(radius) + ' 1 ' + full_id)
    
    return np.array(xyz_list, dtype=float), rn_list


# prepare input for MSMS surface computation
def convert_pqr_to_xyzrn(data_dir):
    lig_pqr_fpath = os.path.join(data_dir, 'ligand.pqr')
    rec_pqr_fpath = os.path.join(data_dir, 'receptor.pqr')
    if not (os.path.isfile(lig_pqr_fpath) and
            os.path.isfile(rec_pqr_fpath)):
        return

    # parse pqr file
    lig_xyz, lig_rn = parse_pqr_file(lig_pqr_fpath)
    rec_xyz, rec_rn = parse_pqr_file(rec_pqr_fpath)

    # skip under/oversized cases
    if min(len(lig_xyz), len(rec_xyz)) < 300 or \
       max(len(lig_xyz), len(rec_xyz)) > 12000:
        return
    
    # write ligand xyzrn file
    lig_xyzrn_fpath = os.path.join(data_dir, 'ligand.xyzrn')
    with open(lig_xyzrn_fpath, 'w') as f:
        for idx in range(len(lig_xyz)):
            coords = '{:.6f} {:.6f} {:.6f} '.format(*lig_xyz[idx])
            f.write(coords + lig_rn[idx] + '\n')
    
    # write receptor xyzrn file
    rec_xyzrn_fpath = os.path.join(data_dir, 'receptor.xyzrn')
    with open(rec_xyzrn_fpath, 'w') as f:
        for idx in range(len(rec_xyz)):
            coords = '{:.6f} {:.6f} {:.6f} '.format(*rec_xyz[idx])
            f.write(coords + rec_rn[idx] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    parser.add_argument('--mute-tqdm', action='store_true')
    args = parser.parse_args()
    print(args)

    # optionally mute tqdm
    if args.mute_tqdm:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # RCSB
    rcsb_mesh_dir = './RCSB_mesh/'
    assert os.path.exists(rcsb_mesh_dir)
    pair_paths = [os.path.join(rcsb_mesh_dir, pair_name) for pair_name in os.listdir(rcsb_mesh_dir)]

    start = time.time()

    if not args.serial:
        pool = multiprocessing.Pool(processes=args.j)
        pool.map(convert_pqr_to_xyzrn, tqdm(pair_paths), chunksize=10)
        pool.terminate()
        print('All processes successfully finished')
    else:
        for data_dir in tqdm(pair_paths):
            convert_pqr_to_xyzrn(data_dir)
    
    print(f'RCSB step4 elapsed time: {(time.time()-start):.1f}s\n')


