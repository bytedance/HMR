"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

# atomic van der waals radii in Angstrom unit
vdw_radii_dict = np.array([0, 1.1, 0, 0, 0, 0, 1.7, 1.55, 1.52, 1.47], dtype=np.float32)

# prepare input for MSMS surface computation
def convert_to_xyzrn(src_fpath, out_root):
    src_data = np.load(src_fpath)
    index = src_data['index']
    split = np.ones_like(index) # training set
    if 'valid' in src_fpath:
        split += 1 # valid set
    elif 'test' in src_fpath:
        split += 2 # test set
    
    # atomic info
    num_atoms = src_data['num_atoms']
    positions = src_data['positions']
    charges = src_data['charges']
    for idx in tqdm(range(len(num_atoms))):
        gdb_id = 'gdb_' + str(index[idx]).rjust(6, '0')
        natom = num_atoms[idx]
        atom_xyz = positions[idx, :natom]
        atom_Z = charges[idx, :natom].astype(int)
        atom_r = vdw_radii_dict[atom_Z]
        atom_rn = [f'{atom_r[i]} 1 {atom_Z[i]}' for i in range(natom)]
        
        # write xyzrn file
        out_dir = os.path.join(out_root, gdb_id)
        os.makedirs(out_dir, exist_ok=False)
        xyzrn_path = os.path.join(out_dir, f'{gdb_id}.xyzrn')
        with open(xyzrn_path, 'w') as f:
            for i in range(natom):
                coords = '{:.6f} {:.6f} {:.6f} '.format(*atom_xyz[i])
                f.write(coords + atom_rn[i] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qm9-source', type=str, default='')
    parser.add_argument('--out-root', type=str, default='./QM9_mesh/')
    args = parser.parse_args()
    print(args)

    # specify IO dir
    assert os.path.exists(args.qm9_source)
    if os.path.exists(args.out_root):
        shutil.rmtree(args.out_root)
    os.makedirs(args.out_root, exist_ok=False)

    # convert to xyzrn file
    for split in ['train', 'valid', 'test']:
        src_fpath = os.path.join(args.qm9_source, f'{split}.npz')
        assert os.path.isfile(src_fpath)
        convert_to_xyzrn(src_fpath, args.out_root)


