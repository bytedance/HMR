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
import multiprocessing
from HMR.geomlib.trimesh import TriMesh
from step2_compute_msms import read_xyzrn_file

def gen_dataset(data_root, gdb_id, out_dir, max_eigen_val):
    data_dir = os.path.join(data_root, gdb_id)
    
    # read PyMesh refined mesh
    mesh_file = os.path.join(data_dir, f'{gdb_id}_mesh.npz')
    if not os.path.isfile(mesh_file):
        return
    mesh = np.load(mesh_file)
    
    # # Laplace-Beltrami basis
    num_verts = len(mesh['verts'])
    assert num_verts > 100
    trimesh = TriMesh(verts=mesh['verts'], faces=mesh['faces'].astype(int))
    num_eigs = int(0.11 * num_verts)
    max_val = 0
    while max_val < max_eigen_val:
        num_eigs += 5
        if num_eigs >= num_verts:
            print('num_eigs >= num_verts, stop processing..', flush=True)
        trimesh.LB_decomposition(k=num_eigs) # scipy eigsh must have k < N
        max_val = np.max(trimesh.eigen_vals)
    cutoff = np.argmax(trimesh.eigen_vals > max_eigen_val)
    eigen_vals = trimesh.eigen_vals[:cutoff]
    eigen_vecs = trimesh.eigen_vecs[:, :cutoff]
    
    # atomic info
    xyzrn_file = os.path.join(data_dir, f'{gdb_id}.xyzrn')
    atom_info = read_xyzrn_file(xyzrn_file)

    # save features
    fout = os.path.join(out_dir, f'{gdb_id}.npz')
    np.savez_compressed(fout, verts=mesh['verts'].astype(np.float32),
                              faces=mesh['faces'].astype(np.float32),
                              atom_info=atom_info.astype(np.float32),
                              eigen_vals=eigen_vals.astype(np.float32),
                              eigen_vecs=eigen_vecs.astype(np.float32),
                              mass=trimesh.mass.astype(np.float32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./QM9_mesh/')
    parser.add_argument('--out-dir', type=str, default='./dataset_QM9/')
    parser.add_argument('--max-eigen-val', type=float, default=5.)
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('-j', type=int, default=2)
    args = parser.parse_args()
    print(args)

    # specify IO dir
    assert os.path.exists(args.data_root)
    gdb_ids = os.listdir(args.data_root)
    out_dir = args.out_dir
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=False)

    if not args.serial:
        pool = multiprocessing.Pool(processes=args.j)
        pool_args = [(args.data_root, gdb_id, out_dir, args.max_eigen_val) for gdb_id in gdb_ids]
        pool.starmap(gen_dataset, tqdm(pool_args), chunksize=10)
        pool.terminate()
        print('All processes successfully finished')
    else:
        for gdb_id in tqdm(gdb_ids):
            gen_dataset(args.data_root, gdb_id, out_dir, args.max_eigen_val)


