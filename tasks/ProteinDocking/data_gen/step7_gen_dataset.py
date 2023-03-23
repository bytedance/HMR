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
from sklearn.neighbors import BallTree
from HMR.geomlib.trimesh import TriMesh
from HMR.data_gen.chemistry import atom_type_dict, res_type_dict

# read xyzrn file for atomic features
def read_xyzrn_file(data_dir, suffix):
    xyzrn_fpath = os.path.join(data_dir, f'{suffix}.xyzrn')
    assert os.path.isfile(xyzrn_fpath)
    atom_info = []
    with open(xyzrn_fpath, 'r') as f:
        for line in f.readlines(): 
            line_info = line.rstrip().split()
            assert len(line_info) == 6
            full_id = line_info[-1]
            assert len(full_id.split('_')) == 6
            res_name, res_id, atom_name, atom_id, charge, radius = full_id.split('_')
            assert res_name in res_type_dict
            assert atom_name[0] in atom_type_dict
            alpha_carbon = atom_name.upper() == 'CA'
            atom_info.append(line_info[:3] + 
                             [res_type_dict[res_name],
                              atom_type_dict[atom_name[0]],
                              float(charge),
                              float(radius),
                              alpha_carbon])

    return np.array(atom_info, dtype=float)


def gen_dataset(data_root, out_dir, pair_name, min_num_eigs, eigs_ratio):
    # IO
    fout = os.path.join(out_dir, f'{pair_name}.npz')
    if os.path.isfile(fout):
        return

    data_dir = os.path.join(data_root, pair_name)
    lig_mesh_fpath = os.path.join(data_dir, 'ligand_mesh.npz')
    rec_mesh_fpath = os.path.join(data_dir, 'receptor_mesh.npz')
    if not (os.path.isfile(lig_mesh_fpath) and \
            os.path.isfile(rec_mesh_fpath)):
        return
    
    # atomic features
    lig_atom_info = read_xyzrn_file(data_dir, suffix='ligand') 
    rec_atom_info = read_xyzrn_file(data_dir, suffix='receptor')

    # load surface mesh
    lig_mesh = np.load(lig_mesh_fpath)
    rec_mesh = np.load(rec_mesh_fpath)

    # apply filter
    lig_verts = lig_mesh['verts']
    lig_num_verts = lig_verts.shape[0]
    rec_verts = rec_mesh['verts']
    rec_num_verts = rec_verts.shape[0]
    iface_cutoff = 3.0
    bt = BallTree(lig_verts)
    dist, _ = bt.query(rec_verts, k=1)
    iface_size = len(np.where(dist < iface_cutoff)[0])
    if (min(len(lig_verts), len(rec_verts)) < 300 or \
        max(iface_size/lig_num_verts, iface_size/rec_num_verts) > 0.9):
        return

    # ligand Laplace-Beltrami basis
    lig_num_eigs = max(min_num_eigs, int(eigs_ratio*lig_num_verts)+1)
    assert lig_num_eigs < lig_num_verts
    lig_trimesh = TriMesh(verts=lig_verts, faces=lig_mesh['faces'])
    lig_trimesh.LB_decomposition(k=lig_num_eigs) # scipy eigsh must have k < N

    # receptor Laplace-Beltrami basis
    rec_num_eigs = max(min_num_eigs, int(eigs_ratio*rec_num_verts)+1)
    assert rec_num_eigs < rec_num_verts
    rec_trimesh = TriMesh(verts=rec_verts, faces=rec_mesh['faces'])
    rec_trimesh.LB_decomposition(k=rec_num_eigs) # scipy eigsh must have k < N
    
    # save features
    np.savez_compressed(fout,
                        # ligand
                        lig_atom_info=lig_atom_info.astype(np.float32),
                        lig_verts=lig_verts.astype(np.float32),
                        lig_faces=lig_mesh['faces'].astype(np.float32),   
                        lig_eigen_vals=lig_trimesh.eigen_vals.astype(np.float32),
                        lig_eigen_vecs=lig_trimesh.eigen_vecs.astype(np.float32),
                        lig_mass=lig_trimesh.mass.astype(np.float32),
                        # receptor
                        rec_atom_info=rec_atom_info.astype(np.float32),
                        rec_verts=rec_verts.astype(np.float32),
                        rec_faces=rec_mesh['faces'].astype(np.float32),
                        rec_eigen_vals=rec_trimesh.eigen_vals.astype(np.float32),
                        rec_eigen_vecs=rec_trimesh.eigen_vecs.astype(np.float32),
                        rec_mass=rec_trimesh.mass.astype(np.float32)
                        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eigs-ratio', type=float, default=0.06)
    parser.add_argument('--min-num-eigs', type=int, default=100)
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
    rcsb_dataset_dir = './dataset_RCSB/'
    os.makedirs(rcsb_dataset_dir, exist_ok=True)

    # RCSB timer
    start = time.time()

    if not args.serial:
        pool = multiprocessing.Pool(processes=args.j)
        pool_args = [(rcsb_mesh_dir, rcsb_dataset_dir, pair_name, args.min_num_eigs, args.eigs_ratio)
                     for pair_name in os.listdir(rcsb_mesh_dir)]
        pool.starmap(gen_dataset, tqdm(pool_args), chunksize=10)
        pool.terminate()
        print('All processes successfully finished')
    else:
        for pair_name in tqdm(os.listdir(rcsb_mesh_dir)):
            gen_dataset(rcsb_mesh_dir, rcsb_dataset_dir, pair_name, args.min_num_eigs, args.eigs_ratio)
    
    print(f'RCSB step7 elapsed time: {(time.time()-start):.1f}s\n')


