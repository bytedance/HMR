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
from pathlib import Path
from HMR.geomlib.trimesh import TriMesh
from HMR.data_gen.input_output import read_xyzrn_file
from HMR.utils.helpers import mp_run


ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
type_idx = {type_: ix for ix, type_ in enumerate(ligands)}


def gen_dataset(pkt_mesh_fpath, out_dir, max_eigen_val):
    """Generate dataset .npz file for a pocket

    Args:
        pkt_mesh_fpath (path): pocket mesh .npz file
        out_dir (path): output directory of generated dataset
        max_eigen_val (float): the maximum eigen values for the patch

    Returns:
        number of eigenbases if success, else np.nan
    """
    
    # IO
    pkt_mesh_fpath = Path(pkt_mesh_fpath)
    mesh_root = pkt_mesh_fpath.parent
    pkt_name = pkt_mesh_fpath.stem
    pdb_id, chain_id, _, lid, lig_type = pkt_name.split('_')
    pid = pdb_id + '_' + chain_id
    atom_info_fpath = mesh_root/f'{pid}.xyzrn'
    fout = os.path.join(out_dir, f'{pkt_name}.npz')
    assert pkt_mesh_fpath.exists()
    
    # load files
    atom_info = read_xyzrn_file(atom_info_fpath)
    pkt_mesh = np.load(pkt_mesh_fpath)

    # An iterative way to compute Laplace-Beltrami basis with maximum eigenvalue ~ 5
    num_verts = len(pkt_mesh['verts'])
    if num_verts < 100:
        print(f"{pkt_name} has < 100 vertices, discard", flush=True)
        return np.nan

    trimesh = TriMesh(verts=pkt_mesh['verts'], faces=pkt_mesh['faces'].astype(int))
    num_eigs = int(0.16 * num_verts)
    max_val = 0
    while max_val < max_eigen_val:
        num_eigs += 5
        if num_eigs >= num_verts:
            print(f'WARN: {pid}_{lid}_{lig_type}: num_eigs >= num_verts, stop processing..', flush=True)
        trimesh.LB_decomposition(k=num_eigs) # scipy eigsh must have k < N
        max_val = np.max(trimesh.eigen_vals)
    cutoff = np.argmax(trimesh.eigen_vals > max_eigen_val)
    eigen_vals = trimesh.eigen_vals[:cutoff]
    eigen_vecs = trimesh.eigen_vecs[:, :cutoff]

    # save features
    np.savez_compressed(
        fout,
        label=type_idx[lig_type],
        # pocket
        atom_info=atom_info.astype(np.float32),
        pkt_verts=pkt_mesh['verts'].astype(np.float32),
        pkt_faces=pkt_mesh['faces'].astype(np.float32),
        eigen_vals=eigen_vals.astype(np.float32),
        eigen_vecs=eigen_vecs.astype(np.float32),
        mass=trimesh.mass.astype(np.float32)
    )
    return cutoff


def main():

    pkt_fpath_list =  [pkt_fpath for pkt_fpath in Path(args.data_root).rglob("*_patch_*_*.npz")]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cutoffs = mp_run(
        gen_dataset, pkt_fpath_list,
        out_dir=args.out_dir, max_eigen_val=args.max_eigen_val,
        j=args.j, check_err=True, err_code=np.nan
    )
    
    print(f"Num of LB basis: {np.nanmean(cutoffs)} ± {np.nanstd(cutoffs)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./MasifLigand_mesh')
    parser.add_argument('--out-dir', type=str, default='dataset-MasifLigand')
    parser.add_argument('--max-eigen-val', type=float, default=5.)
    parser.add_argument('-j', type=int, default=4)
    parser.add_argument('--mute-tqdm', action='store_true')
    args = parser.parse_args()

    assert os.path.exists(args.data_root)
    
    start = time.time()
    main()
    print(f'MaSIF-ligand step4 elapsed time: {(time.time()-start):.1f}s\n')
