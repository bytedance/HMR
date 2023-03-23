"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import time
import pymesh
import argparse
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree
from HMR.data_gen.mol_surface import refine_mesh, check_mesh_validity
from HMR.utils.helpers import mp_run


def extract_lig_pocket(lig_xyz, prot_mesh, radius):
    """Extract the pocket mesh of the ligand

    Interface vertices are those within the 'radius' to any ligand atoms
    Only the largest connected component is returned to avoid disconnected mesh

    Returns:
        pymesh.Mesh: pocket mesh
    """

    # Find iface vertices < radius to the nearest ligand atom
    bt = BallTree(lig_xyz)
    prot_verts = prot_mesh.vertices
    prot_faces = prot_mesh.faces
    d_to_lig, _ = bt.query(prot_verts, k=1)
    iface_ids = np.where(d_to_lig < radius)[0]

    # Skip if the interface is too small (encapsulated ligand)
    if len(iface_ids) <= 100:
        raise ValueError(f"iface vertices {len(iface_ids)} < 100")

    # Extract the interface mesh
    idx_map1 = np.empty(len(prot_verts))
    idx_map1.fill(np.NaN)
    idx_map1[iface_ids] = np.arange(len(iface_ids))
    
    iface_verts = prot_verts[iface_ids]
    iface_faces = idx_map1[prot_faces]
    iface_faces = iface_faces[~np.isnan(iface_faces).any(axis=-1)].astype(int)
    iface_mesh = pymesh.form_mesh(iface_verts, iface_faces)
    iface_mesh.enable_connectivity()
    
    # Extract the largest connected patch as the pocket using BFS
    visited = np.zeros(len(iface_verts)).astype(bool)
    groups = []
    for ivert in range(len(iface_verts)):
        if visited[ivert]:
            continue
        old_visited = visited.copy()
        queue = [ivert]
        visited[ivert] = True
        while queue:
            curr = queue.pop(0)
            for nbr in iface_mesh.get_vertex_adjacent_vertices(curr):
                if not visited[nbr]:
                    queue.append(nbr)
                    visited[nbr] = True
        groups.append(np.where(np.logical_xor(old_visited, visited))[0])
    groups = sorted(groups, key=lambda x:len(x), reverse=True)
    assert sum(len(ig) for ig in groups) == sum(visited) == len(iface_verts)

    # Get the largest patch 
    patch_ids = groups[0]
    assert np.array_equal(patch_ids, sorted(patch_ids))
    idx_map2 = np.zeros(len(iface_verts))
    idx_map2.fill(np.NaN)
    idx_map2[patch_ids] = np.arange(len(patch_ids))
    patch_verts = iface_verts[patch_ids]
    patch_faces = idx_map2[iface_faces]
    patch_faces = patch_faces[~np.isnan(patch_faces).any(axis=-1)]
    patch_mesh = pymesh.form_mesh(patch_verts, patch_faces)
    return patch_mesh


def extract_pockets(msms_fpath, ligand_root, radius, resolution):
    """Extract the ligand binding pocket.

    Args:
        msms_fpath (path): .npz file of full protein mesh
        ligand_root (path): directory contains [pdb_id]_ligand_types.npy and [pdb_id]_ligand_coords.npy
        radius (float): radius to identify interface
        resolution (float): resolution of target mesh
    
    Returns:
        list of float: list of patch sizes extracted
    """
    
    msms_fpath = Path(msms_fpath)
    pid = msms_fpath.stem  # pid_chains
    pdb_id = pid.split('_')[0]
    data_root = msms_fpath.parent

    # load protein mesh
    if not os.path.isfile(msms_fpath):
        print(f"{pid} MSMS protein mesh not found")
        return 0
    msms_npz = np.load(msms_fpath)
    prot_mesh = pymesh.form_mesh(msms_npz['verts'], msms_npz['faces'].astype(int))

    # load ligand coords
    lig_types = [t.decode() for t in np.load(f"{ligand_root}/{pdb_id}_ligand_types.npy")]
    lig_coords = [xyz for xyz in np.load(f"{ligand_root}/{pdb_id}_ligand_coords.npy", allow_pickle=True, encoding='bytes')]
    assert len(lig_types) == len(lig_coords)

    all_patch_size = []

    for ix, (lig_type, lig_coord) in enumerate(zip(lig_types, lig_coords)):
        try:
            neighb_patch = extract_lig_pocket(
                lig_xyz=lig_coord, prot_mesh=prot_mesh, radius=radius
            )

            # refine mesh and check validity
            mesh_refined = refine_mesh(neighb_patch, resolution)
            print(f"{pid}_{ix} verts: {len(neighb_patch.vertices)} -> {len(mesh_refined.vertices)}", flush=True)
            disconnected, has_isolated_verts, has_duplicate_verts, has_abnormal_triangles \
                = check_mesh_validity(mesh_refined, check_triangles=True)
            assert not disconnected, "disconnected"
            assert not has_isolated_verts,'has isolated verts'
            assert not has_duplicate_verts, 'has duplicated verts'
            assert not has_abnormal_triangles, 'has abnormal triangles'
            
            pkt_mesh_fpath = data_root/f'{pid}_patch_{ix}_{lig_type}.npz'
            np.savez(pkt_mesh_fpath, verts=mesh_refined.vertices, faces=mesh_refined.faces) 
            all_patch_size.append(len(mesh_refined.vertices))
        except Exception as e:
            print(f"{pid}_{ix}_{lig_type} error: {e}", flush=True)

    return all_patch_size


def main():

    msms_fpath_list = [
        msms_fpath 
        for msms_fpath in Path(args.data_root).rglob("*.npz") if '_patch_' not in msms_fpath.stem
    ]

    patch_sizes = mp_run(
        extract_pockets, msms_fpath_list,
        ligand_root=args.ligand_root, radius=args.radius, resolution=args.resolution,
        j=args.j, mute_tqdm=args.mute_tqdm,
        check_err=True, err_code=[]
    )

    patch_sizes = [s for patch_size in patch_sizes for s in patch_size]
    print(f"Interface patch size: {np.mean(patch_sizes):.1f} ± {np.std(patch_sizes):.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./MasifLigand_mesh')
    parser.add_argument('--ligand-root', type=str, default='./ligand/')
    parser.add_argument('--resolution', type=float, default=0.5)
    parser.add_argument('--radius', type=float, default=4.0)
    parser.add_argument('--mute-tqdm', action='store_true')
    parser.add_argument('-j', type=int, default=None)
    args = parser.parse_args()

    # IO
    assert os.path.exists(args.data_root)
    assert os.path.exists(args.ligand_root)
    
    start = time.time()
    main()
    print(f'MaSIF-ligand step3 time: {(time.time()-start):.1f}s\n')
