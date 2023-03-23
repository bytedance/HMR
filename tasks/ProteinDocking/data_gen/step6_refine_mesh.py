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
import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import partialmethod

def remove_abnormal_triangles(mesh):
    verts = mesh.vertices
    faces = mesh.faces
    v1 = verts[faces[:, 0]]
    v2 = verts[faces[:, 1]]
    v3 = verts[faces[:, 2]]
    e1 = v3 - v2
    e2 = v1 - v3
    e3 = v2 - v1
    L1 = np.linalg.norm(e1, axis=1)
    L2 = np.linalg.norm(e2, axis=1)
    L3 = np.linalg.norm(e3, axis=1)
    cos1 = np.einsum('ij,ij->i', -e2, e3) / (L2 * L3)
    cos2 = np.einsum('ij,ij->i', e1, -e3) / (L1 * L3)
    cos3 = np.einsum('ij,ij->i', -e1, e2) / (L1 * L2)
    cos123 = np.concatenate((cos1.reshape(-1, 1), 
                             cos2.reshape(-1, 1),
                             cos3.reshape(-1, 1)), axis=-1)
    valid_faces = np.where(np.all(1 - cos123**2 > 1E-5, axis=-1))[0]
    faces_new = faces[valid_faces]

    return pymesh.form_mesh(verts, faces_new)


# refine MSMS surface mesh
def refine_mesh_impl(mesh_msms, resolution):
    mesh, _ = pymesh.remove_duplicated_vertices(mesh_msms, 1E-6)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, _ = pymesh.split_long_edges(mesh, resolution)
    num_verts = mesh.num_vertices
    iteration = 0
    while iteration < 10:
        mesh, _ = pymesh.collapse_short_edges(mesh, 1E-6)
        mesh, _ = pymesh.collapse_short_edges(mesh, resolution)
        mesh, _ = pymesh.remove_obtuse_triangles(mesh, 170.0, 100)
        if abs(mesh.num_vertices - num_verts) < 20:
            break
        num_verts = mesh.num_vertices
        iteration += 1
    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, _ = pymesh.remove_obtuse_triangles(mesh, 179.0, 100)
    mesh = remove_abnormal_triangles(mesh)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)

    return mesh


# skip surface with poor mesh quality
def check_mesh_validity(mesh, check_triangles=False):
    mesh.enable_connectivity()
    verts, faces = mesh.vertices, mesh.faces
    
    # check if a manifold is all-connected using BFS
    visited = np.zeros(len(verts)).astype(bool)
    groups = []
    for ivert in range(len(verts)):
        if visited[ivert]:
            continue
        old_visited = visited.copy()
        queue = [ivert]
        visited[ivert] = True
        while queue:
            curr = queue.pop(0)
            for nbr in mesh.get_vertex_adjacent_vertices(curr):
                if not visited[nbr]:
                    queue.append(nbr)
                    visited[nbr] = True
        groups.append(np.where(np.logical_xor(old_visited, visited))[0])
    groups = sorted(groups, key=lambda x:len(x), reverse=True)
    assert sum(len(ig) for ig in groups) == sum(visited) == len(verts)
    disconnected = len(groups) > 1
    
    # check for isolated vertices
    valid_verts = np.unique(faces)
    has_isolated_verts = verts.shape[0] != len(valid_verts)

    # check for faces with duplicate vertices
    df = pd.DataFrame(faces)
    df = df[df.nunique(axis=1) == 3]
    has_duplicate_verts = df.shape[0] != mesh.num_faces

    # check for abnormal triangles
    if check_triangles:
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]
        e1 = v3 - v2
        e2 = v1 - v3
        e3 = v2 - v1
        L1 = np.linalg.norm(e1, axis=1)
        L2 = np.linalg.norm(e2, axis=1)
        L3 = np.linalg.norm(e3, axis=1)
        cos1 = np.einsum('ij,ij->i', -e2, e3) / (L2 * L3)
        cos2 = np.einsum('ij,ij->i', e1, -e3) / (L1 * L3)
        cos3 = np.einsum('ij,ij->i', -e1, e2) / (L1 * L2)
        cos123 = np.concatenate((cos1.reshape(-1, 1), 
                                 cos2.reshape(-1, 1),
                                 cos3.reshape(-1, 1)), axis=-1)
        valid_faces = np.where(np.all(1 - cos123**2 >= 1E-5, axis=-1))[0]
        has_abnormal_triangles = faces.shape[0] != len(valid_faces)
    else:
        has_abnormal_triangles = False
    
    return disconnected, has_isolated_verts, has_duplicate_verts, has_abnormal_triangles


def refine_mesh(data_dir, resolution):
    # load surface mesh
    lig_msms_fpath = os.path.join(data_dir, 'ligand_msms.npz')
    rec_msms_fpath = os.path.join(data_dir, 'receptor_msms.npz')
    if not (os.path.isfile(lig_msms_fpath) and \
            os.path.isfile(rec_msms_fpath)):
        return
    lig_msms_npz = np.load(lig_msms_fpath)
    lig_mesh_msms = pymesh.form_mesh(lig_msms_npz['verts'], lig_msms_npz['faces'].astype(int))
    rec_msms_npz = np.load(rec_msms_fpath)
    rec_mesh_msms = pymesh.form_mesh(rec_msms_npz['verts'], rec_msms_npz['faces'].astype(int))

    # refine mesh
    lig_mesh = refine_mesh_impl(lig_mesh_msms, resolution)
    rec_mesh = refine_mesh_impl(rec_mesh_msms, resolution)

    # check refined mesh validity
    lig_disconnected, lig_has_isolated_verts, lig_has_duplicate_verts, lig_has_abnormal_triangles \
        = check_mesh_validity(lig_mesh, check_triangles=True)
    rec_disconnected, rec_has_isolated_verts, rec_has_duplicate_verts, rec_has_abnormal_triangles \
        = check_mesh_validity(rec_mesh, check_triangles=True)
    # apply filters
    if lig_disconnected or lig_has_isolated_verts or lig_has_duplicate_verts or lig_has_abnormal_triangles \
        or rec_disconnected or rec_has_isolated_verts or rec_has_duplicate_verts or rec_has_abnormal_triangles:
        print(f'skip {data_dir} due to poor refined mesh quality')
        print(f'\tlig disconnected: {lig_disconnected}')
        print(f'\tlig has isolated verts: {lig_has_isolated_verts}')
        print(f'\tlig has duplicate verts: {lig_has_duplicate_verts}')
        print(f'\tlig has abnormal triangles: {lig_has_abnormal_triangles}')
        print(f'\trec disconnected: {rec_disconnected}')
        print(f'\trec has isolated verts: {rec_has_isolated_verts}')
        print(f'\trec has duplicate verts: {rec_has_duplicate_verts}')
        print(f'\trec has abnormal triangles: {rec_has_abnormal_triangles}\n', flush=True)
        return

    # save ligand mesh
    lig_mesh_fpath = os.path.join(data_dir, 'ligand_mesh.npz')
    np.savez(lig_mesh_fpath, verts=lig_mesh.vertices, faces=lig_mesh.faces) 

    # save receptor mesh
    rec_mesh_fpath = os.path.join(data_dir, 'receptor_mesh.npz')
    np.savez(rec_mesh_fpath, verts=rec_mesh.vertices, faces=rec_mesh.faces)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=float, default=1.2)
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

    # RCSB timer
    start = time.time()

    if not args.serial:
        pool = multiprocessing.Pool(processes=args.j)
        pool_args = [(os.path.join(rcsb_mesh_dir, pair_name), args.resolution) \
                     for pair_name in os.listdir(rcsb_mesh_dir)]
        pool.starmap(refine_mesh, tqdm(pool_args), chunksize=10)
        pool.terminate()
        print('All processes successfully finished')
    else:
        for pair_name in tqdm(os.listdir(rcsb_mesh_dir)):
            refine_mesh(os.path.join(rcsb_mesh_dir, pair_name), args.resolution)
    
    print(f'RCSB step6 elapsed time: {(time.time()-start):.1f}s\n')


