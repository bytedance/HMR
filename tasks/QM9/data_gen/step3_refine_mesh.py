"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import pymesh
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from step2_compute_msms import check_mesh_validity

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
def refine_mesh(data_root, gdb_id, resolution):
    data_dir = os.path.join(data_root, gdb_id)
    # read MSMS mesh
    msms_mesh_file = os.path.join(data_dir, f'{gdb_id}_msms.npz')
    if not os.path.isfile(msms_mesh_file):
        return
    msms_npz = np.load(msms_mesh_file)
    mesh_msms = pymesh.form_mesh(msms_npz['verts'], msms_npz['faces'].astype(int))

    # refine mesh
    mesh, _ = pymesh.remove_duplicated_vertices(mesh_msms, 1E-6)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, _ = pymesh.split_long_edges(mesh, resolution)
    num_vertices = mesh.num_vertices
    iteration = 0
    while iteration < 10:
        mesh, _ = pymesh.collapse_short_edges(mesh, 1E-6)
        mesh, _ = pymesh.collapse_short_edges(mesh, 0.1*resolution)
        mesh, _ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if abs(mesh.num_vertices - num_vertices) < 10:
            break
        num_vertices = mesh.num_vertices
        iteration += 1
    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, _ = pymesh.remove_obtuse_triangles(mesh, 179.0, 100)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    mesh = remove_abnormal_triangles(mesh)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)

    # final checkup
    groups, has_isolated_verts, has_duplicate_verts, has_abnormal_triangles \
        = check_mesh_validity(mesh, check_triangles=True)
    # apply filters
    if not ((len(groups) == 1) and (not has_isolated_verts) and \
            (not has_duplicate_verts) and (not has_abnormal_triangles)):
        print(f'skip {gdb_id} due to poor mesh quality')
        print(f'\tgroup sizes: {[len(g) for g in groups]}')
        print(f'\thas isolated verts: {has_isolated_verts}')
        print(f'\thas duplicate verts: {has_duplicate_verts}')
        print(f'\thas abnormal triangles: {has_abnormal_triangles}\n')
        return
        
    # save refined mesh
    mesh_file = os.path.join(data_dir, f'{gdb_id}_mesh.npz')
    np.savez(mesh_file, verts=mesh.vertices, faces=mesh.faces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./QM9_mesh/')
    parser.add_argument('--resolution', type=float, default=1.0)
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    args = parser.parse_args()
    print(args)

    # specify IO dir
    assert os.path.exists(args.data_root)
    gdb_ids = os.listdir(args.data_root)

    if not args.serial:
        pool = multiprocessing.Pool(processes=args.j)
        pool_args = [(args.data_root, gdb_id, args.resolution) for gdb_id in gdb_ids]
        pool.starmap(refine_mesh, tqdm(pool_args), chunksize=10)
        pool.terminate()
        print('All processes successfully finished')
    else:
        for gdb_id in tqdm(gdb_ids):
            refine_mesh(args.data_root, gdb_id, args.resolution)


