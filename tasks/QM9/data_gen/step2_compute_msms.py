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
import pandas as pd
from tqdm import tqdm
import multiprocessing
from subprocess import Popen, PIPE

# parse MSMS output
def read_msms(data_dir, prefix):
    mesh_prefix = os.path.join(data_dir, prefix)
    assert os.path.isfile(mesh_prefix + '.vert')
    assert os.path.isfile(mesh_prefix + '.face')
    
    # vertices
    with open(mesh_prefix + '.vert') as f:
        vert_data = f.read().rstrip().split('\n')
    num_verts = int(vert_data[2].split()[0])
    assert num_verts == len(vert_data) - 3
    vertices = []
    for idx in range(3, len(vert_data)):
        ifields = vert_data[idx].split()
        assert len(ifields) == 10
        vertices.append(ifields[:3])
    assert len(vertices) == num_verts

    # faces
    with open(mesh_prefix + '.face') as f:
        face_data = f.read().rstrip().split('\n')
    num_faces = int(face_data[2].split()[0])
    assert num_faces == len(face_data) - 3
    faces = []
    for idx in range(3, len(face_data)):
        ifields = face_data[idx].split()
        assert len(ifields) == 5
        faces.append(ifields[:3]) # one-based, to be converted
    assert len(faces) == num_faces

    # solvent excluded surface info
    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int) - 1 # convert to zero-based indexing
    assert np.amin(faces) == 0
    
    return vertices, faces


# read xyzrn file for atomic features
charge_set = set([1, 6, 7, 8, 9])
def read_xyzrn_file(xyzrn_fpath):
    assert os.path.isfile(xyzrn_fpath)
    atom_info = []
    with open(xyzrn_fpath, 'r') as f:
        for line in f.readlines(): 
            line_info = line.rstrip().split()
            assert len(line_info) == 6
            charge = int(line_info[-1])
            assert charge in charge_set
            atom_info.append([line_info[0], line_info[1], line_info[2], charge])

    return np.array(atom_info, dtype=float)


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
    
    return groups, has_isolated_verts, has_duplicate_verts, has_abnormal_triangles


# use MSMS to compute molecular solvent excluded surface
def compute_msms(data_root, gdb_id, msms_bin):
    # specify IO dir
    data_dir = os.path.join(data_root, gdb_id)
    # xyzrn file
    xyzrn_file = os.path.join(data_dir, f'{gdb_id}.xyzrn')
    assert os.path.isfile(xyzrn_file)
    mesh_prefix = os.path.join(data_dir, f'{gdb_id}')
    
    # run MSMS
    msms_args = [msms_bin, '-if', xyzrn_file, '-of', mesh_prefix, \
                    '-probe_radius', '1.0', '-density', '3.0']
    proc = Popen(msms_args, stdout=PIPE, stderr=PIPE)
    _, stderr = proc.communicate()
    # skip if MSMS failed
    errmsg = stderr.decode('utf-8')
    if 'ERROR' in errmsg or 'WARNING' in errmsg:
        print(f'skip {gdb_id}\n  {errmsg}')
        return
    if not (os.path.isfile(mesh_prefix+'.vert') and \
            os.path.isfile(mesh_prefix+'.face')):
        print(f'skip {gdb_id} due to missing MSMS output', flush=True)
        return
    
    # check mesh validity
    verts, faces = read_msms(data_dir, gdb_id)
    mesh = pymesh.form_mesh(verts, faces)
    groups, has_isolated_verts, has_duplicate_verts, _ = check_mesh_validity(mesh, check_triangles=False)
    # apply filters
    if not ((len(groups) == 1) and (not has_isolated_verts) and (not has_duplicate_verts)):
        print(f'skip {gdb_id} due to poor mesh quality')
        print(f'\tgroup sizes: {[len(g) for g in groups]}')
        print(f'\thas isolated verts: {has_isolated_verts}')
        print(f'\thas duplicate verts: {has_duplicate_verts}')
        return

    # save MSMS mesh
    mesh_file = os.path.join(data_dir, f'{gdb_id}_msms.npz')
    np.savez(mesh_file, verts=verts, faces=faces)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./QM9_mesh/')
    parser.add_argument('--msms-bin', type=str, default='')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    args = parser.parse_args()
    print(args)

    # specify IO dir
    assert os.path.exists(args.data_root)
    gdb_ids = os.listdir(args.data_root)

    # MSMS executable
    assert os.path.isfile(args.msms_bin)

    if not args.serial:
        pool = multiprocessing.Pool(processes=args.j)
        pool_args = [(args.data_root, gdb_id, args.msms_bin) for gdb_id in gdb_ids]
        pool.starmap(compute_msms, tqdm(pool_args), chunksize=10)
        pool.terminate()
        print('All processes successfully finished')
    else:
        for gdb_id in tqdm(gdb_ids):
            compute_msms(args.data_root, gdb_id, args.msms_bin)


