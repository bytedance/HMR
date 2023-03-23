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
from tqdm import tqdm
import multiprocessing
from subprocess import Popen, PIPE
from functools import partialmethod

# parse MSMS output
def read_msms(mesh_prefix):
    assert os.path.isfile(mesh_prefix + '.vert')
    assert os.path.isfile(mesh_prefix + '.face')
    
    # vertices
    with open(mesh_prefix + '.vert') as f:
        vert_data = f.read().rstrip().split('\n')
    num_verts = int(vert_data[2].split()[0])
    assert num_verts == len(vert_data) - 3
    vertices = []
    vnormals = []
    for idx in range(3, len(vert_data)):
        ifields = vert_data[idx].split()
        assert len(ifields) == 10
        vertices.append(ifields[:3])
        vnormals.append(ifields[3:6])
        full_id = ifields[-1].split('_')
        assert len(full_id) == 6
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
    vnormals = np.array(vnormals, dtype=float)
    faces = np.array(faces, dtype=int) - 1 # convert to zero-based indexing
    assert np.amin(faces) == 0
    assert np.amax(faces) < num_verts
    
    return vertices, vnormals, faces


def check_mesh_continuity(verts, faces):
    mesh = pymesh.form_mesh(verts, faces)
    mesh.enable_connectivity()
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
    if len(groups) > 1 and len(groups[1])/len(groups[0]) > 0.1:
        return False
    
    return True


# use MSMS to compute molecular solvent excluded surface
def compute_ses(data_root, pair_name, probe_radius, msms_bin):
    # specify IO dir
    data_dir = os.path.join(data_root, pair_name)

    lig_xyzrn_fpath = os.path.join(data_dir, 'ligand.xyzrn')
    rec_xyzrn_fpath = os.path.join(data_dir, 'receptor.xyzrn')
    if not (os.path.isfile(lig_xyzrn_fpath) and \
            os.path.isfile(rec_xyzrn_fpath)):
        return

    # ligand MSMS
    lig_mesh_prefix = os.path.join(data_dir, 'ligand')
    lig_args = [msms_bin, '-if', lig_xyzrn_fpath, '-of', lig_mesh_prefix, \
                '-probe_radius', str(probe_radius), '-density', '1.0']
    lig_proc = Popen(lig_args, stdout=PIPE, stderr=PIPE)
    _, lig_stderr = lig_proc.communicate()
    # skip if MSMS failed
    lig_errmsg = lig_stderr.decode('utf-8')
    if 'ERROR' in lig_errmsg:
        print(f'skip {pair_name} ligand', flush=True)
        return
    
    # receptor MSMS
    rec_mesh_prefix = os.path.join(data_dir, 'receptor')
    rec_args = [msms_bin, '-if', rec_xyzrn_fpath, '-of', rec_mesh_prefix, \
                '-probe_radius', str(probe_radius), '-density', '1.0']
    rec_proc = Popen(rec_args, stdout=PIPE, stderr=PIPE)
    _, rec_stderr = rec_proc.communicate()
    # skip if MSMS failed
    rec_errmsg = rec_stderr.decode('utf-8')
    if 'ERROR' in rec_errmsg:
        print(f'skip {pair_name} receptor', flush=True)
        return
    
    if not (os.path.isfile(lig_mesh_prefix+'.vert') and \
            os.path.isfile(lig_mesh_prefix+'.face') and \
            os.path.isfile(rec_mesh_prefix+'.vert') and \
            os.path.isfile(rec_mesh_prefix+'.face')):
        print(f'skip {pair_name} due to missing MSMS output', flush=True)
        return

    lig_verts, _, lig_faces = read_msms(lig_mesh_prefix)
    rec_verts, _, rec_faces = read_msms(rec_mesh_prefix)
    if not (check_mesh_continuity(lig_verts, lig_faces) and \
            check_mesh_continuity(rec_verts, rec_faces) and \
            min(len(lig_verts), len(rec_verts)) > 1000):
        return
    
    # save ligand surface
    lig_mesh_fpath = os.path.join(data_dir, 'ligand_msms.npz')
    np.savez(lig_mesh_fpath, verts=lig_verts, faces=lig_faces)
    
    # save receptor surface
    rec_mesh_fpath = os.path.join(data_dir, 'receptor_msms.npz')
    np.savez(rec_mesh_fpath, verts=rec_verts, faces=rec_faces)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-radius', type=float, default=1.5)
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    parser.add_argument('--mute-tqdm', action='store_true')
    args = parser.parse_args()
    print(args)

    # optionally mute tqdm
    if args.mute_tqdm:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # MSMS
    msms_bin = '/opt/tiger/MSMS/msms.x86_64Linux2.2.6.1'
    assert os.path.exists(msms_bin)

    # RCSB
    rcsb_mesh_dir = './RCSB_mesh/'
    assert os.path.exists(rcsb_mesh_dir)

    # RCSB timer
    start = time.time()

    if not args.serial:
        pool = multiprocessing.Pool(processes=args.j)
        pool_args = [(rcsb_mesh_dir, pair_name, args.probe_radius, msms_bin) for pair_name in os.listdir(rcsb_mesh_dir)]
        pool.starmap(compute_ses, tqdm(pool_args), chunksize=10)
        pool.terminate()
        print('All processes successfully finished')
    else:
        for pair_name in tqdm(os.listdir(rcsb_mesh_dir)):
            compute_ses(rcsb_mesh_dir, pair_name, args.probe_radius, msms_bin)
    
    print(f'RCSB step5 elapsed time: {(time.time()-start):.1f}s\n')


