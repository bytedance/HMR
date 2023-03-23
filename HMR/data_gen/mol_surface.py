"""Functions related to molecular surface generation

-----
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import pymesh
import numpy as np
import pandas as pd

from pathlib import Path
from HMR.utils.helpers import subprocess_run


def compute_ses(xyzrn_fpath, mesh_prefix, probe_radius, density, msms_bin, print_out=True):
    """use MSMS to compute molecular solvent excluded surface.

    Args:
        xyzrn_fpath (path): input .xyzrn file for msms
        mesh_prefix (path): prefix for output .vert and .face file from msms
        probe_radius (float): args for msms. Radius of probe to generate the surface
        density (float): args for msms. vertex density per Anstrom^2
        msms_bin (path): path to MSMS binary file
        print_out (bool): if print MSMS output to screen
    
    Returns:
        0 if success, else 1
    """

    # IO 
    xyzrn_fpath = Path(xyzrn_fpath)
    pid = xyzrn_fpath.stem
    if not xyzrn_fpath.exists():
        print(f"{pid}.xyzrn not found")
        return 1
    output_dir = Path(mesh_prefix).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    mesh_prefix = str(mesh_prefix)
    msms_log = output_dir/f'{pid}_msms.log'

    # MSMS
    _, err = subprocess_run(
        [msms_bin, '-if', xyzrn_fpath, '-of', mesh_prefix,
         '-probe_radius', str(probe_radius), '-density', str(density)],
        print_out=print_out, out_log=msms_log, err_ignore=True
    )
    if 'ERROR' in err:
        print(f'{pid} error, skip', flush=True)
        return 1

    if not (os.path.isfile(mesh_prefix + '.vert') and \
            os.path.isfile(mesh_prefix + '.face')):
        print(f'skip {pid} due to missing MSMS output', flush=True)
        return 1

    verts, _, faces = read_msms(mesh_prefix)
    if not check_mesh_continuity(verts, faces):
        print(f"{pid} has uncontinuous mesh")
        return 1
    
    # save surface as npz file
    mesh_fpath = output_dir/f'{pid}.npz'
    np.savez(mesh_fpath, verts=verts, faces=faces)
    
    return 0 


def read_msms(mesh_prefix):
    """Read and parse MSMS output
    Args:
        mesh_prefix (path): path prefix for MSMS output mesh. 
            The directory should contain .vert and .face files from MSMS
    
    Returns:
        vertices (np.ndarray): (N, 3) vertex coordinates
        vnormals (np.ndarray): (N, 3) vertex normals
        faces (np.ndarray): (F, 3) vertex ids of faces
    """
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
    """Check if the mesh is continuous (only one connected component)

    Returns:
        bool: if mesh is connected
    """
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


def remove_abnormal_triangles(mesh):
    """Remove abnormal triangles (angels ~180 or ~0) in the mesh

    Returns:
        pymesh.Mesh, a new mesh with abnormal faces removed
    """
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


def check_mesh_validity(mesh, check_triangles=False):
    """Check if a mesh is valid by followin criteria
    
    1) disconnected
    2) has isolated vertex
    3) face has duplicated vertices (same vertex on a face)
    4) has triangles with angle ~0 or ~180

    Returns
        four-tuple of bool: above criteria

    """
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


def refine_mesh(mesh_msms, resolution):
    """Refine a mesh generated from MSMS

    Args:
        mesh_msms (pymesh.Mesh): raw mesh generated by MSMS
        resolution (float): resolution (edge length) of refined mesh
    
    Returns:
        pymesh.Mesh: refined mesh
    """

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
