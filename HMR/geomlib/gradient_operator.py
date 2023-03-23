""""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import scipy
import pymesh
import numpy as np

# least squares fit of directional derivatives
def LSDD(verts, faces, basis1, basis2):
    num_verts = len(verts)
    mesh = pymesh.form_mesh(verts, faces)
    mesh.enable_connectivity()
    # solve the least squares problem for each vertex
    row_ids = []
    col_ids = []
    data = []
    for ivert in range(num_verts):
        basis_x = basis1[ivert]
        basis_y = basis2[ivert]
        nbr_ids = mesh.get_vertex_adjacent_vertices(ivert)
        edges_3d = verts[nbr_ids] - verts[ivert]
        edges_tan_x = np.einsum('kd,d->k', edges_3d, basis_x).reshape(-1, 1)
        edges_tan_y = np.einsum('kd,d->k', edges_3d, basis_y).reshape(-1, 1)
        A = np.concatenate((edges_tan_x, edges_tan_y), axis=-1)
        A_pinv = np.linalg.inv(A.T @ A + 1E-6 * np.eye(2)) @ A.T
        B = np.zeros((len(nbr_ids), len(nbr_ids)+1))
        B[:, 0] = -1
        cols = np.arange(len(nbr_ids)) + 1
        rows = np.arange(len(nbr_ids))
        B[rows, cols] = 1
        D = A_pinv @ B
        # self-interaction
        row_ids.append(ivert)
        col_ids.append(ivert)
        data.append(D[0, 0] + 1j*D[1, 0])
        # neighbors
        for idx in range(len(nbr_ids)):
            row_ids.append(ivert)
            col_ids.append(nbr_ids[idx])
            data.append(D[0, idx+1] + 1j*D[1, idx+1])
    # construct complex sparse matrix
    row_ids = np.array(row_ids)
    col_ids = np.array(col_ids)
    data= np.array(data)
    grad_op = scipy.sparse.coo_matrix(
        (data, (row_ids, col_ids)), shape=(num_verts, num_verts), dtype=np.csingle
    ).tocsc()

    return grad_op


def compute_gradient_operator(verts, faces, vnormals):
    # initial local coordinate system
    x_unit_vec = np.array([1, 0, 0])
    y_unit_vec = np.array([0, 1, 0])
    basis1_xy = np.tile(x_unit_vec, (len(verts), 1))
    basis1_xy[np.where(np.abs(np.einsum('nd,d->n', vnormals, x_unit_vec)) > 0.95)] = y_unit_vec
    
    # Gram–Schmidt process
    basis1_xy = basis1_xy - vnormals * np.einsum('nd,nd->n', basis1_xy, vnormals).reshape(-1, 1)
    basis1_xy /= np.linalg.norm(basis1_xy, axis=-1, keepdims=True)
    basis2_xy = np.cross(vnormals, basis1_xy)
    assert np.all(np.abs(np.einsum('nd,nd->n', vnormals, basis1_xy)) < 1E-4)
    assert np.all(np.abs(np.linalg.norm(basis1_xy, axis=-1) - 1) < 1E-4)
    assert np.all(np.abs(np.linalg.norm(basis2_xy, axis=-1) - 1) < 1E-4)

    # least squares fit
    grad_op = LSDD(verts, faces, basis1_xy, basis2_xy)

    basis = np.concatenate((basis1_xy, basis2_xy), axis=-1)
    
    return grad_op, basis
    

