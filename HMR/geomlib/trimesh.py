"""
This file contains work from https://github.com/RobinMagnet/pyFM. License info:

MIT License

Copyright (c) 2020 Robin Magnet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the Software), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, andor sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


----
This file may have been modifed by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, diags

class TriMesh(object):
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces
        self.normals = None
        self.face_areas = None
        self.per_vert_areas = None
        self.area = None

        self.stiffness = None

        self.eigen_vals = None
        self.eigen_vecs = None
        self.mass = None

        self.feats = None

    def LB_decomposition(self, k=None):
        # stiffness matrix
        if self.stiffness is None:
            self.stiffness = self.compute_stiffness_matrix()
        # mass matrix
        if self.mass is None:
            self.mass = self.compute_fem_mass_matrix()

        if k is None:
            k = self.verts.shape[0] - 1
        elif k <= 0:
            return

        # compute Laplace-Beltrami basis (eigen-vectors are stored column-wise)
        self.eigen_vals, self.eigen_vecs = eigsh(A=self.stiffness, k=k, M=self.mass, sigma=-0.01)

        self.eigen_vals[0] = 0

    def compute_stiffness_matrix(self):
        verts = self.verts
        faces = self.faces
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]

        e1 = v3 - v2
        e2 = v1 - v3
        e3 = v2 - v1

        # compute cosine alpha/beta
        L1 = np.linalg.norm(e1, axis=1)
        L2 = np.linalg.norm(e2, axis=1)
        L3 = np.linalg.norm(e3, axis=1)
        cos1 = np.einsum('ij,ij->i', -e2, e3) / (L2 * L3)
        cos2 = np.einsum('ij,ij->i', e1, -e3) / (L1 * L3)
        cos3 = np.einsum('ij,ij->i', -e1, e2) / (L1 * L2)

        # cot(arccos(x)) = x/sqrt(1-x^2)
        I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        S = np.concatenate([cos3, cos1, cos2])
        S = 0.5 * S / np.sqrt(1 - S**2)

        In = np.concatenate([I, J, I, J]) 
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([-S, -S, S, S])

        N = verts.shape[0]
        stiffness = coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()

        return stiffness

    def compute_fem_mass_matrix(self):
        verts = self.verts
        faces = self.faces
        # compute face areas
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]
        face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)

        I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        S = np.concatenate([face_areas, face_areas, face_areas])

        In = np.concatenate([I, J, I])
        Jn = np.concatenate([J, I, I])
        Sn = 1. / 12. * np.concatenate([S, S, 2*S])

        N = verts.shape[0]
        mass = coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()

        return mass
        
    def compute_normals(self):
        v1 = self.verts[self.faces[:, 0]]
        v2 = self.verts[self.faces[:, 1]]
        v3 = self.verts[self.faces[:, 2]]

        normals = np.cross(v2-v1, v3-v1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        return normals

    def grad(self, f, normalize=False):
        v1 = self.verts[self.faces[:,0]]  # (m,3)
        v2 = self.verts[self.faces[:,1]]  # (m,3)
        v3 = self.verts[self.faces[:,2]]  # (m,3)

        f1 = f[self.faces[:,0]]  # (m,p) or (m,)
        f2 = f[self.faces[:,1]]  # (m,p) or (m,)
        f3 = f[self.faces[:,2]]  # (m,p) or (m,)

        if self.face_areas is None:
            self.face_areas = self.compute_face_areas()

        if self.normals is None:
            self.normals = self.compute_normals()

        grad2 = np.cross(self.normals, v1-v3)/(2*self.face_areas[:,None])  # (m,3)
        grad3 = np.cross(self.normals, v2-v1)/(2*self.face_areas[:,None])  # (m,3)

        if f.ndim == 1:
            gradf = (f2-f1)[:,None] * grad2 + (f3-f1)[:,None] * grad3  # (m,3)
        else:
            gradf = (f2-f1)[:,:,None] * grad2[:,None,:] + (f3-f1)[:,:,None] * grad3[:,None,:]  # (m,3)

        if normalize:
            gradf /= np.linalg.norm(gradf,axis=1,keepdims=True)

        return gradf

    def compute_face_areas(self):
        v1 = self.verts[self.faces[:, 0]]
        v2 = self.verts[self.faces[:, 1]]
        v3 = self.verts[self.faces[:, 2]]

        face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1),axis=1)
        return face_areas

    def compute_per_vert_areas(self):
        n_vertices = self.verts.shape[0]

        if self.face_areas is None:
            self.face_areas = self.compute_face_areas()

        I = np.concatenate([self.faces[:,0], self.faces[:,1], self.faces[:,2]])
        J = np.zeros_like(I)
        
        V = np.tile(self.face_areas / 3, 3)

        per_vert_areas = np.array(coo_matrix((V, (I, J)), shape=(n_vertices, 1)).todense()).flatten()

        return per_vert_areas

    def orientation_op(self, gradf, normalize=False):
        if normalize:
            gradf /= np.linalg.norm(gradf, axis=1, keepdims=True)

        n_vertices = self.verts.shape[0]
        
        v1 = self.verts[self.faces[:,0]]  # (n_f,3)
        v2 = self.verts[self.faces[:,1]]  # (n_f,3)
        v3 = self.verts[self.faces[:,2]]  # (n_f,3)

        # compute normals
        if self.normals is None:
            self.normals = self.compute_normals()
        # computer per vertex area
        if self.per_vert_areas is None:
            self.per_vert_area = self.compute_per_vert_areas()

        # Define (normalized) gradient directions for each barycentric coordinate on each face
        # Remove normalization since it will disappear later on after multiplcation
        Jc1 = np.cross(self.normals, v3-v2)/2
        Jc2 = np.cross(self.normals, v1-v3)/2
        Jc3 = np.cross(self.normals, v2-v1)/2

        # Rotate the gradient field
        rot_field = np.cross(self.normals, gradf)  # (n_f,3)

        I = np.concatenate([self.faces[:,0], self.faces[:,1], self.faces[:,2]])
        J = np.concatenate([self.faces[:,1], self.faces[:,2], self.faces[:,0]])

        # Compute pairwise dot products between the gradient directions
        # and the gradient field
        Sij = 1/3*np.concatenate([np.einsum('ij,ij->i', Jc2, rot_field),
                                np.einsum('ij,ij->i', Jc3, rot_field),
                                np.einsum('ij,ij->i', Jc1, rot_field)])

        Sji = 1/3*np.concatenate([np.einsum('ij,ij->i', Jc1, rot_field),
                                np.einsum('ij,ij->i', Jc2, rot_field),
                                np.einsum('ij,ij->i', Jc3, rot_field)])

        In = np.concatenate([I, J, I, J])
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([Sij, Sji, -Sij, -Sji])

        W = coo_matrix((Sn, (In, Jn)), shape=(n_vertices, n_vertices)).tocsc()
        inv_area = diags(1/self.per_vert_area, shape=(n_vertices, n_vertices), format='csc')

        return inv_area @ W


