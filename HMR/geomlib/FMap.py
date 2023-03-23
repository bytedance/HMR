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
from scipy.linalg import svd
from sklearn.neighbors import KDTree
from scipy.optimize import fmin_l_bfgs_b


class FMap:
    def __init__(self, lig_mesh, rec_mesh):

        self.lig_mesh = lig_mesh
        self.rec_mesh = rec_mesh

        self.lig_feats = lig_mesh.feats
        self.rec_feats = rec_mesh.feats

        self.lig_k = len(lig_mesh.eigen_vals)
        self.rec_k = len(rec_mesh.eigen_vals)

    def fit(self, w_descr, w_lap, w_orient, orient_reversing):

        # spectral features
        lig_feats_spec = (self.lig_mesh.eigen_vecs.T @ self.lig_mesh.mass) @ self.lig_feats
        rec_feats_spec = (self.rec_mesh.eigen_vecs.T @ self.rec_mesh.mass) @ self.rec_feats

        # compute the squared differences between eigenvalues for LB commutativity
        ev_sqdiff = np.square(self.lig_mesh.eigen_vals[None, :] - self.rec_mesh.eigen_vals[:, None])

        # compute orientation-preserving operators
        if w_orient > 0:
            orient_op = self.compute_orientation_op(reversing=orient_reversing)
            # rescale weight
            C0 = np.eye(self.rec_k, self.lig_k)
            eval_native = energy_func_std(C0, w_descr, w_lap, 0, lig_feats_spec, rec_feats_spec, orient_op, ev_sqdiff)
            eval_orient = 0
            for (op1, op2) in orient_op:
               eval_orient += 0.5 * np.square(C0 @ op1 - op2 @ C0).sum()
            w_orient *= eval_native / eval_orient
        else:
            orient_op = None

        # arguments for the optimization problem
        args = (w_descr, w_lap, w_orient, \
                lig_feats_spec, rec_feats_spec, orient_op, ev_sqdiff)

        # initial guess
        x0 = self.get_x0()

        # optimization
        res = fmin_l_bfgs_b(energy_func_std, x0.ravel(), fprime=grad_energy_std, args=args)
        self.FM_classic = res[0].reshape((self.rec_k, self.lig_k))

    def get_x0(self):

        lig_mesh_area = self.lig_mesh.compute_face_areas().sum()
        rec_mesh_area = self.rec_mesh.compute_face_areas().sum()

        x0 = np.zeros((self.rec_k, self.lig_k))

        ev_sign = np.sign(self.lig_mesh.eigen_vecs[0, 0] * self.rec_mesh.eigen_vecs[0, 0])
        area_ratio = np.sqrt(rec_mesh_area /lig_mesh_area)

        x0[:, 0] = np.zeros(self.rec_k)
        x0[0, 0] = ev_sign * area_ratio

        return x0

    def FM_to_p2p(self, FM_12, mode='delta'):
        
        if mode == 'delta':
            lig_emb = self.lig_mesh.eigen_vecs @ FM_12.T
            rec_emb = self.rec_mesh.eigen_vecs
        elif mode == 'Green':
            lig_emb = (self.lig_mesh.eigen_vecs[:, 1:] / self.lig_mesh.eigen_vals[1:].reshape(1, -1)) @ FM_12[1:, 1:].T
            rec_emb = self.rec_mesh.eigen_vecs[:, 1:] / self.rec_mesh.eigen_vals[1:].reshape(1, -1)
        else:
            raise Exception('UnkownMode')

        kdt = KDTree(lig_emb)
        p2p_21 = kdt.query(rec_emb, k=1, return_distance=False)
        
        return p2p_21.flatten()

    def get_p2p(self, FM_type='classic', p2p_mode='delta'):
        if FM_type=='classic':
            return self.FM_to_p2p(self.FM_classic, mode=p2p_mode)
        elif FM_type=='icp':
            return self.FM_to_p2p(self.FM_icp, mode=p2p_mode)
        else:
            raise Exception('UnknownFMtype')

    def p2p_to_FM(self, p2p_21):
        return (self.rec_mesh.eigen_vecs.T @ self.rec_mesh.mass) @ self.lig_mesh.eigen_vecs[p2p_21]

    def icp_refine(self, nit=10, tol=1e-5, p2p_mode='delta'):

        FM_12_curr = self.FM_classic.copy()
    
        for _ in range(nit):
            p2p_21 = self.FM_to_p2p(FM_12_curr, mode=p2p_mode)
            FM_12_icp = self.p2p_to_FM(p2p_21)
            U, _, VT = svd(FM_12_icp)
            FM_12_icp = U @ np.eye(self.rec_k, self.lig_k) @ VT

            if np.max(np.abs(FM_12_curr - FM_12_icp)) <= tol:
                break

            FM_12_curr = FM_12_icp.copy()

        self.FM_icp = FM_12_curr

    def compute_orientation_op(self, reversing=False):

        # inverse transform
        lig_eigen_vecs_inv = self.lig_mesh.eigen_vecs.T @ self.lig_mesh.mass
        rec_eigen_vecs_inv = self.rec_mesh.eigen_vecs.T @ self.rec_mesh.mass

        # compute the gradient of each descriptor
        num_feats = self.lig_feats.shape[1]
        lig_grad = [self.lig_mesh.grad(self.lig_feats[:, i]) for i in range(num_feats)]
        rec_grad = [self.rec_mesh.grad(self.rec_feats[:, i]) for i in range(num_feats)]

        # compute the operator in reduced basis
        lig_op = [lig_eigen_vecs_inv @ (self.lig_mesh.orientation_op(gradf) @ self.lig_mesh.eigen_vecs)
                  for gradf in lig_grad]

        if reversing:
            rec_op = [-rec_eigen_vecs_inv @ (self.rec_mesh.orientation_op(gradg) @ self.rec_mesh.eigen_vecs)
                      for gradg in rec_grad]
        else:
            rec_op = [rec_eigen_vecs_inv @ (self.rec_mesh.orientation_op(gradg) @ self.rec_mesh.eigen_vecs)
                      for gradg in rec_grad]

        list_op = list(zip(lig_op, rec_op))

        return list_op


def energy_func_std(C, descr_mu, lap_mu, orient_mu, lig_feat_spec, rec_feat_spec, orient_op, ev_sqdiff):
    lig_k = lig_feat_spec.shape[0]
    rec_k = rec_feat_spec.shape[0]
    C = C.reshape((rec_k, lig_k))

    energy = 0

    if descr_mu > 0:
        energy += descr_mu * 0.5 * np.square(C @ lig_feat_spec - rec_feat_spec).sum()
    if lap_mu > 0:
        energy += lap_mu * 0.5 * (np.square(C) * ev_sqdiff).sum()

    if orient_mu > 0:
        energy_orient = 0
        for (op1, op2) in orient_op:
            energy_orient += 0.5 * np.square(C @ op1 - op2 @ C).sum()
        energy += orient_mu * energy_orient

    return energy


def grad_energy_std(C, descr_mu, lap_mu, orient_mu, lig_feat_spec, rec_feat_spec, orient_op, ev_sqdiff):
    lig_k = lig_feat_spec.shape[0]
    rec_k = rec_feat_spec.shape[0]
    C = C.reshape((rec_k, lig_k))

    gradient = np.zeros_like(C)

    if descr_mu > 0:
        gradient += descr_mu * (C @ lig_feat_spec - rec_feat_spec) @ lig_feat_spec.T

    if lap_mu > 0:
        gradient += lap_mu * C * ev_sqdiff

    if orient_mu > 0:
        grad_orient = 0
        for (op1, op2) in orient_op:
            grad_orient += op2.T @ (op2 @ C - C @ op1) - (op2 @ C - C @ op1) @ op1.T
        gradient += orient_mu * grad_orient

    gradient[:,0] = 0

    return gradient.reshape(-1)


