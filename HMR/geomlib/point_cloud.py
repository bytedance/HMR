"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import numpy as np
from scipy.linalg import svd, det
from sklearn.neighbors import BallTree


def Kabsch(Y1, Y2, normal1=None, normal2=None):
    # fix Y2 and align Y1 to Y2
    Y1_mean = Y1.mean(axis=0, keepdims=True)
    Y2_mean = Y2.mean(axis=0, keepdims=True)

    if normal1 is not None and normal2 is not None:
        A = np.vstack([Y1 - Y1_mean, normal1]).T @ np.vstack([Y2 - Y2_mean, normal2])
    else:
        A = (Y1 - Y1_mean).T @ (Y2 - Y2_mean)
    U, _, Vt = svd(A)
    d = np.sign(det(U @ Vt))

    corr_mat = np.diag(np.asarray([1, 1, d]))

    rot_mat = U @ corr_mat @ Vt
    transl_vec = Y2_mean - Y1_mean @ rot_mat  # (1,3)

    return rot_mat, transl_vec


def compute_RMSD(pcd1, pcd2):
    return np.sqrt(np.mean(np.sum((pcd1 - pcd2) ** 2, axis=1)))


def compute_CRMSD(pcd1, pcd2):
    rot_mat, transl_vec = Kabsch(pcd1, pcd2)
    pcd1_align = pcd1 @ rot_mat + transl_vec
    return np.sqrt(np.mean(np.sum((pcd1_align - pcd2) ** 2, axis=1)))


def compute_IRMSD(lig, rec, lig_gt, rec_gt, iface_cutoff=8.0):
    # extract interface
    bt1 = BallTree(lig_gt)
    dist1, _ = bt1.query(rec_gt, k=1)
    rec_iface_gt = rec_gt[np.where(dist1 < iface_cutoff)[0]]
    rec_iface = rec[np.where(dist1 < iface_cutoff)[0]]
    bt2 = BallTree(rec_gt)
    dist2, _ = bt2.query(lig_gt, k=1)
    lig_iface_gt = lig_gt[np.where(dist2 < iface_cutoff)[0]]
    lig_iface = lig[np.where(dist2 < iface_cutoff)[0]]
    # alignment
    return compute_CRMSD(np.vstack([lig_iface, rec_iface]), np.vstack([lig_iface_gt, rec_iface_gt]))


def compute_Fnat(lig, rec, lig_gt, rec_gt, iface_cutoff=5.0):
    bt1_gt = BallTree(lig_gt)
    dist1_gt, _ = bt1_gt.query(rec_gt, k=1)
    rec_iface_ind_gt = np.where(dist1_gt < iface_cutoff)[0]
    lig_iface_ind_gt = bt1_gt.query_radius(rec_gt[rec_iface_ind_gt], iface_cutoff)
    iface_pair_gt = set([(i,j) for n, i in enumerate(rec_iface_ind_gt) for j in lig_iface_ind_gt[n]])

    bt1 = BallTree(lig)
    dist1, _ = bt1.query(rec, k=1)
    rec_iface_ind = np.where(dist1 < iface_cutoff)[0]
    lig_iface_ind = bt1.query_radius(rec[rec_iface_ind], iface_cutoff)
    iface_pair = set([(i,j) for n, i in enumerate(rec_iface_ind) for j in lig_iface_ind[n]])

    return len(iface_pair & iface_pair_gt) / len(iface_pair_gt)


def compute_DockQ(Fnat, IRMSD, LRMSD):
    return ( Fnat + 1.0 / (1.0 + (IRMSD/1.5)**2) + 1.0 / (1.0 + (LRMSD/8.5)**2) )/ 3.0

