"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import pymesh
import numpy as np
from tqdm import tqdm
from main import get_config
from sklearn import decomposition
from pyquaternion import Quaternion
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from HMR.geomlib.FMap import FMap
from HMR.geomlib.trimesh import TriMesh
from HMR.geomlib.point_cloud import Kabsch, compute_RMSD, compute_CRMSD, compute_IRMSD, \
                                    compute_Fnat, compute_DockQ

def get_candidate_patches(verts, faces, iface_prob, iface_threshold):
    if iface_threshold < 1:
        patch_cdd_ids = np.zeros(1)
        while len(patch_cdd_ids) < 200:
            patch_cdd_ids = np.where(iface_prob > iface_threshold)[0]
            iface_threshold -= 0.1
    else:
        patch_cdd_ids = np.where(iface_prob.astype(int))[0]
        assert len(patch_cdd_ids) > 100
    
    idx_map1 = np.empty(len(verts))
    idx_map1.fill(np.NaN)
    idx_map1[patch_cdd_ids] = np.arange(len(patch_cdd_ids))
    patch_cdd_verts = verts[patch_cdd_ids]
    patch_cdd_faces = idx_map1[faces]
    patch_cdd_faces = patch_cdd_faces[~np.isnan(patch_cdd_faces).any(axis=-1)].astype(int)
    patch_cdd_mesh = pymesh.form_mesh(patch_cdd_verts, patch_cdd_faces)
    patch_cdd_mesh.enable_connectivity()
    
    # extract patch using BFS
    visited = np.zeros(len(patch_cdd_verts)).astype(bool)
    groups = []
    for ivert in range(len(patch_cdd_verts)):
        if visited[ivert]:
            continue
        old_visited = visited.copy()
        queue = [ivert]
        visited[ivert] = True
        while queue:
            curr = queue.pop(0)
            for nbr in patch_cdd_mesh.get_vertex_adjacent_vertices(curr):
                if not visited[nbr]:
                    queue.append(nbr)
                    visited[nbr] = True
        groups.append(np.where(np.logical_xor(old_visited, visited))[0])
    groups = sorted(groups, key=lambda x:len(x), reverse=True)
    assert sum(len(ig) for ig in groups) == sum(visited) == len(patch_cdd_verts)

    # reindex
    groups_out = [patch_cdd_ids[ig] for ig in groups if len(ig) > 50]
    if len(groups_out) == 0:
        groups_out = get_candidate_patches(verts, faces, iface_prob, iface_threshold)
    assert len(groups_out) > 0
    
    return groups_out


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def FMap_docking(fpath, out_dir, config):
    # load data
    data = np.load(fpath)
    lig_verts = data['lig_verts']
    lig_faces = data['lig_faces'].astype(int)
    lig_pred_prob = sigmoid(data['lig_bsp'].squeeze())
    lig_gt_label = data['lig_iface_label'].astype(int)
    lig_h = data['lig_h']
    lig_alphaC = data['lig_atom_info'][:, -1].astype(int)
    lig_residue_coords = data['lig_atom_info'][np.where(lig_alphaC)[0], :3]
    rec_verts = data['rec_verts']
    rec_faces = data['rec_faces'].astype(int)
    rec_pred_prob = sigmoid(data['rec_bsp'].squeeze())
    rec_gt_label = data['rec_iface_label'].astype(int)
    rec_h = data['rec_h']
    rec_alphaC = data['rec_atom_info'][:, -1].astype(int)
    rec_residue_coords = data['rec_atom_info'][np.where(rec_alphaC)[0], :3]

    if True:
        # use predicted binding site patches
        iface_threshold = 0.8
        lig_cdd_patches = get_candidate_patches(lig_verts, lig_faces, lig_pred_prob, iface_threshold)
        rec_cdd_patches = get_candidate_patches(rec_verts, rec_faces, rec_pred_prob, iface_threshold)
    else:
        # use ground truth interface
        iface_threshold = 1
        lig_cdd_patches = get_candidate_patches(lig_verts, lig_faces, lig_gt_label, iface_threshold)[:1]
        rec_cdd_patches = get_candidate_patches(rec_verts, rec_faces, rec_gt_label, iface_threshold)[:1]

    cdd_pairs = []
    for ipatch in lig_cdd_patches:
        for jpatch in rec_cdd_patches:
            cdd_pairs.append((ipatch.astype(int), jpatch.astype(int)))

    num_k = 30
    num_sigs = 100
    crmsds, lrmsds, irmsds = [], [], []
    fnats, dockqs = [], []
    for idx, (lig_patch_ids, rec_patch_ids) in enumerate(cdd_pairs):
        lig_idx_map = np.empty(len(lig_verts))
        lig_idx_map.fill(np.NaN)
        lig_idx_map[lig_patch_ids] = np.arange(len(lig_patch_ids))
        lig_patch_verts = lig_verts[lig_patch_ids]
        lig_patch_faces = lig_idx_map[lig_faces]
        lig_patch_faces = lig_patch_faces[~np.isnan(lig_patch_faces).any(axis=-1)].astype(int)
        lig_patch = TriMesh(lig_patch_verts, lig_patch_faces)
        lig_patch.LB_decomposition(num_k)
        lig_feats = data['lig_h'][lig_patch_ids]
        lig_patch.feats = lig_feats

        rec_idx_map = np.empty(len(rec_verts))
        rec_idx_map.fill(np.NaN)
        rec_idx_map[rec_patch_ids] = np.arange(len(rec_patch_ids))
        rec_patch_verts = rec_verts[rec_patch_ids]
        rec_patch_faces = rec_idx_map[rec_faces]
        rec_patch_faces = rec_patch_faces[~np.isnan(rec_patch_faces).any(axis=-1)].astype(int)
        rec_patch = TriMesh(rec_patch_verts, rec_patch_faces)
        rec_patch.LB_decomposition(num_k)
        rec_feats = data['rec_h'][rec_patch_ids]
        rec_patch.feats = rec_feats
        
        model = FMap(lig_patch, rec_patch)
        model.fit(w_descr=1, w_lap=0.1, w_orient=1, orient_reversing=True)

        if True:
            # directly apply functional map
            C_lig2rec = np.asarray(model.FM_classic)
            p2p_rec2lig = model.get_p2p(FM_type='classic', p2p_mode='delta')
        else:
            # iterative refinement
            model.icp_refine(p2p_mode='Green')
            C_lig2rec = np.asarray(model.FM_icp)
            p2p_rec2lig = model.get_p2p(FM_type='icp', p2p_mode='Green')
        
        # random rotation
        lig_rand_rot = Quaternion.random().rotation_matrix
        lig_rand_trans = np.random.uniform(-50, 50, (3,))
        lig_verts_rot = lig_verts @ lig_rand_rot + lig_rand_trans
        lig_patch_verts_rot = lig_patch_verts @ lig_rand_rot + lig_rand_trans
        lig_residue_coords_rot = lig_residue_coords @ lig_rand_rot + lig_rand_trans

        # Kabsch
        kabsch_rot, kabsch_trans = Kabsch(lig_patch_verts_rot[p2p_rec2lig], rec_patch_verts)
        lig_verts_docked = lig_verts_rot @ kabsch_rot + kabsch_trans
        lig_residue_coords_docked = lig_residue_coords_rot @ kabsch_rot + kabsch_trans
        
        if True:
            # flip reversed cases
            bt = BallTree(rec_residue_coords)
            dist1, _ = bt.query(lig_residue_coords_docked, k=1)
            overlap1 = len(np.where(dist1 < 3)[0])

            x = StandardScaler().fit_transform(rec_patch_verts)
            pca = decomposition.PCA(n_components=3)
            pca.fit(x)
            rot_axis = list(pca.components_[0])
            rot = Quaternion(axis=rot_axis, angle=np.pi).rotation_matrix
            rec_patch_verts_center = np.mean(rec_patch_verts, axis=0)
            lig_residue_coords_docked_flip = (lig_residue_coords_docked - rec_patch_verts_center) @ rot + \
                                              rec_patch_verts_center
            dist2, _ = bt.query(lig_residue_coords_docked_flip, k=1)
            overlap2 = len(np.where(dist2 < 3)[0])
            if overlap1 - overlap2 > 5:
                lig_residue_coords_docked = lig_residue_coords_docked_flip
                lig_verts_docked = (lig_verts_docked - rec_patch_verts_center) @ rot + \
                                    rec_patch_verts_center
        
        # compute RMSDs
        complex_atom = np.vstack([lig_residue_coords, rec_residue_coords])
        crmsd = compute_CRMSD(np.vstack([lig_residue_coords_docked, rec_residue_coords]), complex_atom)
        lrmsd = compute_RMSD(lig_residue_coords_docked, lig_residue_coords)
        irmsd = compute_IRMSD(lig_residue_coords_docked, rec_residue_coords, lig_residue_coords, rec_residue_coords, iface_cutoff=8.0)
        
        # compute DockQ
        fnat = compute_Fnat(lig_residue_coords_docked, rec_residue_coords, \
                            lig_residue_coords, rec_residue_coords, iface_cutoff=8.0)
        dockq = compute_DockQ(fnat, irmsd, lrmsd)

        crmsds.append(crmsd)
        lrmsds.append(lrmsd)
        irmsds.append(irmsd)
        fnats.append(fnat)
        dockqs.append(dockq)

        # save prediction
        pid = fpath[fpath.rfind('/')+1:fpath.rfind('.')]
        out_file = os.path.join(out_dir, f'{pid}_pair{idx}.npz')
        np.savez(out_file, 
                          lig_verts_docked=lig_verts_docked,
                          lig_verts=lig_verts,
                          lig_faces=lig_faces,
                          lig_gt_label=lig_gt_label,
                          lig_pred_patch=lig_patch_ids,
                          lig_pred_prob=lig_pred_prob,
                          lig_h=lig_h,
                          rec_verts=rec_verts,
                          rec_faces=rec_faces,
                          rec_gt_label=rec_gt_label,
                          rec_pred_patch=rec_patch_ids,
                          rec_pred_prob=rec_pred_prob,
                          rec_h=rec_h,
                          p2p_rec2lig=p2p_rec2lig,
                          C_lig2rec=C_lig2rec,
                          crmsd=crmsd,
                          lrmsd=lrmsd,
                          irmsd=irmsd,
                          dockq=dockq
        )
    
    return (crmsds, lrmsds, irmsds, fnats, dockqs)
    

def PPDock(data_dir, out_dir, config):

    CRMSD_top1 = []
    CRMSD_top3 = []
    CRMSD_topN = []
    LRMSD_top1 = []
    LRMSD_top3 = []
    LRMSD_topN = []
    IRMSD_top1 = []
    IRMSD_top3 = []
    IRMSD_topN = []
    DockQ_top1 = []
    DockQ_top3 = []
    DockQ_topN = []
    fpath_list = [os.path.join(data_dir, fpath) for fpath in os.listdir(data_dir)]
    for fpath in tqdm(fpath_list):
        ret = FMap_docking(fpath, out_dir, config)
        if ret is None:
            continue
        crmsds, lrmsds, irmsds, _, dockqs = ret
        CRMSD_top1.append(crmsds[0])
        LRMSD_top1.append(lrmsds[0])
        IRMSD_top1.append(irmsds[0])
        DockQ_top1.append(dockqs[0])
        CRMSD_top3.append(min(crmsds[:3]))
        LRMSD_top3.append(min(lrmsds[:3]))
        IRMSD_top3.append(min(irmsds[:3]))
        DockQ_top3.append(max(dockqs[:3]))
        CRMSD_topN.append(min(crmsds))
        LRMSD_topN.append(min(lrmsds))
        IRMSD_topN.append(min(irmsds))
        DockQ_topN.append(max(dockqs))
    
    print(
        f'***** Docking results\n', 
        f'Top1:\n'
        f'\tmedian:\n',
        f'\t\tCRMSD: {np.median(CRMSD_top1):.2f}, LRMSD: {np.median(LRMSD_top1):.2f}, IRMSD: {np.median(IRMSD_top1):.2f}, DockQ: {np.median(DockQ_top1):.2f}\n',
        f'\tmean:\n',
        f'\t\tCRMSD: {np.mean(CRMSD_top1):.2f}, LRMSD: {np.mean(LRMSD_top1):.2f}, IRMSD: {np.mean(IRMSD_top1):.2f}, DockQ: {np.mean(DockQ_top1):.2f}\n',
        f'*****\n'
        f'Top3:\n'
        f'\tmedian:\n',
        f'\t\tCRMSD: {np.median(CRMSD_top3):.2f}, LRMSD: {np.median(LRMSD_top3):.2f}, IRMSD: {np.median(IRMSD_top3):.2f}, DockQ: {np.median(DockQ_top3):.2f}\n',
        f'\tmean:\n',
        f'\t\tCRMSD: {np.mean(CRMSD_top3):.2f}, LRMSD: {np.mean(LRMSD_top3):.2f}, IRMSD: {np.mean(IRMSD_top3):.2f}, DockQ: {np.mean(DockQ_top3):.2f}\n',
        f'*****\n'
        f'TopN:\n'
        f'\tmedian:\n',
        f'\t\tCRMSD: {np.median(CRMSD_topN):.2f}, LRMSD: {np.median(LRMSD_topN):.2f}, IRMSD: {np.median(IRMSD_topN):.2f}, DockQ: {np.median(DockQ_topN):.2f}\n',
        f'\tmean:\n',
        f'\t\tCRMSD: {np.mean(CRMSD_topN):.2f}, LRMSD: {np.mean(LRMSD_topN):.2f}, IRMSD: {np.mean(IRMSD_topN):.2f}, DockQ: {np.mean(DockQ_topN):.2f}\n',
        f'*****\n'
    )   



if __name__ == "__main__":
    config = get_config()

    # IO
    input_dir = os.path.join(config.out_dir, 'pred_results')
    out_dir = os.path.join(config.out_dir, 'docking_results')
    os.makedirs(out_dir, exist_ok=True)

    # generate docking conformations
    PPDock(input_dir, out_dir, config)


