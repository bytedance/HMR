"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import igl
import torch
import logging
import numpy as np
from functools import lru_cache
from pyquaternion import Quaternion
from HMR.data import DataLoaderBase
from torch.utils.data import Dataset
from sklearn.neighbors import BallTree
import HMR.geomlib.signatures as signatures
from utils import res_type_dict, hydrophob_dict
from HMR.geomlib.gradient_operator import compute_gradient_operator

class DataLoaderProteinDocking(DataLoaderBase):
    def __init__(self, config):
        super().__init__(config)

        self._init_datasets(config)
        self._init_samplers()
        self._init_loaders()

    # override pure virtual function
    def _init_datasets(self, config):
        self.train_set = DatasetProteinDocking(config, 'train')
        self.valid_set = DatasetProteinDocking(config, 'valid')
        self.test_set = DatasetProteinDocking(config, 'test')
        if config.is_master:
            msg = [f'Protein Docking task, train: {len(self.train_set)},',
                   f'val: {len(self.valid_set)}, test: {len(self.test_set)}']
            logging.info(' '.join(msg))


class CustomBatchProteinDocking:
    def __init__(self, unbatched_list):
        # ligand
        self.lig_iface_p2p = []
        self.lig_vnormals = []
        self.lig_chem_feats = []
        self.lig_geom_feats = []
        self.lig_eigs = []
        self.lig_grad_op = []
        self.lig_grad_basis = []
        self.lig_num_verts = []
        # receptor
        self.rec_iface_p2p = []
        self.rec_vnormals = []
        self.rec_chem_feats = []
        self.rec_geom_feats = []
        self.rec_eigs = []
        self.rec_grad_op = []
        self.rec_grad_basis = []
        self.rec_num_verts = []
        # file path
        self.fpath = []
        for (lig_dict, rec_dict, fpath) in unbatched_list:
            # ligand
            self.lig_iface_p2p.append(lig_dict['iface_p2p'])
            self.lig_vnormals.append(lig_dict['vnormals'])
            self.lig_chem_feats.append(lig_dict['chem_feats'])
            self.lig_geom_feats.append(lig_dict['geom_feats'])
            self.lig_eigs.append(lig_dict['eigs'])
            self.lig_grad_op.append(lig_dict['grad_op'])
            self.lig_grad_basis.append(lig_dict['grad_basis'])
            self.lig_num_verts.append(lig_dict['geom_feats'].size(0))
            # receptor
            self.rec_iface_p2p.append(rec_dict['iface_p2p'])
            self.rec_vnormals.append(rec_dict['vnormals'])
            self.rec_chem_feats.append(rec_dict['chem_feats'])
            self.rec_geom_feats.append(rec_dict['geom_feats'])
            self.rec_eigs.append(rec_dict['eigs'])
            self.rec_grad_op.append(rec_dict['grad_op'])
            self.rec_grad_basis.append(rec_dict['grad_basis'])
            self.rec_num_verts.append(rec_dict['geom_feats'].size(0))
            # file path
            self.fpath.append(fpath)
        self.lig_vnormals = torch.cat(self.lig_vnormals, dim=0)
        self.lig_chem_feats = torch.cat(self.lig_chem_feats, dim=0)
        self.lig_geom_feats = torch.cat(self.lig_geom_feats, dim=0)
        self.lig_grad_basis = torch.cat(self.lig_grad_basis, dim=0)
        self.rec_vnormals = torch.cat(self.rec_vnormals, dim=0)
        self.rec_chem_feats = torch.cat(self.rec_chem_feats, dim=0)
        self.rec_geom_feats = torch.cat(self.rec_geom_feats, dim=0)
        self.rec_grad_basis = torch.cat(self.rec_grad_basis, dim=0)
    
    def pin_memory(self):
        # pin ligand
        self.lig_iface_p2p = [t.pin_memory() for t in self.lig_iface_p2p]
        self.lig_vnormals = self.lig_vnormals.pin_memory()
        self.lig_chem_feats = self.lig_chem_feats.pin_memory()
        self.lig_geom_feats = self.lig_geom_feats.pin_memory()
        self.lig_eigs = [t.pin_memory() for t in self.lig_eigs]
        self.lig_grad_op = [t.pin_memory() for t in self.lig_grad_op]
        self.lig_grad_basis = self.lig_grad_basis.pin_memory()
         # pin receptor
        self.rec_iface_p2p = [t.pin_memory() for t in self.rec_iface_p2p]
        self.rec_vnormals = self.rec_vnormals.pin_memory()
        self.rec_chem_feats = self.rec_chem_feats.pin_memory()
        self.rec_geom_feats = self.rec_geom_feats.pin_memory()
        self.rec_eigs = [t.pin_memory() for t in self.rec_eigs]
        self.rec_grad_op = [t.pin_memory() for t in self.rec_grad_op]
        self.rec_grad_basis = self.rec_grad_basis.pin_memory()
        return self
    
    def to(self, device):
        # ligand
        self.lig_iface_p2p = [t.to(device) for t in self.lig_iface_p2p]
        self.lig_vnormals = self.lig_vnormals.to(device)
        self.lig_chem_feats = self.lig_chem_feats.to(device)
        self.lig_geom_feats = self.lig_geom_feats.to(device)
        self.lig_eigs = [t.to(device) for t in self.lig_eigs]
        self.lig_grad_op = [t.to(device) for t in self.lig_grad_op]
        self.lig_grad_basis = self.lig_grad_basis.to(device)
        # receptor
        self.rec_iface_p2p = [t.to(device) for t in self.rec_iface_p2p]
        self.rec_vnormals = self.rec_vnormals.to(device)
        self.rec_chem_feats = self.rec_chem_feats.to(device)
        self.rec_geom_feats = self.rec_geom_feats.to(device)
        self.rec_eigs = [t.to(device) for t in self.rec_eigs]
        self.rec_grad_op = [t.to(device) for t in self.rec_grad_op]
        self.rec_grad_basis = self.rec_grad_basis.to(device)
    
    def __len__(self):
        return len(self.fpath)


class DatasetProteinDocking(Dataset):
    def __init__(self, config, split):
        # dataset
        self.swap_pairs = split != 'test'
        self.iface_cutoff = config.HMR.iface_cutoff
        self.num_signatures = config.HMR.num_signatures
        self.lb_basis_cutoff = config.HMR.lb_basis_cutoff
        self.vert_nbr_atoms = config.HMR.vert_nbr_atoms
        self.gauss_curv_gdf = GaussianDistance(start=-0.1, stop=0.1, num_centers=config.HMR.num_gdf)
        self.mean_curv_gdf = GaussianDistance(start=-0.5, stop=0.5, num_centers=config.HMR.num_gdf)
        self.dist_gdf = GaussianDistance(start=0., stop=8., num_centers=config.HMR.num_gdf)
        self.angular_gdf = GaussianDistance(start=-1., stop=1., num_centers=config.HMR.num_gdf)
            
        # train-valid-test split
        if split == 'test':
            split_file = './misc/db5_all.txt'
            with open(split_file, 'r') as f:
                pdb_ids = [pid.strip() for pid in f.readlines()]
            self.fpaths = [os.path.join(config.data_dir, 'dataset_DB5', f'{pid}.npz') for pid in pdb_ids]
            assert len(self.fpaths) == 253
        else:
            split_file = f'./misc/rcsb_{split}_cv{config.cv}.txt'
            with open(split_file, 'r') as f:
                pdb_ids = set(pid.strip() for pid in f.readlines())
            rcsb_data_dir = os.path.join(config.data_dir, 'dataset_RCSB')
            fpath_list = [os.path.join(rcsb_data_dir, fname) for fname in os.listdir(rcsb_data_dir)]
            self.fpaths = []
            for fpath in fpath_list:
                pdb_id = fpath[fpath.rfind('/')+1:fpath.rfind('/')+5]
                if pdb_id in pdb_ids:
                    self.fpaths.append(fpath)          

    @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        # load data
        fpath = self.fpaths[idx]
        data = np.load(fpath, allow_pickle=True)

        # ligand data
        lig_atom_info = data['lig_atom_info']
        lig_atom_coords = lig_atom_info[:, :3]
        lig_verts = data['lig_verts']
        lig_faces = data['lig_faces'].astype(int)
        lig_num_lb_bases = int(self.lb_basis_cutoff) if self.lb_basis_cutoff > 1 \
                                                     else int(self.lb_basis_cutoff*len(lig_verts))
        lig_eigen_vals = data['lig_eigen_vals'][:lig_num_lb_bases]
        lig_eigen_vecs = data['lig_eigen_vecs'][:, :lig_num_lb_bases]
        lig_mass = data['lig_mass'].item()
        
        # receptor data
        rec_atom_info = data['rec_atom_info']
        rec_atom_coords = rec_atom_info[:, :3]
        rec_verts = data['rec_verts']
        rec_faces = data['rec_faces'].astype(int)
        rec_num_lb_bases = int(self.lb_basis_cutoff) if self.lb_basis_cutoff > 1 \
                                                  else int(self.lb_basis_cutoff*len(rec_verts))
        rec_eigen_vals = data['rec_eigen_vals'][:rec_num_lb_bases]
        rec_eigen_vecs = data['rec_eigen_vecs'][:, :rec_num_lb_bases]
        rec_mass = data['rec_mass'].item()
        
        # ground truth interface labels and point-to-point correspondence
        # ligand to receptor
        bt_rec = BallTree(rec_verts)
        dist_lig2rec, ind_lig2rec = bt_rec.query(lig_verts, k=1)
        lig_iface = np.where(dist_lig2rec < self.iface_cutoff)[0]
        map_lig2rec = np.concatenate((lig_iface.reshape(-1, 1),
                                      ind_lig2rec[lig_iface]), axis=-1)

        # receptor to ligand
        bt_lig = BallTree(lig_verts)
        dist_rec2lig, ind_rec2lig = bt_lig.query(rec_verts, k=1)
        rec_iface = np.where(dist_rec2lig < self.iface_cutoff)[0]
        map_rec2lig = np.concatenate((rec_iface.reshape(-1, 1),
                                      ind_rec2lig[rec_iface]), axis=-1)

        # random rotation + translation, which has no impact on scalar-type features
        co_rotation = Quaternion.random().rotation_matrix
        lig_atom_coords = lig_atom_coords @ co_rotation
        lig_verts = lig_verts @ co_rotation
        rec_atom_coords = rec_atom_coords @ co_rotation
        rec_verts = rec_verts @ co_rotation
        # additional operations on ligand
        lig_rot = Quaternion.random().rotation_matrix
        lig_trans = np.random.uniform(-50, 50, (3,))
        lig_atom_coords = lig_atom_coords @ lig_rot + lig_trans
        lig_verts = lig_verts @ lig_rot + lig_trans

        # chemical features 
        lig_atom_feats = []
        for atom_info in lig_atom_info:
            _, _, _, res_name, atom_type, charge, radius, is_alpha_carbon = atom_info
            # obtain residue type name from its dictionary value, kinda silly..
            residue = list(res_type_dict.keys())[list(res_type_dict.values()).index(res_name)]
            hphob = hydrophob_dict[residue]
            lig_atom_feats.append([res_name, atom_type, hphob, charge, radius, is_alpha_carbon])
        rec_atom_feats = []
        for atom_info in rec_atom_info:
            _, _, _, res_name, atom_type, charge, radius, is_alpha_carbon = atom_info
            residue = list(res_type_dict.keys())[list(res_type_dict.values()).index(res_name)]
            hphob = hydrophob_dict[residue]
            rec_atom_feats.append([res_name, atom_type, hphob, charge, radius, is_alpha_carbon])
        lig_atom_feats = np.array(lig_atom_feats, dtype=float)
        rec_atom_feats = np.array(rec_atom_feats, dtype=float)

        # vertex normals
        lig_vnormals = igl.per_vertex_normals(lig_verts, lig_faces)
        rec_vnormals = igl.per_vertex_normals(rec_verts, rec_faces)

        # ligand chemical features
        lig_atom_bt = BallTree(lig_atom_coords)
        lig_dist, lig_ind = lig_atom_bt.query(lig_verts, k=self.vert_nbr_atoms)
        lig_dist = np.vstack(lig_dist)
        lig_ind = np.vstack(lig_ind)
        lig_nbr_dist_gdf = self.dist_gdf.expand(lig_dist)
        lig_nbr_vec = lig_atom_coords[lig_ind] - lig_verts.reshape(-1, 1, 3)
        lig_nbr_angular = np.einsum('vkj,vj->vk', 
                                    lig_nbr_vec / np.linalg.norm(lig_nbr_vec, axis=-1, keepdims=True), 
                                    lig_vnormals)
        lig_nbr_angular_gdf = self.angular_gdf.expand(lig_nbr_angular)
        # (num_verts, vert_nbr_atoms, feat_dim)
        lig_chem_feats = np.concatenate((lig_atom_feats[lig_ind],
                                         lig_nbr_dist_gdf,
                                         lig_nbr_angular_gdf), axis=-1)

        # receptor chemical features
        rec_atom_bt = BallTree(rec_atom_coords)
        rec_dist, rec_ind = rec_atom_bt.query(rec_verts, k=self.vert_nbr_atoms)
        rec_dist = np.vstack(rec_dist)
        rec_ind = np.vstack(rec_ind)
        rec_nbr_dist_gdf = self.dist_gdf.expand(rec_dist)
        rec_nbr_vec = rec_atom_coords[rec_ind] - rec_verts.reshape(-1, 1, 3)
        rec_nbr_angular = np.einsum('vkj,vj->vk', 
                                    rec_nbr_vec / np.linalg.norm(rec_nbr_vec, axis=-1, keepdims=True), 
                                    rec_vnormals)
        rec_nbr_angular_gdf = self.angular_gdf.expand(rec_nbr_angular)
        # (num_verts, vert_nbr_atoms, feat_dim)
        rec_chem_feats = np.concatenate((rec_atom_feats[rec_ind],
                                         rec_nbr_dist_gdf,
                                         rec_nbr_angular_gdf), axis=-1)

        # curvatures
        _, _, lig_k1, lig_k2 = igl.principal_curvature(lig_verts, lig_faces)
        lig_gauss_curvs = lig_k1 * lig_k2
        lig_gauss_curvs_gdf = self.gauss_curv_gdf.expand(lig_gauss_curvs)
        lig_mean_curvs = 0.5 * (lig_k1 + lig_k2)
        lig_mean_curvs_gdf = self.mean_curv_gdf.expand(lig_mean_curvs)
        _, _, rec_k1, rec_k2 = igl.principal_curvature(rec_verts, rec_faces)
        rec_gauss_curvs = rec_k1 * rec_k2
        rec_gauss_curvs_gdf = self.gauss_curv_gdf.expand(rec_gauss_curvs)
        rec_mean_curvs = 0.5 * (rec_k1 + rec_k2)
        rec_mean_curvs_gdf = self.mean_curv_gdf.expand(rec_mean_curvs)

        # geometric signatures
        lig_hks = signatures.compute_HKS(lig_eigen_vecs, lig_eigen_vals, self.num_signatures)
        rec_hks = signatures.compute_HKS(rec_eigen_vecs, rec_eigen_vals, self.num_signatures)

        # assemble coordinate-free geometric features
        lig_geom_feats = np.concatenate((lig_gauss_curvs_gdf,
                                         lig_mean_curvs_gdf,
                                         lig_hks), axis=-1)
        rec_geom_feats = np.concatenate((rec_gauss_curvs_gdf,
                                         rec_mean_curvs_gdf,
                                         rec_hks), axis=-1)
        
        # Laplace-Beltrami basis
        lig_eigen_vecs_inv = lig_eigen_vecs.T @ lig_mass
        lig_eigs = np.concatenate((lig_eigen_vals.reshape(1, -1),
                                   lig_eigen_vecs,
                                   lig_eigen_vecs_inv.T), axis=0)
        rec_eigen_vecs_inv = rec_eigen_vecs.T @ rec_mass    
        rec_eigs = np.concatenate((rec_eigen_vals.reshape(1, -1),
                                   rec_eigen_vecs,
                                   rec_eigen_vecs_inv.T), axis=0)

        # compute gradient operator
        lig_grad_op, lig_grad_basis = compute_gradient_operator(lig_verts, lig_faces, lig_vnormals)
        lig_grad_op = lig_grad_op.tocoo()
        lig_grad_op_dense = np.concatenate((lig_grad_op.data.real.reshape(1, -1), 
                                            lig_grad_op.data.imag.reshape(1, -1),
                                            lig_grad_op.row.reshape(1, -1), 
                                            lig_grad_op.col.reshape(1, -1)), axis=0)
        rec_grad_op, rec_grad_basis = compute_gradient_operator(rec_verts, rec_faces, rec_vnormals)
        rec_grad_op = rec_grad_op.tocoo()
        rec_grad_op_dense = np.concatenate((rec_grad_op.data.real.reshape(1, -1), 
                                            rec_grad_op.data.imag.reshape(1, -1),
                                            rec_grad_op.row.reshape(1, -1), 
                                            rec_grad_op.col.reshape(1, -1)), axis=0)

        # ligand features
        lig_dict = {
            'iface_p2p': torch.tensor(map_lig2rec, dtype=torch.int64),
            'chem_feats': torch.tensor(lig_chem_feats, dtype=torch.float32),
            'geom_feats': torch.tensor(lig_geom_feats, dtype=torch.float32),
            'eigs': torch.tensor(lig_eigs, dtype=torch.float32),
            'vnormals': torch.tensor(lig_vnormals, dtype=torch.float32),
            'grad_op': torch.tensor(lig_grad_op_dense, dtype=torch.float32),
            'grad_basis': torch.tensor(lig_grad_basis, dtype=torch.float32),
        }

        # receptor features
        rec_dict = {
            'iface_p2p': torch.tensor(map_rec2lig, dtype=torch.int64),
            'vnormals': torch.tensor(rec_vnormals, dtype=torch.float32),
            'chem_feats': torch.tensor(rec_chem_feats, dtype=torch.float32),
            'geom_feats': torch.tensor(rec_geom_feats, dtype=torch.float32),
            'eigs': torch.tensor(rec_eigs, dtype=torch.float32),
            'vnormals': torch.tensor(rec_vnormals, dtype=torch.float32),
            'grad_op': torch.tensor(rec_grad_op_dense, dtype=torch.float32),
            'grad_basis': torch.tensor(rec_grad_basis, dtype=torch.float32),
        }

        if self.swap_pairs and np.random.binomial(n=1, p=0.5):
            return rec_dict, lig_dict, fpath
        else:
            return lig_dict, rec_dict, fpath

    @staticmethod
    def collate_wrapper(unbatched_list):
        return CustomBatchProteinDocking(unbatched_list)
    
    def __len__(self):
        return len(self.fpaths)


class GaussianDistance(object):
    def __init__(self, start, stop, num_centers):
        self.filters = np.linspace(start, stop, num_centers)
        self.var = (stop - start) / (num_centers - 1)

    def expand(self, d):
        return np.exp(-0.5 * (d[..., None] - self.filters)**2 / self.var**2)


