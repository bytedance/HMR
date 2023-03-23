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
from HMR.data import DataLoaderBase
from torch.utils.data import Dataset
from sklearn.neighbors import BallTree

class DataLoaderQM9(DataLoaderBase):
    def __init__(self, config):
        super().__init__(config)

        self._init_datasets(config)
        self._init_samplers()
        self._init_loaders()

    # override pure virtual function
    def _init_datasets(self, config):
        id_prop_file = os.path.join(config.data_dir, 'id_prop.npy')
        assert os.path.isfile(id_prop_file), 'make sure id_prop.npy is in your dataset dir'
        id_prop = np.load(id_prop_file)
        partitions = id_prop[:, 1].astype(int)
        target_dict = {
            'alpha': 2, 'gap': 3, 'homo': 4, 'lumo': 5, 'mu': 6, 'Cv': 7, 
            'G': 8, 'H': 9, 'r2': 10, 'U': 11, 'U0': 12, 'zpve': 13,
        }
        target_id = target_dict[config.target]
        # target statistics (training set)
        target_arr = id_prop[np.where(partitions == 1)[0], target_id]
        self.target_mean = np.mean(target_arr)
        self.target_mad = np.mean(np.abs(target_arr - self.target_mean))

        # train-valid-test split
        train_fpaths = []
        valid_fpaths = []
        test_fpaths = []
        for fname in os.listdir(config.data_dir):
            if not fname.startswith('gdb'):
                continue
            fpath = os.path.join(config.data_dir, fname)
            gdb_id = int(fname[fname.rfind('_')+1:fname.rfind('.')])
            target = id_prop[gdb_id, target_id]
            if partitions[gdb_id] == 1:
                train_fpaths.append((fpath, target))
            elif partitions[gdb_id] == 2:
                valid_fpaths.append((fpath, target))
            else:
                assert partitions[gdb_id] == 3
                test_fpaths.append((fpath, target))

        self.train_set = DatasetQM9(config, train_fpaths)
        self.valid_set = DatasetQM9(config, valid_fpaths)
        self.test_set = DatasetQM9(config, test_fpaths)
        if config.is_master:
            msg = [f'QM9 task, train: {len(self.train_set)},',
                   f'val: {len(self.valid_set)}, test: {len(self.test_set)}']
            logging.info(' '.join(msg))


class CustomBatchQM9:
    def __init__(self, unbatched_list):
        self.num_verts = []
        self.chem_feats = []
        self.nbr_vid = []
        self.eigs = []
        self.target = []
        base_idx = 0
        for feat_dict in unbatched_list:
            num_verts = feat_dict['num_verts']
            self.num_verts.append(num_verts)
            self.chem_feats.append(feat_dict['chem_feats'])
            self.nbr_vid.append(feat_dict['nbr_vid'] + base_idx)
            base_idx += num_verts
            self.eigs.append(feat_dict['eigs'])
            self.target.append(feat_dict['target'])
        self.chem_feats = torch.cat(self.chem_feats, dim=0)
        self.nbr_vid = torch.cat(self.nbr_vid, dim=0)
        self.target = torch.cat(self.target, dim=0)
    
    def pin_memory(self):
        self.chem_feats = self.chem_feats.pin_memory()
        self.nbr_vid = self.nbr_vid.pin_memory()
        self.eigs = [t.pin_memory() for t in self.eigs]
        self.target = self.target.pin_memory()
        return self
    
    def to(self, device):
        self.chem_feats = self.chem_feats.to(device)
        self.nbr_vid = self.nbr_vid.to(device)
        self.eigs = [t.to(device) for t in self.eigs]
        self.target = self.target.to(device)
    
    def __len__(self):
        return len(self.target)


class DatasetQM9(Dataset):
    def __init__(self, config, fpaths):
        self.cutoff_radius = config.cutoff_radius
        self.dist_gdf = GaussianDistance(start=0., stop=self.cutoff_radius+1, num_centers=config.num_gdf)
        self.angular_gdf = GaussianDistance(start=-1., stop=1., num_centers=config.num_gdf)
        self.fpaths = fpaths       

    @lru_cache(maxsize=128)
    def __getitem__(self, idx):
        # load data
        fpath, target = self.fpaths[idx]
        data = np.load(fpath, allow_pickle=True)
        verts = data['verts']
        faces = data['faces'].astype(int)
        eigen_vals = data['eigen_vals']
        eigen_vecs = data['eigen_vecs']
        mass = data['mass'].item()
        eigen_vecs_inv = eigen_vecs.T @ mass
        atom_info = data['atom_info']

        # Laplace-Beltrami basis        
        eigs = np.concatenate((eigen_vals.reshape(1, -1),
                               eigen_vecs,
                               eigen_vecs_inv.T), axis=0)

        # chemical features
        atom_coords = atom_info[:, :3]
        atom_Z = atom_info[:, -1].astype(int)
        charge_tensor = np.power(atom_Z[..., np.newaxis] / 9., np.arange(3)).reshape((atom_Z.shape[0], 1, 3))
        Z_one_hot = atom_Z.reshape(-1, 1) == np.array([1, 6, 7, 8, 9], dtype=int).reshape(1, -1)
        atom_feats = (Z_one_hot[..., np.newaxis] * charge_tensor).reshape(atom_Z.shape[0], -1)

        # project to surface
        bt = BallTree(atom_coords)
        ind, dist = bt.query_radius(verts, r=self.cutoff_radius, return_distance=True)
        ind_flat = np.concatenate(ind, axis=0)
        dist_flat = np.concatenate(dist, axis=0)
        nbr_vid = np.concatenate([[i]*len(ind[i]) for i in range(len(ind))]) # nbr-vert correspondence
        nbr_dist_gdf = self.dist_gdf.expand(np.array(dist_flat))
        nbr_atom_feats = atom_feats[ind_flat]
        nbr_vec = atom_coords[ind_flat] - verts[nbr_vid]
        vnormals = igl.per_vertex_normals(verts, faces)
        nbr_vnormals = vnormals[nbr_vid]
        nbr_angular = np.einsum('vj,vj->v', 
                                nbr_vec / np.linalg.norm(nbr_vec, axis=-1, keepdims=True), 
                                nbr_vnormals)
        nbr_angular_gdf = self.angular_gdf.expand(nbr_angular)
        chem_feats = np.concatenate((nbr_atom_feats,
                                     nbr_dist_gdf, 
                                     nbr_angular_gdf), axis=-1)

        # receptor features
        feat_dict = {
            'num_verts': len(verts),
            'chem_feats': torch.tensor(chem_feats, dtype=torch.float32),
            'nbr_vid': torch.tensor(nbr_vid, dtype=torch.int64), 
            'eigs': torch.tensor(eigs, dtype=torch.float32),
            'target': torch.tensor([target], dtype=torch.float32),
        }

        return feat_dict

    @staticmethod
    def collate_wrapper(unbatched_list):
        return CustomBatchQM9(unbatched_list)
    
    def __len__(self):
        return len(self.fpaths)


class GaussianDistance(object):
    def __init__(self, start, stop, num_centers):
        self.filters = np.linspace(start, stop, num_centers)
        self.var = (stop - start) / (num_centers - 1)

    def expand(self, d):
        return np.exp(-0.5 * (d[..., None] - self.filters)**2 / self.var**2)


