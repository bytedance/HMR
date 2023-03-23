"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import igl
import torch
import logging
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree
from torch.utils.data import Dataset
from HMR.data import DataLoaderBase
from HMR.data_gen.chemistry import res_type_to_hphob
from HMR.geomlib import signatures


def load_data_fpaths_from_split_file(data_dir, split_fpath):
    """Load a list of data fpath available under data_dir based on the split list"""
    with open(split_fpath, 'r') as handles:
        data_list = [l.strip('\n').strip() for l in handles.readlines()]
        fpaths = [(kw, list(Path(data_dir).glob(f'{kw}*.npz'))) for kw in data_list]
        not_found = [f[0] for f in fpaths if len(f[1]) == 0]
        fpaths = [f for f_list in fpaths for f in f_list[1]]
        if len(not_found) > 0:
            print(f"{len(not_found)} data in the split file not found under data dir")
    return fpaths


class DataLoaderMasifLigand(DataLoaderBase):

    def __init__(self, config):
        super().__init__(config)

        self._init_datasets(config)
        self._init_samplers()
        self._init_loaders()

    
    def _load_split_file(self, split_fpath):
        """Load all matching data (patch) under self.data_dir in the split_fpath"""
        return load_data_fpaths_from_split_file(self.data_dir, split_fpath)
        

    def _init_datasets(self, config):

        self.data_dir = Path(config.data_dir)
        self.processed_dir = Path(config.processed_dir)

        # load train-valid-test split
        train_fpaths = []
        valid_fpaths = []
        test_fpaths = []

        if config.train_split_file:
            train_fpaths = self._load_split_file(config.train_split_file)
        if config.valid_split_file:
            valid_fpaths = self._load_split_file(config.valid_split_file)
        if config.test_split_file:
            test_fpaths = self._load_split_file(config.test_split_file)

        self.train_set = DatasetMasifLigand(config, train_fpaths)
        self.valid_set = DatasetMasifLigand(config, valid_fpaths)
        self.test_set = DatasetMasifLigand(config, test_fpaths)

        if not config.use_hvd or config.is_master:
            msg = [f'MaSIF-ligand task, train: {len(self.train_set)},',
                   f'val: {len(self.valid_set)}, test: {len(self.test_set)}']
            logging.info(' '.join(msg))


class CustomBatchMasifLigand:
    """Object to batched dataset"""

    def __init__(self, unbatched_list):
        b_labels = []
        b_chem_feats = []
        b_geom_feats = []
        b_eigs = []
        b_num_verts = []
        b_nbr_vids = []
        b_fpaths = []

        base_idx = 0
        for feat_dict in unbatched_list:

            b_labels.append(feat_dict['label'])
            b_chem_feats.append(feat_dict['chem_feats'])
            b_geom_feats.append(feat_dict['geom_feats'])
            b_eigs.append(feat_dict['eigs'])
            num_verts = feat_dict['geom_feats'].size(0)
            b_num_verts.append(num_verts)
            b_nbr_vids.append(feat_dict['nbr_vids'] + base_idx)
            base_idx += num_verts
            b_fpaths.append(feat_dict['fpath'])
        
        self.labels = torch.Tensor(b_labels).long()
        self.chem_feats = torch.cat(b_chem_feats, dim=0)
        self.geom_feats = torch.cat(b_geom_feats, dim=0)
        self.eigs = b_eigs
        self.num_verts = b_num_verts
        self.nbr_vids = torch.cat(b_nbr_vids, dim=0)
        self.fpaths = b_fpaths
    
    def pin_memory(self):
        
        self.labels = self.labels.pin_memory()
        self.chem_feats = self.chem_feats.pin_memory()
        self.geom_feats = self.geom_feats.pin_memory()
        self.eigs = [t.pin_memory() for t in self.eigs]
        self.nbr_vids = self.nbr_vids.pin_memory()
       
        return self
    
    def to(self, device):
        self.labels = self.labels.to(device)
        self.chem_feats = self.chem_feats.to(device)
        self.geom_feats = self.geom_feats.to(device)
        self.nbr_vids = self.nbr_vids.to(device)
        self.eigs = [t.to(device) for t in self.eigs]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, name):
        return getattr(self, name)


class DatasetMasifLigand(Dataset):

    def __init__(self, config, fpaths):

        # feature args
        self.use_chem_feat = config.use_chem_feat
        self.use_geom_feat = config.use_geom_feat
        self.atom_dist = True
        self.atom_angle = True
        
        self.max_eigen_val = config.max_eigen_val
        self.smoothing = config.smoothing
        self.vert_nbr_atoms = config.vert_nbr_atoms
        self.num_signatures = config.num_signatures
        
        self.gauss_curv_gdf = GaussianDistance(start=-0.1, stop=0.1, num_centers=config.num_gdf)
        self.mean_curv_gdf = GaussianDistance(start=-0.5, stop=0.5, num_centers=config.num_gdf)
        self.dist_gdf = GaussianDistance(start=0., stop=8., num_centers=config.num_gdf)
        self.angular_gdf = GaussianDistance(start=-1., stop=1., num_centers=config.num_gdf)
        
        # data dir
        self.data_dir = Path(config.data_dir)
        assert self.data_dir.exists(), f"Dataset dir {self.data_dir} not found"
        self.processed_dir = Path(config.processed_dir)
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        self.fpaths = fpaths


    def __getitem__(self, idx):
        """Neighboring atomic environment information is too large to store on HDD,
        we do it on-the-fly for the moment
        """
        
        # load data
        fpath = self.fpaths[idx]
        fname = Path(fpath).name
        processed_fpath = self.processed_dir/fname
        if not processed_fpath.exists():
            preprocess_data(
                fpath, processed_fpath,
                self.max_eigen_val, self.smoothing, self.vert_nbr_atoms, self.num_signatures
            )
        
        data = np.load(processed_fpath, allow_pickle=True)

        atom_coords = data['atom_info'][:, :3]
        atom_feats = data['atom_info'][:, 3:]
        verts = data['geom_info'][:, :3]
        vnormals = data['geom_info'][:, 3:6]
        geom_feats_in = data['geom_info'][:, 6:]
        vert_nbr_dist = data['vert_nbr_dist']
        vert_nbr_ind = data['vert_nbr_ind']

        ##############################  chem feats  ##############################
        # full chemistry features
        # coords res_type  atom_type  hphob  charge  radius  is_alphaC
        # 0~2       0         1         2      3        4        5

        chem_feats = atom_feats[:, [2, 3]]
        dist_flat = np.concatenate(vert_nbr_dist, axis=0)
        ind_flat = np.concatenate(vert_nbr_ind, axis=0)
        # vert-to-atom mapper
        nbr_vid = np.concatenate([[i] * len(vert_nbr_ind[i]) for i in range(len(vert_nbr_ind))])
        
        if self.use_chem_feat:
            chem_feats = [chem_feats[ind_flat]]
            # atom_dist
            chem_feats.append(self.dist_gdf.expand(dist_flat))
            # atom angular
            nbr_vec = atom_coords[ind_flat] - verts[nbr_vid]
            nbr_vnormals = vnormals[nbr_vid]
            nbr_angular = np.einsum('vj,vj->v', 
                                    nbr_vec / np.linalg.norm(nbr_vec, axis=-1, keepdims=True), 
                                    nbr_vnormals)
            nbr_angular_gdf = self.angular_gdf.expand(nbr_angular)
            chem_feats.append(nbr_angular_gdf)
            chem_feats = np.concatenate(chem_feats, axis=-1)
        else:
            chem_feats = np.zeros((len(nbr_vid), 1))

        ##############################  geom feats  ##############################        
        # full geom features 
        #     verts   vnormal gauss_curv  mean_curv   signature
        #     0 ~ 2   3 ~ 5   0           1            2 ~ 2 + num_signature - 1

        if self.use_geom_feat:
            # expand curvatures to gdf
            gauss_curvs = geom_feats_in[:, 0]
            gauss_curvs_gdf = self.gauss_curv_gdf.expand(gauss_curvs)
            mean_curvs = geom_feats_in[:, 1]
            mean_curvs_gdf = self.mean_curv_gdf.expand(mean_curvs)
            geom_feats = np.concatenate(
                (gauss_curvs_gdf, mean_curvs_gdf, geom_feats_in[:, 2:]), 
                axis=-1
            )
        else:
            geom_feats = np.zeros(shape=(verts.shape[0], 1))

        # Laplace-Beltrami basis
        eigs = data['eigs']

        # features
        feat_dict = {
            'label': torch.tensor(data['label'], dtype=torch.int64),
            'chem_feats': torch.tensor(chem_feats, dtype=torch.float32),
            'geom_feats': torch.tensor(geom_feats, dtype=torch.float32),
            'nbr_vids': torch.tensor(nbr_vid, dtype=torch.int64),
            'eigs': torch.tensor(eigs, dtype=torch.float32),
            'fpath': fpath
        }

        return feat_dict

    @staticmethod
    def collate_wrapper(unbatched_list):
        return CustomBatchMasifLigand(unbatched_list)
    
    def __len__(self):
        return len(self.fpaths)


def preprocess_data(
    data_fpath, processed_fpath,
    max_eigen_val, smoothing, vert_nbr_atoms, num_signatures
):
    """Preprocess data and cache on disk

    """

    # load data
    data = np.load(data_fpath, allow_pickle=True)
    label = data['label']

    atom_info = data['atom_info']
    atom_coords = atom_info[:, :3]
    verts = data['pkt_verts']
    faces = data['pkt_faces'].astype(int)

    if max_eigen_val is not None:
        ev = np.where(data['eigen_vals'] < max_eigen_val)[0]
        assert len(ev) > 1
        eigen_vals = data['eigen_vals'][ev]
        eigen_vecs = data['eigen_vecs'][:, ev]
    else:
        eigen_vals = data['eigen_vals']
        eigen_vecs = data['eigen_vecs']
    mass = data['mass'].item()
    eigen_vecs_inv = eigen_vecs.T @ mass

    if smoothing:
        verts = eigen_vecs @ (eigen_vecs_inv @ verts)

    ##############################  atom chem feats  ##############################
     # Atom chemical features
        # x  y  z  res_type  atom_type  charge  radius  is_alphaC
        # 0  1  2  3         4          5       6       7
    # get hphob
    atom_hphob = np.array([[res_type_to_hphob[atom_inf[3]]] for atom_inf in atom_info])
    atom_feats = np.concatenate([atom_info[:, :5], atom_hphob, atom_info[:, 5:]], axis=1)

    atom_bt = BallTree(atom_coords)
    vert_nbr_dist, vert_nbr_ind = atom_bt.query(verts, k=vert_nbr_atoms)

    ##############################  Geom feats  ##############################
    vnormals = igl.per_vertex_normals(verts, faces)

    geom_feats = []
    
    _, _, k1, k2 = igl.principal_curvature(verts, faces)
    gauss_curvs = k1 * k2
    mean_curvs = 0.5 * (k1 + k2)
    geom_feats.extend([gauss_curvs.reshape(-1, 1), mean_curvs.reshape(-1, 1)])
    # HKS:
    geom_feats.append(signatures.compute_HKS(eigen_vecs, eigen_vals, num_signatures))

    geom_feats = np.concatenate(geom_feats, axis=-1)
    geom_feats = np.concatenate([verts, vnormals, geom_feats], axis=-1)

    ##############################  Laplace-Beltrami basis  ##############################
    eigs = np.concatenate(
        (eigen_vals.reshape(1, -1), eigen_vecs, eigen_vecs_inv.T), 
        axis=0
    )

    ##############################  Cache processed  ##############################

    out_fpath = processed_fpath
    np.savez(
        out_fpath,
        label=label.astype(np.int8),
        # input
        atom_info=atom_feats.astype(np.float32),
        geom_info=geom_feats.astype(np.float32),
        eigs=eigs.astype(np.float32),
        # vert_nbr
        vert_nbr_dist=vert_nbr_dist.astype(np.float32),
        vert_nbr_ind=vert_nbr_ind.astype(np.int32)
    )



class GaussianDistance(object):
    def __init__(self, start, stop, num_centers):
        self.filters = np.linspace(start, stop, num_centers)
        self.var = (stop - start) / (num_centers - 1)

    def expand(self, d):
        return np.exp(-0.5 * (d[..., None] - self.filters)**2 / self.var**2)
