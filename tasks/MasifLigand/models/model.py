"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import torch
import torch.nn as nn
from torch_scatter import scatter
from . import register_model


@register_model
class HMR(nn.Module):
    """HMR model for MaSIF-ligand"""

    def __init__(self, config):
        super().__init__()
        model_config = config[config.model]
        h_dim = model_config.h_dim
        dropout = model_config.dropout

        self.feat_extractor = FeatExtractor(config)
        self.propagation_layers = nn.ModuleList([PropagationLayer(config) for _ in range(model_config.num_prop_layers)])

        self.surf_pooling_agg_first = True
        
        # classification
        self.clas_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(h_dim),
                nn.SiLU()
            ) for _ in range(1)
        ])
        
        self.scorer = nn.Linear(h_dim, 7)

    def forward(self, feat_dict):

        # feature extraction
        h = self.feat_extractor(feat_dict=feat_dict)

        # propagation
        for layer in self.propagation_layers:
            h = layer.forward(h=h, feat_dict=feat_dict)

        h_split = torch.split(h, feat_dict['num_verts'])
        h_per_prot = torch.cat([torch.mean(hh, dim=0, keepdim=True) for hh in h_split], dim=0)
        for layer in self.clas_layers:
            h_per_prot = layer(h_per_prot)
        h_per_prot = self.scorer(h_per_prot)

        return h_per_prot


class FeatExtractor(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_config = config[config.model]
        h_dim = model_config.h_dim
        dropout = model_config.dropout

        # chem feat
        if config.use_chem_feat:
            # hphob, charge, atom_dist, atom_angular
            chem_feat_dim = 2 + config.num_gdf * 2
        else:
            chem_feat_dim = 1
        
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2*h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2*h_dim)
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # geom feats
        if config.use_geom_feat:
            geom_input_dim = config.num_gdf * 2 + config.num_signatures
        else:
            geom_input_dim = 1
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_input_dim, h_dim // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, h_dim // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim // 2)
        )

        # chem + geom feats
        self.feat_mlp = nn.Sequential(
            nn.Linear(h_dim + h_dim // 2, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim)
        )

    def forward(self, feat_dict):
        
        # chemical features
        h_chem = self.chem_mlp(feat_dict['chem_feats'])

        # self-filter
        nbr_filter, nbr_core = h_chem.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        h_chem = nbr_filter * nbr_core
        h_chem = scatter(h_chem, feat_dict['nbr_vids'], dim=0, reduce='sum')
        
        # geometric features
        geom_feats = feat_dict['geom_feats']
        h_geom = self.geom_mlp(geom_feats)

        # combine chemical and geometric features
        h_out = self.feat_mlp(torch.cat((h_chem, h_geom), dim=-1))
        
        return h_out


class PropagationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config[config.model]
        h_dim = model_config.h_dim

        # propagation
        self.propagation_time = nn.Parameter(torch.Tensor(h_dim))
        nn.init.normal_(self.propagation_time, mean=model_config.init_prop_time, std=model_config.init_prop_time)
        self.propagation_time_scale = model_config.propagation_time_scale
        self.apply_band_filter = model_config.apply_band_filter
        self.band_e_mean = nn.Parameter(torch.Tensor(h_dim))
        nn.init.normal_(self.band_e_mean, mean=model_config.band_e_mean, std=model_config.band_e_mean)
        self.band_e_std = nn.Parameter(torch.Tensor(h_dim))
        nn.init.normal_(self.band_e_std, mean=model_config.band_e_std, std=model_config.band_e_std)

        self.mlp = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
        )

    def forward(self, h, feat_dict):
        batch_idx = feat_dict['num_verts']
        h_list = torch.split(h, batch_idx)
        eigs = feat_dict['eigs']
        
        with torch.no_grad():
            self.propagation_time.data = torch.clamp(self.propagation_time, min=1e-6)
            self.band_e_mean.data = torch.clamp(self.band_e_mean, min=1e-3, max=5.)
            self.band_e_std.data = torch.clamp(self.band_e_std, min=1.)
        
        b_h_prop = []
        for idx in range(len(batch_idx)):
            hh = h_list[idx]
            num_verts = hh.size(0)
            # LB basis
            LB = eigs[idx]
            eigen_vals = LB[0]
            eigen_vecs = LB[1:1+num_verts]
            eigen_vecs_inv = LB[1+num_verts:].t()
            
            # propagation in eigen space
            h_spec = torch.mm(eigen_vecs_inv, hh)
            band_filter = 1
            if self.apply_band_filter:
                band_filter = torch.exp(-(self.band_e_mean.unsqueeze(0) - eigen_vals.unsqueeze(-1))**2 \
                                        / (2 * self.band_e_std.unsqueeze(0)**2))
            time = self.propagation_time_scale * self.propagation_time
            prop_coeff = torch.exp(-eigen_vals.unsqueeze(-1) * time.unsqueeze(0))
            h_prop_spec = band_filter * prop_coeff * h_spec
            h_prop = torch.mm(eigen_vecs, h_prop_spec)
            b_h_prop.append(h_prop)

        b_h_prop = torch.cat(b_h_prop, dim=0)
        h_out = h + self.mlp(torch.cat((h, b_h_prop), dim=-1))
        
        return h_out

