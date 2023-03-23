"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import torch
import torch.nn as nn
from models import register_model
from torch_scatter import scatter

@register_model
class HMR(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config[config.model]
        h_dim = model_config.h_dim

        # local feature extraction
        self.feat_extractor = FeatExtractor(config)
        
        # propagation
        self.propagation_layers = nn.ModuleList([PropagationLayer(config) \
                                                 for _ in range(model_config.num_prop_layers)])
        
        # property prediction
        self.out_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, batch):

        # local feature extraction
        h = self.feat_extractor(batch)

        # propagation
        for ilayer in self.propagation_layers:
            h = ilayer.forward(h=h, batch=batch)

        h = self.out_mlp(h)

        # aggregation
        h_split = torch.split(h, batch.num_verts)
        h = [torch.mean(hh, dim=0, keepdim=True) for hh in h_split]
        h = torch.cat(h, dim=0)

        return h


class FeatExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config[config.model]
        h_dim = model_config.h_dim

        # chem feats
        chem_feat_dim = 15 + 2*config.num_gdf
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2*h_dim),
            nn.BatchNorm1d(2*h_dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, batch):
        # chemical features
        chem_feats = batch.chem_feats
        nbr_vid = batch.nbr_vid

        h_chem = self.chem_mlp(chem_feats)
        nbr_filter, nbr_core = h_chem.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        h = nbr_filter * nbr_core

        h = scatter(h, nbr_vid, dim=0, reduce='sum')

        return h


class PropagationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config[config.model]
        h_dim = model_config.h_dim

        # propagation
        self.propagation_time = nn.Parameter(torch.Tensor(h_dim))
        nn.init.normal_(self.propagation_time, mean=model_config.init_prop_time, std=model_config.init_prop_time)
        self.propagation_time_scale = model_config.propagation_time_scale

        self.mlp = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
        )

    def forward(self, h, batch):
        batch_idx = batch.num_verts
        h_list = torch.split(h, batch_idx)
        eigs = batch.eigs
        
        with torch.no_grad():
            self.propagation_time.data = torch.clamp(self.propagation_time, min=1e-6)
        
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
            time = self.propagation_time_scale * self.propagation_time
            prop_coeff = torch.exp(-eigen_vals.unsqueeze(-1) * time.unsqueeze(0))
            h_prop_spec = prop_coeff * h_spec
            h_prop = torch.mm(eigen_vecs, h_prop_spec)
            b_h_prop.append(h_prop)

        b_h_prop = torch.cat(b_h_prop, dim=0)
        h_out = h + self.mlp(torch.cat((h, b_h_prop), dim=-1))
        
        return h_out


