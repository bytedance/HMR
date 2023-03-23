"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import torch.nn as nn
from models import register_model
from models.modules import FeatNet, PropagationLayer, CrossAttentionLayer

class MPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # intra-surface propagation
        self.propagation_layers = nn.ModuleList([PropagationLayer(config) \
                                                 for _ in range(config.num_propagation_layers)])

        # inter-surface cross attention
        self.cross_attn_layers = nn.ModuleList([CrossAttentionLayer(config) \
                                                for _ in range(config.num_cross_attn_layers)])

    def forward(self, lig_h, rec_h, batch):

        # intra-surface propagation
        for ilayer in self.propagation_layers:
            lig_h = ilayer.forward(h=lig_h, 
                                   num_verts=batch.lig_num_verts, 
                                   eigs=batch.lig_eigs, 
                                   grad_op=batch.lig_grad_op,
                                   grad_basis=batch.lig_grad_basis,
                                   vnormals=batch.lig_vnormals)
            rec_h = ilayer.forward(h=rec_h,
                                   num_verts=batch.rec_num_verts, 
                                   eigs=batch.rec_eigs, 
                                   grad_op=batch.rec_grad_op,
                                   grad_basis=batch.rec_grad_basis,
                                   vnormals=batch.rec_vnormals)

        # inter-surface attention
        for jlayer in self.cross_attn_layers:
            lig_out = jlayer.forward(src_h=rec_h, dst_h=lig_h, src_num_verts=batch.rec_num_verts, dst_num_verts=batch.lig_num_verts)
            rec_out = jlayer.forward(src_h=lig_h, dst_h=rec_h, src_num_verts=batch.lig_num_verts, dst_num_verts=batch.rec_num_verts)
            lig_h = lig_out
            rec_h = rec_out

        return lig_h, rec_h


@register_model
class HMR(nn.Module):
    def __init__(self, config):
        super().__init__()
        h_dim = config.h_dim
        dropout = config.dropout

        # initialize features
        self.feat_init = FeatNet(config)
        
        # message passing
        self.message_passing_blocks = nn.ModuleList([MPBlock(config) \
                                                     for _ in range(config.num_message_passing_blocks)])
        
        # smoothing
        self.smoothing_layers = nn.ModuleList([PropagationLayer(config) \
                                               for _ in range(config.num_smoothing_layers)])
        
        # binding site prediction
        self.bsp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, batch):

        # feature extraction
        lig_h = self.feat_init(chem_feats=batch.lig_chem_feats, geom_feats=batch.lig_geom_feats)
        rec_h = self.feat_init(chem_feats=batch.rec_chem_feats, geom_feats=batch.rec_geom_feats)

        # propagation
        for block in self.message_passing_blocks:
            lig_h, rec_h = block.forward(lig_h=lig_h, rec_h=rec_h, batch=batch)

        # smoothing
        for layer in self.smoothing_layers:
            lig_h = layer.forward(h=lig_h, 
                                  num_verts=batch.lig_num_verts, 
                                  eigs=batch.lig_eigs, 
                                  grad_op=batch.lig_grad_op,
                                  grad_basis=batch.lig_grad_basis,
                                  vnormals=batch.lig_vnormals)
            rec_h = layer.forward(h=rec_h,
                                  num_verts=batch.rec_num_verts, 
                                  eigs=batch.rec_eigs, 
                                  grad_op=batch.rec_grad_op,
                                  grad_basis=batch.rec_grad_basis,
                                  vnormals=batch.rec_vnormals)
        
        batch.lig_h = lig_h
        batch.rec_h = rec_h

        # binding site prediction
        batch.lig_bsp = self.bsp(lig_h)
        batch.rec_bsp = self.bsp(rec_h)
        
        return batch


