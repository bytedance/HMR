"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        h_dim = config.h_dim
        dropout = config.dropout

        # chem feats
        embed_dim = config.chem_embed_dim
        self.res_type_embedding = nn.Embedding(21, embed_dim)
        self.atom_type_embedding = nn.Embedding(7, embed_dim)
        chem_feat_dim = 2*embed_dim + 4 + 2*config.num_gdf
        self.vert_nbr_atoms = config.vert_nbr_atoms
        #self.chem_pooling = config.chem_pooling
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, 2*h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2*h_dim),
            nn.ReLU(),
            nn.Linear(2*h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim)
        )

        # geom feats
        geom_input_dim = 2*config.num_gdf + config.num_signatures
        geom_feat_dim = config.geom_feat_dim
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_input_dim, 2*geom_feat_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2*geom_feat_dim),
            nn.ReLU(),
            nn.Linear(2*geom_feat_dim, geom_feat_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(geom_feat_dim)
        )

        # chem + geom feats
        self.feat_mlp = nn.Sequential(
            nn.Linear(h_dim+geom_feat_dim, 2*h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2*h_dim),
            nn.ReLU(),
            nn.Linear(2*h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim)
        )

    def forward(self, chem_feats, geom_feats):
        
        # chemical features
        res_type_embedding = self.res_type_embedding(chem_feats[:, :, 0].long())
        atom_type_embedding = self.atom_type_embedding(chem_feats[:, :, 1].long())
        chem_feats = torch.cat((res_type_embedding, atom_type_embedding, chem_feats[:, :, 2:]), dim=-1)
        h_chem = self.chem_mlp(chem_feats.view(-1, chem_feats.size(-1)))
        #if self.chem_pooling == 'max':
        #    h_chem = torch.max(h_chem.view(-1, self.vert_nbr_atoms, h_chem.size(-1)), dim=1)[0]
        #else:
        h_chem = torch.mean(h_chem.view(-1, self.vert_nbr_atoms, h_chem.size(-1)), dim=1)
        
        # geometric features
        h_geom = self.geom_mlp(geom_feats)

        # combine chemical and geometric features
        h_out = self.feat_mlp(torch.cat((h_chem, h_geom), dim=-1))
        
        return h_out


class PropagationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fp16 = config.fp16
        h_dim = config.h_dim
        dropout = config.dropout

        # propagation
        self.propagation_time = nn.Parameter(torch.Tensor(h_dim))
        nn.init.normal_(self.propagation_time, mean=1., std=1.)
        self.propagation_time_scale = config.propagation_time_scale
        self.apply_band_filter = config.apply_band_filter
        self.band_e_mean = nn.Parameter(torch.Tensor(h_dim))
        nn.init.normal_(self.band_e_mean, mean=config.band_e_mean, std=0.1)
        self.band_e_std = nn.Parameter(torch.Tensor(h_dim))
        nn.init.normal_(self.band_e_std, mean=config.band_e_std, std=0.1)
        
        # directional gradient
        self.grad_mlp = nn.Linear(h_dim, h_dim, bias=False)
        self.grad_mlp2 = nn.Linear(h_dim, h_dim, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(3*h_dim, 2*h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2*h_dim),
            nn.ReLU(),
            nn.Linear(2*h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim)
        )

    def forward(self, h, num_verts, eigs, grad_op, grad_basis, vnormals):
        h_list = torch.split(h, num_verts)
        grad_basis_list = torch.split(grad_basis, num_verts)
        vnormals_list = torch.split(vnormals, num_verts)
        
        with torch.no_grad():
            self.propagation_time.data = torch.clamp(self.propagation_time, min=1e-6)
            self.band_e_mean.data = torch.clamp(self.band_e_mean, min=1e-6, max=0.3)
            self.band_e_std.data = torch.clamp(self.band_e_std, min=0.05)
        
        b_h_prop = []
        b_h_prop_grad = []
        for idx in range(len(num_verts)):
            hh = h_list[idx]
            num_v = hh.size(0)
            # LB basis
            LB = eigs[idx]
            eigen_vals = LB[0]
            eigen_vecs = LB[1:1+num_v]
            eigen_vecs_inv = LB[1+num_v:].t()
            
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

            # construct gradient operator
            igrad = grad_op[idx]
            vx, vy, row_grad, col_grad = igrad[0], igrad[1], igrad[2], igrad[3]
            ind_grad = torch.vstack((row_grad, col_grad)).long()
            size = torch.Size((num_v, num_v))
            gx_op = torch.sparse_coo_tensor(ind_grad, vx, size)
            gy_op = torch.sparse_coo_tensor(ind_grad, vy, size)
            if self.fp16:
                with torch.cuda.amp.autocast(enabled=False):
                    gradx = torch.clamp(torch.sparse.mm(gx_op, hh.float()), min=-1, max=1)
                    grady = torch.clamp(torch.sparse.mm(gy_op, hh.float()), min=-1, max=1)
            else:
                gradx = torch.clamp(torch.sparse.mm(gx_op, hh), min=-1, max=1)
                grady = torch.clamp(torch.sparse.mm(gy_op, hh), min=-1, max=1)
            
            grad_basis_x = grad_basis_list[idx][:, :3]
            grad_basis_y = grad_basis_list[idx][:, 3:]
        
            gradx_vec = self.grad_mlp(gradx).unsqueeze(-1) * grad_basis_x.unsqueeze(1)
            grady_vec = self.grad_mlp(grady).unsqueeze(-1) * grad_basis_y.unsqueeze(1)
            grad_vec = gradx_vec + grady_vec # (N, h_dim, 3)

            gradxp_vec = self.grad_mlp2(gradx).unsqueeze(-1) * grad_basis_x.unsqueeze(1)
            gradyp_vec = self.grad_mlp2(grady).unsqueeze(-1) * grad_basis_y.unsqueeze(1)
            gradp_vec = gradxp_vec + gradyp_vec # (N, h_dim, 3)
            
            grad_cross = torch.cross(grad_vec, gradp_vec, dim=-1)
            vnormals = vnormals_list[idx].unsqueeze(1).expand(-1, gradx.size(-1), -1) # (N, h_dim, 3)
            h_grad = torch.einsum('ijk,ijk->ij', grad_cross, vnormals) # (N, h_dim)
            
            b_h_prop_grad.append(h_grad)

        b_h_prop = torch.cat(b_h_prop, dim=0)
        b_h_prop_grad = torch.cat(b_h_prop_grad, dim=0)
        h_out = h + self.mlp(torch.cat((h, b_h_prop, b_h_prop_grad), dim=-1))
        
        return h_out


class CrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h_dim = config.h_dim
        self.h_dim_div = self.h_dim // config.h_dim_div
        self.num_heads = config.num_attn_heads
        assert self.h_dim_div % self.num_heads == 0
        self.head_dim = self.h_dim_div // self.num_heads
        self.merge = nn.Conv1d(self.h_dim_div, self.h_dim_div, kernel_size=1)
        self.proj = nn.ModuleList([nn.Conv1d(self.h_dim, self.h_dim_div, kernel_size=1) for _ in range(3)])
        dropout = config.dropout

        self.mlp = nn.Sequential(
            nn.Linear(self.h_dim+self.h_dim_div, self.h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.h_dim)
        )
    
    def forward(self, src_h, dst_h, src_num_verts, dst_num_verts):
        h = dst_h
        src_h_list = torch.split(src_h, src_num_verts)
        dst_h_list = torch.split(dst_h, dst_num_verts)
        h_msg = []
        for idx in range(len(src_num_verts)):
            src_hh = src_h_list[idx].unsqueeze(0).transpose(1, 2)
            dst_hh = dst_h_list[idx].unsqueeze(0).transpose(1, 2)
            query, key, value = [ll(hh).view(1, self.head_dim, self.num_heads, -1) \
                for ll, hh in zip(self.proj, (dst_hh, src_hh, src_hh))]
            dim = query.shape[1]
            scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / (dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            h_dst = torch.einsum('bhnm,bdhm->bdhn', attn, value) 
            h_dst = self.merge(h_dst.contiguous().view(1, self.h_dim_div, -1))
            h_msg.append(h_dst.squeeze(0).transpose(0, 1))
        h_msg = torch.cat(h_msg, dim=0)

        # skip connection
        h_out = h + self.mlp(torch.cat((h, h_msg), dim=-1))

        return h_out


