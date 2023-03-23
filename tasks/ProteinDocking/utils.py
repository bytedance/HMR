"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import torch
import logging
import torch.nn as nn
from torchmetrics import AUROC
from torchmetrics.functional import average_precision

class NCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nce_num_samples = config.HMR.nce_num_samples
        self.nce_T = config.HMR.nce_T
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, lig_iface_feats, rec_iface_feats):
        
        num_samples = min(self.nce_num_samples, rec_iface_feats.size(0))
        choices = torch.randperm(lig_iface_feats.size(0))[:num_samples]

        query = rec_iface_feats[choices]
        keys = lig_iface_feats[choices]

        logits = -torch.cdist(query, keys, p=2.0) / self.nce_T
        labels = torch.arange(query.size(0)).to(lig_iface_feats.device)
        loss = self.cross_entropy(logits, labels)
        
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, bsp, iface_label):
        bce_loss = self.bce_loss(bsp, iface_label.float())
        iface_label = iface_label.long()
        at = self.alpha.to(bsp.device).gather(0, iface_label.data.view(-1))
        pt = torch.exp(-bce_loss)
        focal_loss = at * (1-pt)**self.gamma * bce_loss

        return focal_loss.mean()


class DockingLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nce_loss_weight = config.HMR.nce_loss_weight

        # BSP loss
        if config.HMR.bsp_loss == 'bce':
            self.binary_loss = nn.BCEWithLogitsLoss()
        elif config.HMR.bsp_loss == 'focal':
            self.binary_loss = FocalLoss(alpha=config.HMR.focal_alpha, gamma=config.HMR.focal_gamma)
        else:
            raise NotImplementedError
        
        # NCE loss
        if config.HMR.nce_loss:
            self.nce_loss = NCELoss(config)
        else:
            self.nce_loss = None

    def forward(self, output):
        # features
        lig_bid = output.lig_num_verts
        rec_bid = output.rec_num_verts
        lig_iface_p2p = output.lig_iface_p2p
        rec_iface_p2p = output.rec_iface_p2p
        lig_h_split = torch.split(output.lig_h, lig_bid)
        rec_h_split = torch.split(output.rec_h, rec_bid)
        lig_bsp_split = torch.split(output.lig_bsp, lig_bid)
        rec_bsp_split = torch.split(output.rec_bsp, rec_bid)
        bsize = len(lig_bid)
        # init loss
        lig_bsp_loss = torch.tensor([0.]).to(output.lig_bsp.device)
        rec_bsp_loss = torch.tensor([0.]).to(output.lig_bsp.device)
        nce_loss = torch.tensor([0.]).to(output.lig_bsp.device)
        for bid in range(bsize):
            # lig bsp
            lig_bsp = lig_bsp_split[bid].squeeze()
            lig_label = torch.zeros(lig_bsp.size(0), dtype=torch.float32).to(lig_bsp.device)
            lig_label[lig_iface_p2p[bid][:, 0]] = 1
            lig_bsp_loss += self.binary_loss(lig_bsp, lig_label)
            # rec bsp
            rec_bsp = rec_bsp_split[bid].squeeze()
            rec_label = torch.zeros(rec_bsp.size(0), dtype=torch.float32).to(rec_bsp.device)
            rec_label[rec_iface_p2p[bid][:, 0]] = 1
            rec_bsp_loss += self.binary_loss(rec_bsp, rec_label)
            # NCE loss
            if self.nce_loss is not None:
                map_rec2lig = rec_iface_p2p[bid]
                rec_iface_feats = rec_h_split[bid][map_rec2lig[:, 0]]
                lig_iface_feats = lig_h_split[bid][map_rec2lig[:, 1]]
                nce_loss += self.nce_loss.forward(lig_iface_feats, rec_iface_feats)

        lig_bsp_loss /= bsize
        rec_bsp_loss /= bsize
        nce_loss /= bsize  

        total_loss = lig_bsp_loss + rec_bsp_loss + self.nce_loss_weight * nce_loss
        return total_loss


def class_eval(output):
    auroc = AUROC(task="binary", pos_label=1)
    precisions = []
    recalls = []
    fscores = []
    AUCs = []
    APs = []
    with torch.no_grad():
        for b_iface_p2p, bsp, batch_idx in zip([output.lig_iface_p2p, output.rec_iface_p2p], \
                                               [output.lig_bsp, output.rec_bsp], \
                                               [output.lig_num_verts,  output.rec_num_verts]):
            b_pred_prob = torch.sigmoid(bsp.squeeze())
            b_pred_label = b_pred_prob > 0.5
            pred_prob_list = torch.split(b_pred_prob, batch_idx)
            pred_label_list = torch.split(b_pred_label, batch_idx)
            for bid in range(len(batch_idx)):
                pred_label = pred_label_list[bid]
                pred_prob = pred_prob_list[bid]
                label = torch.zeros(pred_label.size(0), dtype=torch.long).to(pred_label.device)
                label[b_iface_p2p[bid][:, 0]] = 1
                tp = torch.sum((pred_label == label) & (pred_label==1))
                ps = torch.sum(pred_label)
                rs = torch.sum(label)
                prec = tp/ps if ps > 0 else 0
                rec = tp/rs if rs > 0 else 0
                fsc = (2*prec*rec)/(prec+rec) if prec + rec > 0 else 0
                precisions.append(prec)
                recalls.append(rec)
                fscores.append(fsc)
                try:
                    AUC = auroc(pred_prob, label)
                    AP = average_precision(pred_prob, label, task="binary")
                except:
                    AUC = 0.5
                    AP = 0.5
                    logging.info('Invalid AUC/AP evaluation encountered, setting AUC/AP to 0.5')
                AUCs.append(AUC)
                APs.append(AP)
        
    return precisions, recalls, fscores, AUCs, APs


# atomic van der waals radii in Angstrom unit (from mendeleev)
vdw_radii_dict = { 
    'H':   1.1,
    'C':   1.7,
    'N':   1.55,
    'O':   1.52,
    'S':   1.8,
    'F':   1.47,
    'P':   1.8,
    'Cl':  1.75,
    'Se':  1.9,
    'Br':  1.85,
    'I':   1.98,
    'UNK': 2.0,
}

# atom type label for one-hot-encoding
atom_type_dict = {
    'H':  0,
    'C':  1,
    'N':  2,
    'O':  3,
    'S':  4,
    'F':  5,
    'P':  6,
    'Cl': 7,
    'Se': 8,
    'Br': 9,
    'I':  10,
    'UNK': 11,
}

# residue type label for one-hot-encoding
res_type_dict = {
    'ALA': 0,
    'GLY': 1,
    'SER': 2,
    'THR': 3,
    'LEU': 4,
    'ILE': 5,
    'VAL': 6,
    'ASN': 7,
    'GLN': 8,
    'ARG': 9,
    'HIS': 10,
    'TRP': 11,
    'PHE': 12,
    'TYR': 13,
    'GLU': 14,
    'ASP': 15,
    'LYS': 16,
    'PRO': 17,
    'CYS': 18,
    'MET': 19,
    'UNK': 20,
}

# Kyte Doolittle scale for hydrophobicity
hydrophob_dict = {
    'ILE': 4.5,
    'VAL': 4.2,
    'LEU': 3.8,
    'PHE': 2.8,
    'CYS': 2.5,
    'MET': 1.9,
    'ALA': 1.8,
    'GLY': -0.4,
    'THR': -0.7,
    'SER': -0.8,
    'TRP': -0.9,
    'TYR': -1.3,
    'PRO': -1.6,
    'HIS': -3.2,
    'GLU': -3.5,
    'GLN': -3.5,
    'ASP': -3.5,
    'ASN': -3.5,
    'LYS': -3.9,
    'ARG': -4.5,
    'UNK': 0.0,
}


