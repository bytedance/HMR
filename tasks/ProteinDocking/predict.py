"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import torch
import numpy as np
from tqdm import tqdm
from main import get_config
from models import load_model
from data import DatasetProteinDocking
from torch.utils.data import DataLoader
from HMR.utils.meters import AverageMeter
from utils import class_eval, DockingLoss

def predict(config, out_dir):
    
    test_dataset = DatasetProteinDocking(config, split='test')
    test_loader = DataLoader(test_dataset, 
                             batch_size=config.batch_size,
                             num_workers=config.num_data_workers,
                             collate_fn=test_dataset.collate_wrapper,
                             pin_memory=torch.cuda.is_available())  
    
    # initialize model
    Model = load_model(config.model)
    model = Model(config[config.model])
    
    if torch.cuda.is_available():
        model.cuda()
    criterion = DockingLoss(config)

    with open(config.restore, 'rb') as fin:
        key = 'state_dict' if 'trained' in config.restore else 'model'  # compatibility issue
        state_dict = torch.load(fin, map_location='cpu')[key]
    model.load_state_dict(state_dict)
    
    eval(model, test_loader, criterion, config.data_dir, out_dir)


def eval(model, data_loader, criterion, data_dir, out_dir):
    # init average meters
    losses = AverageMeter('Loss')
    precisions = AverageMeter('Precision')
    recalls = AverageMeter('Recall')
    fscores = AverageMeter('Fscore')
    aucs = AverageMeter('AUC')
    aps = AverageMeter('AP')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # send data to device and compute model output
            if torch.cuda.is_available():
                batch.to('cuda')
            output = model.forward(batch)
            loss = criterion(output)

            # update metrics
            prec, rec, fsc, auc, ap = class_eval(output)
            losses.update(loss.item(), len(batch))
            precisions.update(sum(prec)/len(prec), len(prec))
            recalls.update(sum(rec)/len(rec), len(rec))
            fscores.update(sum(fsc)/len(fsc), len(fsc))
            aucs.update(sum(auc)/len(auc), len(auc))
            aps.update(sum(ap)/len(ap), len(ap))
            
            # features
            lig_bsp = batch.lig_bsp.split(batch.lig_num_verts)
            lig_h = batch.lig_h.split(batch.lig_num_verts)
            rec_bsp = batch.rec_bsp.split(batch.rec_num_verts)
            rec_h = batch.rec_h.split(batch.rec_num_verts)

            # save predictions
            for idx in range(len(batch)):
                fpath = batch.fpath[idx]
                # load original data for vertices, faces, and atom info
                fname = fpath[fpath.rfind('/')+1:]
                raw_fpath = os.path.join(data_dir, 'dataset_DB5/', fname)
                raw_data = np.load(raw_fpath)
                lig_verts = raw_data['lig_verts']
                lig_faces = raw_data['lig_faces']
                lig_atom_info = raw_data['lig_atom_info']
                rec_verts = raw_data['rec_verts']
                rec_faces = raw_data['rec_faces']
                rec_atom_info = raw_data['rec_atom_info']
                # ground truth labels
                lig_iface_label = torch.zeros(lig_bsp[idx].size(0), dtype=torch.float32, requires_grad=False)
                lig_iface_label[batch.lig_iface_p2p[idx][:, 0]] = 1
                assert len(lig_verts) == lig_iface_label.size(0)
                rec_iface_label = torch.zeros(rec_bsp[idx].size(0), dtype=torch.float32, requires_grad=False)
                rec_iface_label[batch.rec_iface_p2p[idx][:, 0]] = 1
                assert len(rec_verts) == rec_iface_label.size(0)
                # save output
                out_file = os.path.join(out_dir, fname)
                np.savez(out_file, lig_verts=lig_verts,
                                   lig_faces=lig_faces,
                                   lig_atom_info=lig_atom_info,
                                   lig_bsp=lig_bsp[idx].cpu().numpy(),
                                   lig_h=lig_h[idx].cpu().numpy(),
                                   lig_iface_label=lig_iface_label.cpu().numpy(),
                                   rec_verts=rec_verts,
                                   rec_faces=rec_faces,
                                   rec_atom_info=rec_atom_info,
                                   rec_bsp=rec_bsp[idx].cpu().numpy(),
                                   rec_h=rec_h[idx].cpu().numpy(),
                                   rec_iface_label=rec_iface_label.cpu().numpy())
    
    print_info = ['***** test\n', 
                 f'Loss: {losses.avg:.3f}, ',
                 f'Prec: {precisions.avg:.3f}, Rec: {recalls.avg:.3f}, ',
                 f'Fsc: {fscores.avg:.3f}, AUC: {aucs.avg:.3f}, ',
                 f'AP: {aps.avg:.3f}\n',
                 f'*****\n']
    print(''.join(print_info))  


if __name__ == '__main__':
    config = get_config()
    config.serial = True
    if config.restore is None:
        config.restore = f'./output/model_best.pt'

    out_dir = os.path.join(config.out_dir, 'pred_results')
    os.makedirs(out_dir, exist_ok=True)

    predict(config, out_dir)


