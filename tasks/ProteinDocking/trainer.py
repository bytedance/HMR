"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import torch
import logging
import contextlib
import numpy as np
from tqdm import tqdm
from HMR.trainer import TrainerBase
from HMR.utils.meters import CSVWriter
from HMR.utils.meters import AverageMeter
from utils import DockingLoss, class_eval
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

class Trainer(TrainerBase):
    def __init__(self, config, data, model):
        super().__init__(config, data, model)
        self.loss = DockingLoss(config)
        # tensorboard
        if self.is_master:
            self.tb_writer = SummaryWriter(log_dir=self.out_dir, filename_suffix=f'.{self.run_name}')
            columns = ['Partition', 'Epoch', 'Loss', 'Precision', 'Recall', 'Fscore', 'AUC', 'AP']
            self.csv_writer = CSVWriter(os.path.join(self.out_dir, 'metrics.csv'), columns, overwrite=False)

    # override pure virtual function
    def _train_epoch(self, epoch, data_loader, data_sampler, partition):
        # init average meters
        losses = AverageMeter('Loss')
        precisions = AverageMeter('Precision')
        recalls = AverageMeter('Recall')
        fscores = AverageMeter('Fscore')
        aucs = AverageMeter('AUC')
        aps = AverageMeter('AP')

        # reshuffle data across GPU workers
        if isinstance(data_sampler, DistributedSampler):
            data_sampler.set_epoch(epoch)
        
        if partition == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        exploding_grad = []
        context = contextlib.nullcontext() if partition == 'train' else torch.no_grad()
        with context:
            for batch in tqdm(data_loader):
                # send data to device and compute model output
                #assert batch.target.is_pinned() == torch.cuda.is_available()
                batch.to(self.device)
                if partition == 'train' and self.fp16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output = self.model.forward(batch)
                        loss = self.loss(output)
                else:
                    output = self.model.forward(batch)
                    loss = self.loss(output)
                
                if partition == 'train':
                    # compute gradient and optimize
                    self.optimizer.zero_grad()
                    if self.fp16: # mixed precision
                        self.scaler.scale(loss).backward()
                        if self.use_hvd:
                            self.optimizer.synchronize()
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        if grad_norm > self.clip_grad_norm:
                            exploding_grad.append(grad_norm.item())
                        opt_context = self.optimizer.skip_synchronize() if self.use_hvd \
                                                                        else contextlib.nullcontext()
                        with opt_context:
                            self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else: # torch.float32 default precision
                        loss.backward()
                        if self.use_hvd:
                            self.optimizer.synchronize()
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        if grad_norm > self.clip_grad_norm:
                            exploding_grad.append(grad_norm.item())
                        opt_context = self.optimizer.skip_synchronize() if self.use_hvd \
                                                                        else contextlib.nullcontext()
                        with opt_context:
                            self.optimizer.step()
                
                # update metrics
                prec, rec, fsc, auc, ap = class_eval(output)
                losses.update(loss.item(), len(batch))
                precisions.update(sum(prec)/len(prec), len(prec))
                recalls.update(sum(rec)/len(rec), len(rec))
                fscores.update(sum(fsc)/len(fsc), len(fsc))
                aucs.update(sum(auc)/len(auc), len(auc))
                aps.update(sum(ap)/len(ap), len(ap))
        
        # synchronize metrics
        loss_avg = self.all_reduce(losses.avg)
        prec_avg = self.all_reduce(precisions.avg)
        rec_avg = self.all_reduce(recalls.avg)
        fsc_avg = self.all_reduce(fscores.avg)
        auc_avg = self.all_reduce(aucs.avg)
        ap_avg = self.all_reduce(aps.avg)
        if self.is_master:
            current_lr = self.optimizer.param_groups[0]['lr']
            lr = f'{current_lr:.8f}' if partition=='train' else '--'
            logging.info(f'Epoch {epoch} {partition.upper()}, Loss: {loss_avg:.4f}, '
                         f'Precision: {prec_avg:.3f}, Recall: {rec_avg:.3f}, Fscore: {fsc_avg:.3f}, '
                         f'AUC: {auc_avg:.3f}, AP: {ap_avg:.3f}, LR: {lr}')
            self.csv_writer.add_scalar('Epoch', epoch)
            self.csv_writer.add_scalar('Partition', partition)
            self.csv_writer.add_scalar('Loss', loss_avg)
            self.csv_writer.add_scalar('Precision', prec_avg)
            self.csv_writer.add_scalar('Recall', rec_avg)
            self.csv_writer.add_scalar('Fscore', fsc_avg)
            self.csv_writer.add_scalar('AUC', auc_avg)
            self.csv_writer.add_scalar('AP', ap_avg)
            self.csv_writer.write()
            if self.tb_writer is not None:
                if partition == 'test':
                    epoch = epoch // self.test_freq
                self.tb_writer.add_scalar(f'Loss/{partition}', loss_avg, epoch)
                self.tb_writer.add_scalar(f'Precision/{partition}', prec_avg, epoch)
                self.tb_writer.add_scalar(f'Recall/{partition}', rec_avg, epoch)
                self.tb_writer.add_scalar(f'Fscore/{partition}', fsc_avg, epoch)
                self.tb_writer.add_scalar(f'AUC/{partition}', auc_avg, epoch)
                self.tb_writer.add_scalar(f'AP/{partition}', ap_avg, epoch)
                if partition == 'train':
                    self.tb_writer.add_scalar('LR', current_lr, epoch)

        # check exploding gradient
        explode_ratio = len(exploding_grad) / len(data_loader)
        if explode_ratio > 0.01 and self.is_master:
            log_msg = [f'Exploding gradient ratio: {100*explode_ratio:.1f}%,',
                       f'exploded gradient mean: {np.mean(exploding_grad):.2f}']
            logging.info(' '.join(log_msg))
        
        performance = ap_avg # we always maximize model performance
        return loss_avg, performance


