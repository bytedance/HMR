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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

class Trainer(TrainerBase):
    def __init__(self, config, data, model):
        super().__init__(config, data, model)
        self.criterion = torch.nn.L1Loss()
        self.target_mean = data.target_mean
        self.target_mad = data.target_mad
        # tensorboard and HDFS
        if self.is_master:
            self.tb_writer = SummaryWriter(log_dir=self.out_dir, filename_suffix=f'.{self.run_name}')
            columns = ['Partition', 'Epoch', 'Loss', 'MAE']
            self.csv_writer = CSVWriter(os.path.join(self.out_dir, 'metrics.csv'), columns, overwrite=False)

    # override pure virtual function
    def _train_epoch(self, epoch, data_loader, data_sampler, partition):
        # init average meters
        losses = AverageMeter('Loss')
        maes = AverageMeter('MAE')

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
            for batch in data_loader:
                # send data to device and compute model output
                assert batch.target.is_pinned() == torch.cuda.is_available()
                batch.to(self.device)
                target = batch.target
                if partition == 'train':
                    if self.fp16:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            output = self.model.forward(batch).squeeze(-1)
                            loss = self.criterion(output, (target - self.target_mean) / self.target_mad)
                    else:
                        output = self.model.forward(batch).squeeze(-1)
                        loss = self.criterion(output, (target - self.target_mean) / self.target_mad)
                else:
                    output = self.model.forward(batch).squeeze(-1)
                    loss = self.criterion(output * self.target_mad + self.target_mean, target)
                
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
                losses.update(loss.item(), target.size(0))
                maes.update(loss.item(), target.size(0))
        
        # synchronize metrics
        loss_avg = self.all_reduce(losses.avg)
        mae_avg = self.all_reduce(maes.avg)
        if self.is_master:
            current_lr = self.optimizer.param_groups[0]['lr']
            lr = f'{current_lr:.8f}' if partition=='train' else '--'
            logging.info(f'Epoch {epoch} {partition.upper()}, Loss: {loss_avg:.4f}, '
                         f'MAE: {mae_avg:.3f}, LR: {lr}')
            self.csv_writer.add_scalar('Epoch', epoch)
            self.csv_writer.add_scalar('Partition', partition)
            self.csv_writer.add_scalar('Loss', loss_avg)
            self.csv_writer.add_scalar('MAE', mae_avg)
            self.csv_writer.write()
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(f'Loss/{partition}', loss_avg, epoch)
                self.tb_writer.add_scalar(f'MAE/{partition}', mae_avg, epoch)
                if partition == 'train':
                    self.tb_writer.add_scalar('LR', current_lr, epoch)

        # check exploding gradient
        explode_ratio = len(exploding_grad) / len(data_loader)
        if explode_ratio > 0.01 and self.is_master:
            log_msg = [f'Exploding gradient ratio: {100*explode_ratio:.1f}%,',
                       f'exploded gradient mean: {np.mean(exploding_grad):.2f}']
            logging.info(' '.join(log_msg))
        
        performance = -mae_avg # we always maximize model performance
        return loss_avg, performance


