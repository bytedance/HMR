"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""

import os
import time
import torch
import shutil
import logging
import numpy as np
from tqdm import tqdm
import horovod.torch as hvd
from abc import ABC, abstractmethod
from HMR.utils.lr_scheduler import get_lr_scheduler

# base trainer class
class TrainerBase(ABC):
    def __init__(self, config, data, model):
        self.config = config
        self.device = config.device
        self.use_hvd = config.use_hvd
        self.is_master = config.is_master
        self.run_name = config.run_name
        self.train_loader, self.valid_loader, self.test_loader = \
            data.train_loader, data.valid_loader, data.test_loader
        self.train_sampler = data.train_sampler
        self.best_perf = float('-inf')
        self.epochs = config.epochs
        self.start_epoch = 1
        self.warmup_epochs = config.warmup_epochs
        self.test_freq = config.test_freq
        self.clip_grad_norm = config.clip_grad_norm
        self.fp16 = config.fp16
        self.scaler = torch.cuda.amp.GradScaler()
        self.out_dir = config.out_dir
        self.auto_resume = config.auto_resume
        
        # model
        self.model = model
        if self.device == 'cuda':
            self.model.cuda()
        if self.is_master:
            logging.info(self.model)
            learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f'Number of learnable model parameters: {learnable_params}')
            msg = [f'Batch size: {config.batch_size}, number of batches in data loaders - train:',
                   f'{len(self.train_loader)}, valid: {len(self.valid_loader)}, test: {len(self.test_loader)}']
            logging.info(' '.join(msg))
        
        # optimizer
        if not config.optimizer in ['Adam', 'AdamW']:
            raise NotImplementedError
        optim = getattr(torch.optim, config.optimizer)
        self.optimizer = optim(self.model.parameters(),
                               lr=config.lr,
                               betas=(0.9, 0.999), 
                               eps=1e-08,
                               weight_decay=config.weight_decay)

        # distributed training
        if self.use_hvd:
            compression = hvd.Compression.fp16 if self.fp16 else hvd.Compression.none
            self.optimizer = hvd.DistributedOptimizer(self.optimizer, compression=compression)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        
        # LR scheduler
        self.scheduler = get_lr_scheduler(scheduler=config.lr_scheduler,
                                          optimizer=self.optimizer,
                                          warmup_epochs=config.warmup_epochs, 
                                          total_epochs=config.epochs)

    def train(self):
        # automatically resume training
        try:
            self._auto_resume()
        except:
            if self.is_master:
                logging.info(f'Failed to load checkpoint from {self.out_dir}, start training from scratch..')

        train_t0 = time.time()
        epoch_times = []
        with tqdm(range(self.start_epoch, self.epochs+1)) as tq:
            for epoch in tq:
                tq.set_description(f'Epoch {epoch}')
                epoch_t0 = time.time()
                
                train_loss, train_perf = self._train_epoch(epoch=epoch, 
                                                           data_loader=self.train_loader, 
                                                           data_sampler=self.train_sampler, 
                                                           partition='train')
                valid_loss, valid_perf = self._train_epoch(epoch=epoch, 
                                                           data_loader=self.valid_loader, 
                                                           data_sampler=None, 
                                                           partition='valid')
                
                self.scheduler.step()
                
                tq.set_postfix(train_loss=train_loss, valid_loss=valid_loss, 
                               train_perf=abs(train_perf), valid_perf=abs(valid_perf))

                epoch_times.append(time.time() - epoch_t0)

                # save checkpoint
                is_best = valid_perf > self.best_perf
                self.best_perf = max(valid_perf, self.best_perf)
                if self.is_master:
                    self._save_checkpoint(epoch=epoch,
                                          is_best=is_best,
                                          best_perf=self.best_perf)

                # predict on test set using the latest model
                if epoch % self.test_freq == 0:
                    if self.is_master:
                        logging.info('Evaluating the latest model on test set')
                    self._train_epoch(epoch=epoch, 
                                      data_loader=self.test_loader, 
                                      data_sampler=None, 
                                      partition='test')
        
        # evaluate best model on test set
        if self.is_master:
            log_msg = [f'Total training time: {time.time() - train_t0:.1f} sec,',
                       f'total number of epochs: {epoch:d},',
                       f'average epoch time: {np.mean(epoch_times):.1f} sec']
            logging.info(' '.join(log_msg))
            self.tb_writer = None # do not write to tensorboard
            logging.info('---------Evaluate Best Model on Test Set---------------')
        with open(os.path.join(self.out_dir, 'model_best.pt'), 'rb') as fin:
            best_model = torch.load(fin, map_location='cpu')['model']
        self.model.load_state_dict(best_model)
        self._train_epoch(epoch=-1, 
                          data_loader=self.test_loader, 
                          data_sampler=None, 
                          partition='test')

    def _auto_resume(self):
        assert self.auto_resume
        # load from local output directory
        with open(os.path.join(self.out_dir, 'model_last.pt'), 'rb') as fin:
            checkpoint = torch.load(fin, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_perf = checkpoint['best_perf']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.is_master:
            logging.info(f'Loaded checkpoint from {self.out_dir}, resume training at epoch {self.start_epoch}..')
    
    def _save_checkpoint(self, epoch, is_best, best_perf):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_perf': best_perf,
        }
        filename = os.path.join(self.out_dir, 'model_last.pt')
        torch.save(state_dict, filename)
        if is_best:
            logging.info(f'Saving current model as the best')
            shutil.copyfile(filename, os.path.join(self.out_dir, 'model_best.pt')) 
    
    @abstractmethod
    def _train_epoch(self):
        pass
    
    @staticmethod
    def all_reduce(val):
        if torch.cuda.device_count() < 2:
            return val
        
        if not isinstance(val, torch.tensor):
            val = torch.tensor(val)
        avg_tensor = hvd.allreduce(val)
        return avg_tensor.item()


