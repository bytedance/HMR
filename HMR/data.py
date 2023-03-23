"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import torch
import horovod.torch as hvd
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

# base class for data loaders
class DataLoaderBase(ABC):
    def __init__(self, config):
        self.use_hvd = config.use_hvd
        self.batch_size = config.batch_size
        self.num_data_workers = config.num_data_workers

        self.train_set = None
        self.valid_set = None
        self.test_set = None

        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
    
    @abstractmethod
    def _init_datasets(self):
        pass

    def _init_samplers(self):
        if self.use_hvd:
            self.train_sampler = DistributedSampler(self.train_set, num_replicas=hvd.size(), rank=hvd.rank())
            self.valid_sampler = DistributedSampler(self.valid_set, num_replicas=hvd.size(), rank=hvd.rank())
            self.test_sampler = DistributedSampler(self.test_set, num_replicas=hvd.size(), rank=hvd.rank())
        else:
            self.train_sampler = RandomSampler(self.train_set)
            self.valid_sampler = RandomSampler(self.valid_set)
            self.test_sampler = RandomSampler(self.test_set)

    def _init_loaders(self):
        self.train_loader = DataLoader(self.train_set, 
                                       batch_size=self.batch_size,
                                       sampler=self.train_sampler,
                                       num_workers=self.num_data_workers,
                                       collate_fn=self.train_set.collate_wrapper,
                                       pin_memory=torch.cuda.is_available())
        self.valid_loader = DataLoader(self.valid_set, 
                                       batch_size=self.batch_size,
                                       sampler=self.valid_sampler,
                                       num_workers=self.num_data_workers,
                                       collate_fn=self.valid_set.collate_wrapper,
                                       pin_memory=torch.cuda.is_available())
        self.test_loader = DataLoader(self.test_set, 
                                      batch_size=self.batch_size,
                                      sampler=self.test_sampler,
                                      num_workers=self.num_data_workers,
                                      collate_fn=self.test_set.collate_wrapper,
                                      pin_memory=torch.cuda.is_available())  


