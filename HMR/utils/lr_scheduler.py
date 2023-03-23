"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


from torch.optim.lr_scheduler import _LRScheduler, LinearLR, CosineAnnealingLR, SequentialLR

def get_lr_scheduler(scheduler, optimizer, warmup_epochs, total_epochs):
    warmup_scheduler = LinearLR(optimizer, 
                                start_factor=1E-3,
                                total_iters=warmup_epochs)
    
    if scheduler == 'PolynomialLRWithWarmup':
        decay_scheduler = PolynomialLR(optimizer,
                                       total_iters=total_epochs-warmup_epochs,
                                       power=1)
    elif scheduler == 'CosineAnnealingLRWithWarmup':
        decay_scheduler = CosineAnnealingLR(optimizer,
                                            T_max=total_epochs-warmup_epochs, 
                                            eta_min=1E-8)                        
    else:
        raise NotImplementedError
    
    return SequentialLR(optimizer, 
                        schedulers=[warmup_scheduler, decay_scheduler], 
                        milestones=[warmup_epochs])


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, power, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group['lr'] * decay_factor for group in self.optimizer.param_groups]


