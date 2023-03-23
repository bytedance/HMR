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
import horovod.torch as hvd

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from HMR.trainer import TrainerBase
from HMR.utils.meters import CSVWriter

from metrics import multi_class_eval


class Trainer(TrainerBase):

    def __init__(self, config, data, model):
        super().__init__(config, data, model)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # tensorboard and HDFS
        if self.is_master:
            self.tb_writer = SummaryWriter(log_dir=self.out_dir, filename_suffix=f'.{self.run_name}')
            columns = [
                "Epoch", "Partition",
                "CrossEntropy_avg",
                "Accuracy_micro", "Accuracy_macro", "Accuracy_balanced",
                "Precision_micro", "Precision_macro", 
                "Recall_micro", "Recall_macro", 
                "F1_micro", "F1_macro", 
                "AUROC_macro",
            ]
            self.csv_writer = CSVWriter(os.path.join(self.out_dir, 'metrics.csv'), columns, overwrite=False)

    # override pure virtual function
    def _train_epoch(self, epoch, data_loader, data_sampler, partition):
        # init average meters
        pred_scores = []
        labels = []
        cross_entropy_avg_list = []

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
                assert batch.labels.is_pinned() == torch.cuda.is_available()
                batch.to(self.device)
                
                if partition == 'train':
                    if self.fp16:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            output = self.model.forward(batch)
                            cross_entropy_loss = self.criterion(output, batch.labels)
                    else:
                        output = self.model.forward(batch)
                        cross_entropy_loss = self.criterion(output, batch.labels)
                else:
                    output = self.model.forward(batch).squeeze(-1)
                    cross_entropy_loss = self.criterion(output, batch.labels)
                
                if partition == 'train':
                    # compute gradient and optimize
                    self.optimizer.zero_grad()
                    if self.fp16: # mixed precision
                        self.scaler.scale(cross_entropy_loss).backward()
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
                        cross_entropy_loss.backward()
                        if self.use_hvd:
                            self.optimizer.synchronize()
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        if grad_norm > self.clip_grad_norm:
                            exploding_grad.append(grad_norm.item())
                        opt_context = self.optimizer.skip_synchronize() if self.use_hvd \
                                                                        else contextlib.nullcontext()
                        with opt_context:
                            self.optimizer.step()
                
                # add ouputs
                pred_scores.append(output)
                labels.append(batch['labels'])
                cross_entropy_avg_list += [cross_entropy_loss.item()] * batch['labels'].size(dim=0)
        
        
        # synchronize metrics
        cross_entropy = np.mean(cross_entropy_avg_list)
        accuracy_macro, accuracy_micro, accuracy_balanced, \
            precision_macro, precision_micro, \
            recall_macro, recall_micro, \
            f1_macro, f1_micro, \
            auroc_macro = multi_class_eval(torch.cat(pred_scores, dim=0), torch.cat(labels, dim=0), K=7)

        cross_entropy_avg = self.avg(cross_entropy)
        auroc_macro_avg = self.avg(auroc_macro)
        accuracy_macro_avg, accuracy_micro_avg, accuracy_balanced_avg = self.avg(accuracy_macro), self.avg(accuracy_micro), self.avg(accuracy_balanced)
        precision_macro_avg, precision_micro_avg = self.avg(precision_macro), self.avg(precision_micro)
        recall_macro_avg, recall_micro_avg = self.avg(recall_macro), self.avg(recall_micro)
        f1_macro_avg, f1_micro_avg = self.avg(f1_macro), self.avg(f1_micro)

        if self.is_master:
            current_lr = self.optimizer.param_groups[0]['lr']
            lr = f'{current_lr:.8f}' if partition == 'train' else '--'

            print_info= [
                f'===> Epoch {epoch} {partition.upper()}, LR: {lr}\n',
                f'CrossEntropyAvg: {cross_entropy_avg:.3f}\n', 
                f'AccuracyAvg: {accuracy_macro_avg:.3f} (macro), {accuracy_micro_avg:.3f} (micro), {accuracy_balanced_avg:.3f} (balanced)\n',
                f'PrecisionAvg: {precision_macro_avg:.3f} (macro), {precision_micro_avg:.3f} (micro)\n',
                f'RecallAvg: {recall_macro_avg:.3f} (macro), {recall_micro_avg:.3f} (micro)\n',
                f'F1Avg: {f1_macro_avg:.3f} (macro), {f1_micro_avg:.3f} (micro)\n',
                f'AUROCAvg: {auroc_macro_avg:.3f} (macro)\n',
            ]
            logging.info(''.join(print_info))

            self.csv_writer.add_scalar("Epoch", epoch)
            self.csv_writer.add_scalar("Partition", partition)
            self.csv_writer.add_scalar("CrossEntropy_avg", cross_entropy_avg)

            self.csv_writer.add_scalar("Accuracy_macro", accuracy_macro_avg)
            self.csv_writer.add_scalar("Accuracy_micro", accuracy_micro_avg)
            self.csv_writer.add_scalar("Accuracy_balanced", accuracy_balanced_avg)

            self.csv_writer.add_scalar("Precision_macro", precision_macro_avg)
            self.csv_writer.add_scalar("Precision_micro", precision_micro_avg)

            self.csv_writer.add_scalar("Recall_macro", recall_macro_avg)
            self.csv_writer.add_scalar("Recall_micro", recall_micro_avg)

            self.csv_writer.add_scalar("F1_macro", f1_macro_avg)
            self.csv_writer.add_scalar("F1_micro", f1_micro_avg)

            self.csv_writer.add_scalar("AUROC_macro", auroc_macro_avg)

            self.csv_writer.write()

            if self.tb_writer is not None:
                self.tb_writer.add_scalar(f"CrossEntropy_avg/{partition}", cross_entropy_avg, epoch)

                self.tb_writer.add_scalar(f"Accuracy_macro/{partition}", accuracy_macro_avg, epoch)
                self.tb_writer.add_scalar(f"Accuracy_micro/{partition}", accuracy_micro_avg, epoch)
                self.tb_writer.add_scalar(f"Accuracy_balanced/{partition}", accuracy_balanced_avg, epoch)

                self.tb_writer.add_scalar(f"Precision_macro/{partition}", precision_macro_avg, epoch)
                self.tb_writer.add_scalar(f"Precision_micro/{partition}", precision_micro_avg, epoch)

                self.tb_writer.add_scalar(f"Recall_macro/{partition}", recall_macro_avg, epoch)
                self.tb_writer.add_scalar(f"Recall_micro/{partition}", recall_micro_avg, epoch)

                self.tb_writer.add_scalar(f"F1_macro/{partition}", f1_macro_avg, epoch)
                self.tb_writer.add_scalar(f"F1_micro/{partition}", f1_micro_avg, epoch)

                self.tb_writer.add_scalar(f"AUROC_macro/{partition}", auroc_macro_avg, epoch)
                if partition == 'train':
                    self.tb_writer.add_scalar('LR', current_lr, epoch)
                

        # check exploding gradient
        explode_ratio = len(exploding_grad) / len(data_loader)
        if explode_ratio > 0.01 and self.is_master:
            log_msg = [f'Exploding gradient ratio: {100*explode_ratio:.1f}%,',
                       f'exploded gradient mean: {np.mean(exploding_grad):.2f}']
            logging.info(' '.join(log_msg))
        
        performance = accuracy_balanced_avg # we always maximize model performance
        return cross_entropy_avg, performance


    def avg(self, val):
        if self.use_hvd:
            tensor = torch.tensor(val)
            avg_tensor = hvd.allreduce(tensor)
            return avg_tensor.item()
        else:
            return val

