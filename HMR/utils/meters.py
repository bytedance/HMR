"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class CSVWriter(object):
    def __init__(self, csv_fpath, columns, overwrite):
        self.csv_fpath = csv_fpath
        self.columns = columns
        
        if os.path.isfile(self.csv_fpath) and overwrite:
            os.remove(csv_fpath)

        if not os.path.isfile(self.csv_fpath):
            # write columns
            with open(self.csv_fpath, 'w') as handles:
                handles.write(','.join(self.columns) + '\n')
        
        self.values = {key: '' for key in self.columns}
    
    def add_scalar(self, name, value):
        assert name in self.columns
        self.values[name] = value
    
    def write(self):
        with open(self.csv_fpath, 'a') as handles:
            handles.write(','.join([str(self.values[key]) for key in self.columns]) + '\n')
        self.values = {key: '' for key in self.columns}


