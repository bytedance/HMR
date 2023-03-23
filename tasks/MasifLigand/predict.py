"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from functools import partialmethod
from easydict import EasyDict as edict
from pathlib import Path

from HMR.utils.helpers import set_seed
from models import load_model
from data import load_data_fpaths_from_split_file, DatasetMasifLigand, CustomBatchMasifLigand
from metrics import multi_class_eval


def get_config():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    # logging arguments
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--pred_out_dir', type=str, default=None)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--mute_tqdm', type=lambda x: eval(x), default=False)
    # data arguments
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--processed_dir', type=str)
    parser.add_argument('--test_list', type=str, default=None)    
    parser.add_argument('--num_data_workers', type=int)
    # model arguments
    parser.add_argument('--use_chem_feat', type=lambda x: eval(str(x)), default=True)
    parser.add_argument('--use_geom_feat', type=lambda x: eval(x), default=True)

    # model-specific arguments
    model_names = ['HMR']
    parser.add_argument('--model', type=str, choices=model_names)
    args = parser.parse_args()
    
    # load default config
    with open(args.config) as f:
        config = json.load(f)
    
    # update config with user-defined args
    for arg in vars(args):
        if getattr(args, arg) is not None:
            model_name = arg[:arg.find('_')]
            if model_name in model_names:
                model_arg = arg[arg.find('_')+1:]
                config[model_name][model_arg] = getattr(args, arg)
            else:
                config[arg] = getattr(args, arg)
    config = edict(config)

    if config.mute_tqdm:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    # adjust config for prediction
    config.train_split_file = None
    config.valid_split_file = None
    config.test_split_file = None
    config.batch_size = 1
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.use_hvd = False
    
    if config.model_dir is None:
        config.model_dir = os.path.join(config.run_name, 'model_best.pt')
    
    assert os.path.exists(config.model_dir), f"Model path {config.model_dir} doesn't exist"
    
    return config


def predict(model, dataset, device):
    """Get model predictions"""
    test_pids = []
    pred_scores = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            # send data to device and compute model output
            batch = CustomBatchMasifLigand([data])
            test_pids.append(Path(batch['fpaths'][0]).stem)

            batch.to(device)

            output = model(batch)
            pred_scores.append(output)

            if len(batch.labels) != 0:
                labels.append(batch['labels'])
    
    pred_scores = torch.cat(pred_scores, dim=0)
    if labels != []:
        labels = torch.cat(labels, dim=0)

    return test_pids, pred_scores, labels


def evalulate(pred_scores, labels):
    """Evaluate predictions"""

    accuracy_macro, accuracy_micro, accuracy_balanced, \
        precision_macro, precision_micro, \
        recall_macro, recall_micro, \
        f1_macro, f1_micro, \
        auroc_macro = multi_class_eval(pred_scores, labels, K=7)

    print_info= [
        f'<===== Test =====>\n',
        f'AccuracyAvg: {accuracy_macro:.3f} (macro), {accuracy_micro:.3f} (micro), {accuracy_balanced:.3f} (balanced)\n',
        f'PrecisionAvg: {precision_macro:.3f} (macro), {precision_micro:.3f} (micro)\n',
        f'RecallAvg: {recall_macro:.3f} (macro), {recall_micro:.3f} (micro)\n',
        f'F1Avg: {f1_macro:.3f} (macro), {f1_micro:.3f} (micro)\n',
        f'AUROCAvg: {auroc_macro:.3f} (macro)\n',
    ]
    print(''.join(print_info))


def main(config, out_dir):
    
    # load model
    Model = load_model(config.model)
    model = Model(config)
    model.to(config.device)
    state_dict = torch.load(config.model_dir, map_location='cpu')
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    else:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    # predict and evaluate
    fpaths = load_data_fpaths_from_split_file(config.data_dir, config.test_list)
    dataset = DatasetMasifLigand(config, fpaths)


    test_pids, pred_scores, labels = predict(model, dataset, config.device)
    if labels != []:
        evalulate(pred_scores, labels)

    # save
    if out_dir is not None:
        pred_scores = pred_scores.cpu().numpy()
        labels = labels.cpu().numpy()
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, 'pred_result.npz'),
            test_pids=test_pids,
            pred_scores=pred_scores,
            labels=labels
        )


if __name__ == '__main__':
    config = get_config()

    set_seed(config.seed)
    main(config, config.pred_out_dir)
