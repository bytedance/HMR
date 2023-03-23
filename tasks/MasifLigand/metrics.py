"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import torch
from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc
from sklearn.metrics import balanced_accuracy_score
import numpy as np


def multi_class_eval(scores, labels, K):

    with torch.no_grad():
        accuracy_macro = accuracy(preds=scores, target=labels, average='macro', task='multiclass', num_classes=K).item()
        accuracy_micro = accuracy(preds=scores, target=labels, average='micro', task='multiclass', num_classes=K).item()
        preds = np.argmax(scores.cpu().numpy(), axis=1)
        accuracy_balanced = balanced_accuracy_score(y_true=labels.cpu().numpy(), y_pred=preds)

        precision_macro = precision(preds=scores, target=labels, average='macro', task='multiclass', num_classes=K).item()
        precision_micro = precision(preds=scores, target=labels, average='micro', task='multiclass', num_classes=K).item()

        recall_macro = recall(preds=scores, target=labels, average='macro', task='multiclass', num_classes=K).item()
        recall_micro = recall(preds=scores, target=labels, average='micro', task='multiclass', num_classes=K).item()

        f1_macro = f1_score(preds=scores, target=labels, average='macro', task='multiclass', num_classes=K).item()
        f1_micro = f1_score(preds=scores, target=labels, average='micro', task='multiclass', num_classes=K).item()

        auroc_macro = auroc(preds=scores, target=labels, average='macro', task='multiclass', num_classes=K).item()
        
    return (
        accuracy_macro, accuracy_micro, accuracy_balanced,
        precision_macro, precision_micro, 
        recall_macro, recall_micro, 
        f1_macro, f1_micro,
        auroc_macro, 
    )
    