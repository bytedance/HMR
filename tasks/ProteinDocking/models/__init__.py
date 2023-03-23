"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import logging
import importlib

model_list = []

def register_model(model):
    model_list.append(model)

def load_model(name):
    mdict = {model.__name__: model for model in model_list}
    if name not in mdict:
        logging.info(f"Invalid model index. You put {name}. Options are:")
        for model in model_list:
            logging.info("\t* {}".format(model.__name__))
        return None
    NetClass = mdict[name]
    return NetClass

modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        module_name = file[:file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("models." + module_name)


