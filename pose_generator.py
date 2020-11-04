from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion import H36motion
from utils.cmu_motion import CMU_Motion
from utils.cmu_motion_3d import CMU_Motion3D
import utils.model as nnmodel
import utils.data_utils as data_utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='h3.6m', help='which dataset to use')
parser.add_argument('--model_path', type=str, default=None, help='path to checkpoint')

opt = parser.parse_args()

is_cuda = torch.cuda.is_available()

#Define actions
acts_train = data_utils.define_actions('all', opt.dataset, out_of_distribution=False)
acts_test = data_utils.define_actions('all', opt.dataset, out_of_distribution=False)

#Read data into input_dct output_dct, all_sequences
train_dataset = H36motion(path_to_data=opt.data_dir, actions=acts_train, input_n=input_n, output_n=output_n,
                                split=0, sample_rate=sample_rate, dct_n=dct_n)
data_std = train_dataset.data_std
data_mean = train_dataset.data_mean
val_dataset = H36motion(path_to_data=opt.data_dir, actions=acts_train, input_n=input_n, output_n=output_n,
                        split=2, sample_rate=sample_rate, data_mean=data_mean, data_std=data_std, dct_n=dct_n)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=opt.train_batch,
    shuffle=True,
    num_workers=opt.job,
    pin_memory=True)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=opt.test_batch,
    shuffle=False,
    num_workers=opt.job,
    pin_memory=True)
test_data = dict()
for act in acts_test:
    test_dataset = H36motion(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n, split=1,
                             sample_rate=sample_rate, data_mean=data_mean, data_std=data_std, dct_n=dct_n)
    test_data[act] = DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)







