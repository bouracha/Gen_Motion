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

from data import DATA

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='h3.6m', help='which dataset to use')
parser.add_argument('--model_path', type=str, default=None, help='path to checkpoint')

opt = parser.parse_args()

is_cuda = torch.cuda.is_available()

#####################################################
# Load data
#####################################################

data = DATA(opt.dataset, opt.data_dir)
out_of_distribution = data.get_dct_and_sequences(input_n=1, output_n=1, sample_rate=2, dct_n=2, out_of_distribution=None)
train_loader, val_loader, OoD_val_loader, test_loaders = data.get_dataloaders(train_batch=16, test_batch=128, job=10)
print(">>> data loaded !")
print(">>> train data {}".format(data.train_dataset.__len__()))
if opt.dataset == 'h3.6m':
    print(">>> validation data {}".format(data.val_dataset.__len__()))

in_dct, out_dct, all_seq = data.train_dataset
print(all_seq.shape)






