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

parser.add_argument('--num_trials', type=int, default='10',
                    help='number of trials of randomly selected hyperparameters')
parser.add_argument('--variational', type=bool, default=False, help='true if want to include a latent variable')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')

opt = parser.parse_args()


is_cuda = torch.cuda.is_available()




actions = data_utils.define_actions('all', 'cmu_mocap', out_of_distribution=False)

input_n = 10
output_n = 25
dct_n = 35
sample_rate = 2

cartesian = False
node_n = 64



model = nnmodel.GCN(input_feature=dct_n, hidden_feature=256, p_dropout=0.3,
                        num_stage=12, node_n=node_n, variational=True, n_z=8, num_decoder_stage=6)
if is_cuda:
    model.cuda()
print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)


model_path_len = 'checkpoint/test/ckpt_main_cmu_mocap_in10_out25_dctn35_dropout_0.3_var_lambda_0.003_nz_8_lr_0.0005_n_layers_6_best.pth.tar'
print(">>> loading ckpt len from '{}'".format(model_path_len))
if is_cuda:
    ckpt = torch.load(model_path_len)
else:
    ckpt = torch.load(model_path_len, map_location='cpu')


train_data = dict()
test_data = dict()
for act in actions:
    train_data[act] = CMU_Motion(path_to_data=opt.data_dir, actions=[act], input_n=input_n, output_n=output_n,
                               split=0, dct_n=dct_n)
    data_std = train_data[act].data_std
    data_mean = train_data[act].data_mean
    dim_used = train_data[act].dim_used
    train_data = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    test_data = CMU_Motion(path_to_data=opt.data_dir, actions=[act], input_n=input_n, output_n=output_n,
                              split=1, data_mean=data_mean, data_std=data_std, dim_used=dim_used, dct_n=dct_n)
    test_data[act] = DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)



if output_n >= 25:
    eval_frame = [1, 3, 7, 9, 13, 24]
elif output_n == 10:
    eval_frame = [1, 3, 7, 9]


for act in actions:
    for i, (inputs, targets, all_seq) in enumerate(train_data[act]):
        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()

        outputs, reconstructions, log_var, z = model(inputs.float())

        print("For action {} the train z is: ".format(act, z))

    for i, (inputs, targets, all_seq) in enumerate(test_data[act]):
        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()

        outputs, reconstructions, log_var, z = model(inputs.float())

        print("For action {} the test z is: ".format(act, z))
