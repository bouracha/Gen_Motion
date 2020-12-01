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

from utils import utils as utils
from utils.opt import Options
from utils.h36motion import H36motion
from utils.cmu_motion import CMU_Motion
from utils.cmu_motion_3d import CMU_Motion3D

import utils.data_utils as data_utils

from data import DATA

import utils.vgae as nnmodel

import utils.viz as viz
from matplotlib import pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='ckpt_49_weights.path.tar', help='path to saved model')
opt = parser.parse_args()

is_cuda = torch.cuda.is_available()

print(">>> creating model")
model = nnmodel.VGAE(input_feature=1, hidden_feature=256, p_dropout=0,
                        num_stage=1, node_n=48, n_z=1)
print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
lr=0.00003
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if is_cuda:
    model.cuda()
    ckpt = torch.load(opt.ckpt)
start_epoch = ckpt['epoch']
err_best = ckpt['err']
lr_now = ckpt['lr']
model.load_state_dict(ckpt['state_dict'])
optimizer.load_state_dict(ckpt['optimizer'])





print(">>> loading data")
acts = data_utils.define_actions('all')
test_data = dict()
for act in acts:
    test_dataset = H36motion(path_to_data='h3.6m/dataset/', actions=act, input_n=10, output_n=10, split=1,
                                 sample_rate=2)
    test_data[act] = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True)
dim_used = test_dataset.dim_used
print(">>> data loaded !")
for i, (inputs, targets, all_seq) in enumerate(test_data['walking']):
    inputs = Variable(inputs).float()
    all_seq = Variable(all_seq).float()
    if is_cuda:
        inputs = inputs.cuda()
        all_seq = all_seq.cuda()
    all_seq = all_seq[:, 0, :]
    all_seq = all_seq.reshape((1, 1, 99))
    #print(all_seq.shape)
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)



model.eval()

#print(y)
n_samples = 128
n_z = 1
noise = torch.normal(mean=0, std=1.0, size=(n_samples, 48, n_z)).to(torch.device("cuda"))

#noise_const = torch.normal(mean=0, std=1.0, size=(1, 47, n_z)).to(torch.device("cuda"))
#noise_const = noise_const.repeat(n_samples, 1, 1)
#print(noise_const.shape)
#noise_var = torch.normal(mean=0, std=1.0, size=(n_samples, 1, n_z)).to(torch.device("cuda"))
#val = -3
#for i in range(n_samples):
#    noise_var[0] = val
#    val += (6.0/128)
#noise = torch.cat((noise_const, noise_var), 1)
#print(noise.shape)


for i in range(0, n_samples):
    #print(i)
    z = noise[i, :, :]
    z = z.reshape(1, 48, n_z)

    y = model.generate(z)

    outputs_t = y.view(-1, seq_len).transpose(0, 1)

    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_t
    targ_expmap = all_seq
    pred_expmap = pred_expmap.cpu().data.numpy()
    targ_expmap = targ_expmap.cpu().data.numpy()

    fig = plt.figure()
    ax = plt.gca(projection='3d')

    for k in range(1):
      plt.cla()
      figure_title = "action:{}, seq:{},".format('pose', (i + 1))
      viz.plot_predictions(pred_expmap[k, :, :], pred_expmap[k, :, :], fig, ax, figure_title)



