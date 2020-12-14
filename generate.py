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

import models.VAE as nnmodel

import utils.viz as viz
from matplotlib import pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='ckpt_49_weights.path.tar', help='path to saved model')
opt = parser.parse_args()

is_cuda = torch.cuda.is_available()
n_z = 2

print(">>> creating model")
model = nnmodel.VAE(encoder_layers=[48, 100, 50, n_z],  decoder_layers = [n_z, 50, 100, 48], variational=True, output_variance=True, device="cuda", batch_norm=True, p_dropout=0.0, beta=1.0)
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



all_seq = torch.tensor([[[ 3.0088e+00,  2.3565e+00,  1.2685e+01,  9.0479e-03, -2.3448e-02,
           3.0427e-03,  1.8167e-01,  6.5091e-02, -3.8717e-02, -1.0212e+00,
          -0.0000e+00, -0.0000e+00, -5.2310e-01, -1.0842e-01, -2.3475e-01,
           9.2063e-01, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          -0.0000e+00,  2.3131e-01, -9.5584e-02,  1.1178e-01, -1.9508e-01,
          -0.0000e+00, -0.0000e+00, -4.6369e-02, -2.4654e-01,  6.8490e-02,
           5.9577e-01, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          -0.0000e+00,  1.1304e-01, -5.4122e-02,  5.0800e-03,  1.0436e-01,
          -3.3177e-02, -1.4137e-01, -1.0382e+00, -2.9327e-01,  3.3821e-01,
           1.4437e+00,  6.4356e-02, -1.9727e-01, -0.0000e+00, -0.0000e+00,
          -0.0000e+00, -6.9275e-02, -2.1161e-01,  2.1942e+00, -9.6483e-02,
          -2.9132e-01,  7.7537e-01, -2.2463e-01, -0.0000e+00, -0.0000e+00,
           2.1930e-01, -3.6066e-02, -2.8463e-02, -0.0000e+00, -0.0000e+00,
          -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          -2.4333e-01,  2.3345e-01, -2.0615e+00,  1.0563e-01, -1.7902e-01,
          -7.2105e-01, -2.1639e-01, -0.0000e+00, -0.0000e+00,  1.8257e-01,
           3.8653e-01,  6.5862e-02, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00]]])
all_seq = all_seq.cuda()
dim_used = [6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86]



model.eval()

#print(y)
n_samples = 128
noise = torch.normal(mean=0, std=1.0, size=(n_samples, n_z)).to(torch.device("cuda"))

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
    z = noise[i, :]
    z = z.reshape(1, n_z)

    y = model.generate(z)

    outputs_t = y.view(-1, 1).transpose(0, 1)

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



