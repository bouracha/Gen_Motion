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




actions = data_utils.define_actions('all', 'h3.6m', out_of_distribution=False)

input_n = 10
output_n = 25
dct_n = 35
sample_rate = 2

cartesian = False
node_n = 48



model = nnmodel.GCN(input_feature=dct_n, hidden_feature=256, p_dropout=0.3,
                        num_stage=12, node_n=node_n, variational=True, n_z=8, num_decoder_stage=6)
if is_cuda:
    model.cuda()
print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
#optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)


model_path_len = 'checkpoint/test/ckpt_main_h3.6m_in10_out10_dctn20_dropout_0.0_var_lambda_0.003_nz_8_lr_0.0005_n_layers_6_last.pth.tar'
print(">>> loading ckpt len from '{}'".format(model_path_len))
if is_cuda:
    ckpt = torch.load(model_path_len)
else:
    ckpt = torch.load(model_path_len, map_location='cpu')

batch_size=128
train_data = dict()
test_data = dict()
for act in actions:
    print("Loading action {} for train set".format(act))
    train_dataset = H36motion(path_to_data='h3.6m/', actions=[act], input_n=input_n, output_n=output_n,
                               sample_rate=sample_rate, split=0, dct_n=dct_n)
    #print(train_dataset.__len__())
    data_std = train_dataset.data_std
    data_mean = train_dataset.data_mean
    dim_used = train_dataset.dim_used
    train_data[act] = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True)
    print("Loading action {} for test set".format(act))
    test_dataset = H36motion(path_to_data='h3.6m/', actions=[act], input_n=input_n, output_n=output_n,
                              sample_rate=sample_rate, split=1, data_mean=data_mean, data_std=data_std, dct_n=dct_n)
    #print(test_dataset.__len__())
    test_data[act] = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True)



if output_n >= 25:
    eval_frame = [1, 3, 7, 9, 13, 24]
elif output_n == 10:
    eval_frame = [1, 3, 7, 9]


for act in actions:
    new_action = True
    for i, (inputs, targets, all_seq) in enumerate(train_data[act]):
        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()

        outputs, reconstructions, log_var, z = model(inputs.float())

        print("For action {} the train z is: {}".format(act, z.shape))
        for sample in z:
          sample = sample.reshape(384).cpu().detach().numpy()
          #print(sample.shape)
          df = pd.DataFrame(np.expand_dims(sample, axis=0))
          if new_action:
            df.to_csv('latents/'+str(act)+'_train_z.csv', header=False, index=False)
            new_action = False
          else:
            with open('latents/'+str(act)+'_train_z.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)

    new_action = True
    for i, (inputs, targets, all_seq) in enumerate(test_data[act]):
        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(non_blocking=True)).float()

        outputs, reconstructions, log_var, z = model(inputs.float())

        print("For action {} the test z is: {}".format(act, z.shape))
        for sample in z:
          sample = sample.reshape(384).cpu().detach().numpy()
          #print(sample.shape)
          df = pd.DataFrame(np.expand_dims(sample, axis=0))
          if new_action:
            df.to_csv('latents/'+str(act)+'_test_z.csv', header=False, index=False)
            new_action = False
          else:
            with open('latents/'+str(act)+'_test_z.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)




