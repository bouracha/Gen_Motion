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

import utils.data_utils as data_utils

from data import DATA

import utils.vgae as nnmodel


is_cuda = torch.cuda.is_available()

#####################################################
# Load data
#####################################################
data = DATA("h3.6m", "h3.6m/dataset/")
out_of_distribution = data.get_dct_and_sequences(input_n=1, output_n=1, sample_rate=2, dct_n=2, out_of_distribution_action=None)
train_loader, val_loader, OoD_val_loader, test_loaders = data.get_dataloaders(train_batch=16, test_batch=128, job=10)
print(">>> data loaded !")
print(">>> train data {}".format(data.train_dataset.__len__()))
print(">>> validation data {}".format(data.val_dataset.__len__()))

##################################################################
# Instantiate model, and methods used fro training and valdation
##################################################################
print(">>> creating model")
model = nnmodel.VGAE(input_feature=1, hidden_feature=256, p_dropout=0,
                        num_stage=12, node_n=data.node_n, n_z=1)
if is_cuda:
    model.cuda()

print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


for epoch in range(0, 5):

    bar = Bar('>>>', fill='>', max=len(train_loader))
    st = time.time()
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        sequences = all_seq[:, :, data.train_dataset.dim_used]
        b, l, n = sequences.shape
        inputs = sequences.view(b, n, l)
        inputs = sequences[:, 0, :]
        inputs = inputs.reshape((b, n, 1))

        model.train()
        if is_cuda:
          inputs = Variable(inputs.cuda()).float()

        mu, log_var, z, KL = model(inputs.float())

        loss = model.loss_VLB(inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                              time.time() - st)
        bar.next()
    bar.finish()
    print("Train: ")
    print("loss", loss)
    print("neg_gauss_log_lik", model.neg_gauss_log_lik)
    print("latent_loss", model.latent_loss)

    for i, (inputs, targets, all_seq) in enumerate(val_loader):
        bt = time.time()

        sequences = all_seq[:, :, data.train_dataset.dim_used]
        b, l, n = sequences.shape
        inputs = sequences.view(b, n, l)
        inputs = sequences[:, 0, :]
        inputs = inputs.reshape((b, n, 1))

        if is_cuda:
          inputs = Variable(inputs.cuda()).float()

        mu, log_var, z, KL = model(inputs.float())

        loss = model.loss_VLB(inputs)

    print("Val: ")
    print("loss", loss)
    print("neg_gauss_log_lik", model.neg_gauss_log_lik)
    print("latent_loss", model.latent_loss)





