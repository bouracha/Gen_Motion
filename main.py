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


is_cuda = torch.cuda.is_available()

#####################################################
# Load data
#####################################################
data = DATA("h3.6m", "h3.6m/dataset/")
out_of_distribution = data.get_poses(input_n=1, output_n=1, sample_rate=2, dct_n=2, out_of_distribution_action=None)
train_loader, val_loader, OoD_val_loader, test_loaders = data.get_dataloaders(train_batch=16, test_batch=128, job=10)
print(">>> data loaded !")
print(">>> train data {}".format(data.train_dataset.__len__()))
print(">>> validation data {}".format(data.val_dataset.__len__()))

##################################################################
# Instantiate model, and methods used fro training and valdation
##################################################################
print(">>> creating model")
model = nnmodel.VAE(encoder_layers=[48, 100, 50, 2],  decoder_layers = [2, 50, 100, 48], variational=True, device="cuda")
clipping_value = 1
torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
if is_cuda:
    model.cuda()
print(model)

print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
lr=0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for epoch in range(0, 50):
    print("Epoch: ", epoch+1)

    model.train_epoch(train_loader, optimizer)

    model.eval()
    for i, (all_seq) in enumerate(val_loader):
        bt = time.time()

        inputs = all_seq

        if is_cuda:
          inputs = Variable(inputs.cuda()).float()

        mu = model(inputs.float())

        loss = model.loss_VLB(inputs)

        model.accum_update('val_loss', loss)
        model.accum_update('val_neg_gauss_log_lik', model.neg_gauss_log_lik)
        model.accum_update('val_KL', model.KL)

    print("Val: ")
    print("loss", model.accum_loss['val_loss'].avg)
    print("neg_gauss_log_lik", model.accum_loss['val_neg_gauss_log_lik'].avg)
    print("KL", model.accum_loss['val_KL'].avg)

    model.accum_reset()

    state = {'epoch': epoch + 1,
                         'lr': lr,
                         'err':  model.accum_loss['val_loss'].avg,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
    file_path = 'ckpt_' + str(epoch) + '_weights.path.tar'
    torch.save(state, file_path)





