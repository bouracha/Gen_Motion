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

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--variational', dest='variational', action='store_true', help='toggle VAE or AE')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='toggle use batch_norm or not')
parser.add_argument('--output_variance', dest='output_variance', action='store_true', help='toggle model output variance or use as constant')
parser.add_argument('--use_MNIST', dest='use_MNIST', action='store_true', help='toggle to use MNIST data instead')
parser.add_argument('--use_bernoulli_loss', dest='use_bernoulli_loss', action='store_true', help='toggle to bernoulli of gauss loss')
parser.add_argument('--beta', type=float, default=1.0, help='Downweighting of the KL divergence')
parser.set_defaults(variational=False)
parser.set_defaults(batch_norm=False)
parser.set_defaults(output_variance=False)
parser.set_defaults(use_MNIST=False)
parser.set_defaults(use_bernoulli_loss=False)

opt = parser.parse_args()

folder_name=""
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = "cuda"
    job = 10 #Can change this to multi-process dataloading
else:
    device = "cpu"
    job = 0

#####################################################
# Load data
#####################################################
train_batch=100
test_batch=128
if opt.use_MNIST:
    ### MNIST
    folder_name = folder_name+"MNIST"
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        MNIST('.', train=True, download=False, transform=transform),
        batch_size=train_batch,
        shuffle=True)
    val_loader = DataLoader(
        MNIST('.', train=False, download=False, transform=transform),
        batch_size=train_batch,
        shuffle=True)
else:
    ### Human Motion Data
    data = DATA("h3.6m_3d", "h3.6m/dataset/")
    out_of_distribution = data.get_poses(input_n=1, output_n=1, sample_rate=2, dct_n=2, out_of_distribution_action=None)
    train_loader, val_loader, OoD_val_loader, test_loader = data.get_dataloaders(train_batch=train_batch, test_batch=test_batch, job=job)
    print(">>> train data {}".format(data.train_dataset.__len__()))
    print(">>> validation data {}".format(data.val_dataset.__len__()))

print(">>> data loaded !")
##################################################################
# Instantiate model, and methods used fro training and valdation
##################################################################
n_zs = [2, 3, 5, 10, 20, 48]
input_n = data.node_n
#n_z = 2

for n_z in n_zs:
    print(">>> creating model")
    model = nnmodel.VAE(encoder_layers=[input_n, 100, n_z],  decoder_layers = [n_z, 100, input_n], variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=opt.batch_norm, p_dropout=0.0, beta=opt.beta, new_model=True, folder_name=folder_name)
    clipping_value = 1
    torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
    if is_cuda:
        model.cuda()
    print(model)

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    lr=0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for epoch in range(0, 50):
        print("Epoch: ", epoch+1)

        if opt.use_MNIST:
            model.train_epoch_mnist(epoch, train_loader, optimizer, opt.use_bernoulli_loss)
            model.eval_full_batch_mnist(train_loader, epoch, 'train', opt.use_bernoulli_loss)
            model.eval_full_batch_mnist(val_loader, epoch, 'val', opt.use_bernoulli_loss)
        else:
            model.train_epoch(epoch, train_loader, optimizer)
            model.eval_full_batch(train_loader, epoch, 'train')
            model.eval_full_batch(val_loader, epoch, 'val')

        model.save_checkpoint_and_csv(epoch, lr, optimizer)







