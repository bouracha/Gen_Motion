from __future__ import print_function, absolute_import, division

import torch.optim

from scipy.stats import norm
import numpy as np

import models.VAE as nnmodel
import models.utils as utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--variational', dest='variational', action='store_true', help='toggle VAE or AE')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='toggle use batch_norm or not')
parser.add_argument('--weight_decay', dest='weight_decay', action='store_true', help='toggle use weight decay or not')
parser.add_argument('--output_variance', dest='output_variance', action='store_true', help='toggle model output variance or use as constant')
parser.add_argument('--use_MNIST', dest='use_MNIST', action='store_true', help='toggle to use MNIST data instead')
parser.add_argument('--use_bernoulli_loss', dest='use_bernoulli_loss', action='store_true', help='toggle to bernoulli of gauss loss')
parser.add_argument('--beta', type=float, default=1.0, help='Downweighting of the KL divergence')
parser.add_argument('--n_z', type=int, default=2, help='Number of latent variables')
parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=1, help='If not 1, load checkpoint at this epoch')
parser.add_argument('--grid_size', type=int, default=10, help='Size of grid in each dimension')
parser.add_argument('--name', type=str, default='deep_2_model-var', help='If not 1, load checkpoint at this epoch')
parser.add_argument('--encoder_hidden_layers', nargs='+', type=int, default=[500, 200, 100, 50], help='input the out of distribution action')
parser.set_defaults(variational=False)
parser.set_defaults(batch_norm=False)
parser.set_defaults(weight_decay=False)
parser.set_defaults(output_variance=False)
parser.set_defaults(use_MNIST=False)
parser.set_defaults(use_bernoulli_loss=False)

opt = parser.parse_args()

folder_name=opt.name
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = "cuda"
    job = 10 #Can change this to multi-process dataloading
else:
    device = "cpu"
    job = 0

##################################################################
# Instantiate model, and methods used fro training and valdation
##################################################################
input_n = 96 #data.node_n
n_z = opt.n_z
n_epochs = opt.n_epochs
encoder_hidden_layers = opt.encoder_hidden_layers
start_epoch = opt.start_epoch
batch_norm = opt.batch_norm


model = nnmodel.VAE(input_n=96, encoder_hidden_layers=encoder_hidden_layers, n_z=opt.n_z, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=batch_norm, p_dropout=0.0)
model.initialise(start_epoch=start_epoch, folder_name=folder_name, lr=0.0001, beta=opt.beta, l2_reg=opt.weight_decay, train_batch_size=100)
model.eval()

num_grid_points = opt.grid_size

z = np.random.randn(num_grid_points**2, 2)
linspace = np.linspace(0.01, 0.99, num=num_grid_points)
count=0
for i in linspace:
  for j in linspace:
      z[count, 0] = j
      z[count, 1] = i
      count += 1

z = norm.ppf(z)
inputs = torch.from_numpy(z).to(device)
mu = model.generate(inputs.float())

file_path = model.folder_name + '/' + str(start_epoch) + '_' + 'poses_xz'
utils.plot_poses(mu.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=num_grid_points**2, azim=0, evl=90, save_as=file_path)
file_path = model.folder_name + '/' + str(start_epoch) + '_' + 'poses_yz'
utils.plot_poses(mu.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=num_grid_points**2, azim=0, evl=-0, save_as=file_path)
file_path = model.folder_name + '/' + str(start_epoch) + '_' + 'poses_xy'
utils.plot_poses(mu.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=num_grid_points**2, azim=90, evl=90, save_as=file_path)