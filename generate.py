from __future__ import print_function, absolute_import, division

import torch.optim

from scipy.stats import norm
import numpy as np

import models.VAE as nnmodel

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

encoder_layers = []
decoder_layers = []
encoder_layers.append(input_n)
decoder_layers.append(n_z)
n_hidden = len(encoder_hidden_layers)
for i in range(n_hidden):
    encoder_layers.append(encoder_hidden_layers[i])
    decoder_layers.append(encoder_hidden_layers[n_hidden-1-i])
encoder_layers.append(n_z)
decoder_layers.append(input_n)

print(">>> creating model")
model = nnmodel.VAE(encoder_layers=encoder_layers,  decoder_layers=decoder_layers, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=batch_norm, p_dropout=0.0, beta=opt.beta, start_epoch=start_epoch, folder_name=folder_name)
clipping_value = 1
torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
if is_cuda:
    model.cuda()
print(model)

print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
if start_epoch != 1:
    model.book_keeping(model.encoder_layers, model.decoder_layers, start_epoch=start_epoch, batch_norm=batch_norm, p_dropout=0.0)
    ckpt_path = model.folder_name + '/checkpoints/' + 'ckpt_' + str(start_epoch-1) + '_weights.path.tar'
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])

num_grid_points = opt.grid_size

z = torch.randn(num_grid_points**2, 2)
linspace = np.linspace(0.01, 0.99, num=num_grid_points)
count=0
for i in linspace:
  for j in linspace:
      z[count, 0] = j
      z[count, 1] = i
      count += 1

inputs = z.to(device).float()
mu = model.generate(inputs.float())


file_path = folder_name + '_VAE/'+str(start_epoch) + '_' + 'poses_xz'
model.plot_poses(mu, mu, num_images=num_grid_points**2, azim=0, evl=90, save_as=file_path)
file_path = folder_name + '_VAE/' + str(start_epoch) + '_' + 'poses_yz'
model.plot_poses(mu, mu, num_images=num_grid_points**2, azim=0, evl=-0, save_as=file_path)
file_path = folder_name + '_VAE/' + str(start_epoch) + '_' + 'poses_xy'
model.plot_poses(mu, mu, num_images=num_grid_points**2, azim=90, evl=90, save_as=file_path)