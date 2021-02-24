from __future__ import print_function, absolute_import, division

import torch.optim

from scipy.stats import norm
import numpy as np

import models.VAE as nnmodel
import models.utils as utils

from opt import Options
opt = Options().parse()

folder_name=opt.name
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = "cuda"
    job = 10 #Can change this to multi-process dataloading
else:
    device = "cpu"
    job = 0

##################################################################
# Instantiate model, and methods used for inference
##################################################################
start_epoch = opt.start_epoch

model = nnmodel.VAE(input_n=96, encoder_hidden_layers=opt.encoder_hidden_layers, n_z=opt.n_z, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=opt.batch_norm, p_dropout=0.0)
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