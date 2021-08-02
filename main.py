from __future__ import print_function, absolute_import, division

import torch.optim

import data as data

import train as train
import models.utils as model_utils

from opt import Options
opt = Options().parse()

import experiments.utils as experiment_utils
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

l2_reg = 1e-4

folder_name=opt.name
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = "cuda"
else:
    device = "cpu"

# ===============================================================
# Load data
# ===============================================================
train_batch_size=opt.train_batch_size
test_batch_size=opt.test_batch_size
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
        batch_size=train_batch_size,
        shuffle=True)
    val_loader = DataLoader(
        MNIST('.', train=False, download=False, transform=transform),
        batch_size=train_batch_size,
        shuffle=True)
    input_n=784
elif opt.motion:
    ### Human Motion Data
    data = data.DATA("h3.6m_3d", "h3.6m/dataset/")
    timepoints = opt.timepoints
    out_of_distribution = data.get_dct_and_sequences(input_n=timepoints, output_n=0, sample_rate=opt.sample_rate, dct_n=10, out_of_distribution_action=None)
    train_loader, val_loader, OoD_val_loader, test_loader = data.get_dataloaders(train_batch=train_batch_size, test_batch=test_batch_size)
    #input_n=data.node_n*timepoints
    #input_n=96*timepoints
    input_n=[96, timepoints]
else:
    ### Human PoseData
    data = data.DATA("h3.6m_3d", "h3.6m/dataset/")
    out_of_distribution = data.get_poses(input_n=1, output_n=1, sample_rate=opt.sample_rate, dct_n=2, out_of_distribution_action=None)
    train_loader, val_loader, OoD_val_loader, test_loader = data.get_dataloaders(train_batch=train_batch_size, test_batch=test_batch_size)
    input_n=data.node_n
print(">>> data loaded !")
# ===============================================================
# Instantiate model, and methods used fro training and valdation
# ===============================================================

#import models.VAE as nnmodel
#model = nnmodel.VAE(input_n=input_n, hidden_layers=opt.hidden_layers,  n_z=opt.n_z, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=opt.batch_norm, p_dropout=opt.p_drop)
import models.VDVAE as nnmodel
model = nnmodel.VDVAE(input_n=input_n, variational=opt.variational, output_variance=opt.output_variance, device=device, batch_norm=opt.batch_norm, p_dropout=opt.p_drop, n_zs=opt.n_zs, residual_size=opt.highway_size, gen_disc=opt.gen_disc)
train.initialise(model, start_epoch=opt.start_epoch, folder_name=folder_name, lr=opt.lr, beta=opt.beta, l2_reg=l2_reg, train_batch_size=train_batch_size, warmup_time=opt.warmup_time, beta_final=opt.beta_final)

for epoch in range(opt.start_epoch, opt.n_epochs+1):
    print("Epoch:{}/{}".format(epoch, opt.n_epochs))
    model.epoch_cur = epoch

    if opt.use_MNIST:
        train.train_epoch_mnist(model, train_loader, opt.use_bernoulli_loss)
        train.eval_full_batch_mnist(model, train_loader, 'train', opt.use_bernoulli_loss)
        train.eval_full_batch_mnist(model, val_loader, 'val', opt.use_bernoulli_loss)
    elif opt.motion:
        use_dct = True
        train.train_motion_epoch(model, train_loader, use_dct=use_dct, gen_disc=opt.gen_disc)
        train.eval_motion_batch(model, val_loader, 'train', use_dct=use_dct, gen_disc=opt.gen_disc)
        train.eval_motion_batch(model, val_loader, 'val', use_dct=use_dct, gen_disc=opt.gen_disc)
    else:
        train.train_epoch(model, train_loader)
        train.eval_full_batch(model, train_loader, 'train')
        train.eval_full_batch(model, val_loader, 'val')

    if model.epoch_cur % model.figs_checkpoints_save_freq == 0:
        #experiment_utils.generate_samples(model, num_grid_points=20, use_bernoulli_loss=opt.use_bernoulli_loss)
        experiment_utils.generate_motion_frames(model, use_bernoulli_loss=opt.use_bernoulli_loss, gen_disc=opt.gen_disc)
        experiment_utils.generate_motion_samples(model, use_bernoulli_loss=opt.use_bernoulli_loss, gen_disc=opt.gen_disc)
        experiment_utils.generate_motion_samples_resolution(model, use_bernoulli_loss=opt.use_bernoulli_loss, gen_disc=opt.gen_disc)

    model_utils.save_checkpoint_and_csv(model)
    model.writer.close()






