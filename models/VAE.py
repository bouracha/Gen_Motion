#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

from models.utils import *
from models.encoders import VAE_Encoder
from models.decoders import VAE_Decoder

import utils.viz_3d as viz_3d

from progress.bar import Bar
import time
from torch.autograd import Variable
from tqdm.auto import tqdm
import os
import sys
import numpy as np
import pandas as pd

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, encoder_layers=[48, 100, 50, 2],  decoder_layers = [2, 50, 100, 48], lr=0.001, train_batch_size=100, variational=False, output_variance=False, device="cuda", batch_norm=False, weight_decay=0.0, p_dropout=0.0, beta=1.0, start_epoch=1, folder_name=""):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VAE, self).__init__()
        self.n_x = encoder_layers[0]
        self.n_z = encoder_layers[-1]
        assert(self.n_x == decoder_layers[-1])
        assert(self.n_z == decoder_layers[0])
        self.variational = variational
        self.device = device
        self.beta = beta
        self.output_variance = output_variance
        self.folder_name= folder_name
        self.activation = nn.LeakyReLU(0.1)

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.encoder = VAE_Encoder(layers=encoder_layers, activation=self.activation, variational=variational, device=device, batch_norm=batch_norm, p_dropout=p_dropout)
        self.decoder = VAE_Decoder(layers=decoder_layers, activation=self.activation, output_variance=output_variance, device=device, batch_norm=batch_norm, p_dropout=p_dropout)

        if start_epoch==1:
            self.losses_file_exists = False
            self.book_keeping(encoder_layers, decoder_layers, start_epoch=start_epoch, lr=lr, train_batch_size=train_batch_size, batch_norm=batch_norm, weight_decay=weight_decay, p_dropout=p_dropout)
        else:
            self.losses_file_exists = True

    def forward(self, x):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        self.mu, self.z, self.KL = self.encoder(x)
        if self.output_variance:
            self.reconstructions_mu, self.reconstructions_log_var = self.decoder(self.z)
        else:
            self.reconstructions_mu = self.decoder(self.z)
            self.reconstructions_log_var = torch.zeros_like(self.reconstructions_mu)
        sig = nn.Sigmoid()
        self.bernoulli_output = sig(self.reconstructions_mu)

        return self.reconstructions_mu

    def generate(self, z):
        """

        :param z: batch of random variables
        :return: batch of generated samples
        """
        if self.output_variance:
            reconstructions_mu, _ = self.decoder(z)
        else:
            reconstructions_mu = self.decoder(z)
        return reconstructions_mu

    def loss_bernoulli(self, x):
        """
        :param x: batch of inputs
        :return
        """
        b_n, n_x = x.shape
        assert (n_x == self.n_x)

        loss_fn = nn.BCELoss()
        self.loss = loss_fn(self.bernoulli_output, x)
        BCE = torch.maximum(self.reconstructions_mu, torch.zeros_like(self.reconstructions_mu)) - torch.multiply(self.reconstructions_mu, x) + torch.log(1 + torch.exp(-torch.abs(self.reconstructions_mu)))
        BCE_per_sample = torch.sum(BCE, axis=1)
        self.recon_loss = torch.mean(BCE_per_sample)
        if self.variational:
            self.loss = self.recon_loss + self.KL
            self.VLB = -self.loss
        else:
            self.loss = self.recon_loss

        return self.loss


    def loss_VLB(self, x):
        """
        :param x: batch of inputs
        :return: loss of reconstructions
        """
        b_n, n_x = x.shape
        assert(n_x == self.n_x)

        self.mse = torch.pow((self.reconstructions_mu - x), 2)
        self.recon_loss = torch.mean(torch.sum(self.mse, axis=1))
        self.gauss_log_lik = -0.5*(self.reconstructions_log_var + np.log(2*np.pi) + (self.mse/(1e-8 + torch.exp(self.reconstructions_log_var))))
        self.gauss_log_lik = torch.mean(torch.sum(self.gauss_log_lik, axis=1))
        if self.variational:
            self.VLB = self.gauss_log_lik - self.beta*self.KL
            self.loss = -self.VLB
        else:
            self.loss = -self.gauss_log_lik
        return self.loss

    def accum_update(self, key, val):
        if key not in self.accum_loss.keys():
            self.accum_loss[key] = AccumLoss()
        val = val.cpu().data.numpy()
        self.accum_loss[key].update(val)

    def accum_reset(self):
        for key in self.accum_loss.keys():
            self.accum_loss[key].reset()

    def train_epoch(self, epoch, train_loader, optimizer):
        self.train()
        bar = Bar('>>>', fill='>', max=len(train_loader))
        st = time.time()
        for i, (all_seq) in enumerate(train_loader):
            bt = time.time()

            inputs = all_seq.to(self.device).float()

            mu = self(inputs.float())
            loss = self.loss_VLB(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                                  time.time() - st)
            bar.next()

        head = ['Epoch']
        ret_log = [epoch]
        self.head = np.append(self.head, head)
        self.ret_log = np.append(self.ret_log, ret_log)

        bar.finish()

    def train_epoch_mnist(self, epoch, train_loader, optimizer, use_bernoulli_loss=False):
        self.train()
        i=0
        for image, labels in tqdm(train_loader):
            i+=1
            cur_batch_size = len(image)

            image.to(self.device)
            image_flattened = image.reshape(cur_batch_size, -1)

            mu = self(image_flattened.float())
            if use_bernoulli_loss:
                loss = self.loss_bernoulli(image_flattened)
            else:
                loss = self.loss_VLB(image_flattened)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head = ['Epoch']
        ret_log = [epoch]
        self.head = np.append(self.head, head)
        self.ret_log = np.append(self.ret_log, ret_log)

    def eval_full_batch_mnist(self, loader, epoch, dataset_name='val', use_bernoulli_loss=False):
        self.eval()
        i=0
        for image, labels in tqdm(loader):
            i+=1
            cur_batch_size = len(image)

            image.to(self.device)
            image_flattened = image.reshape(cur_batch_size, -1)

            reconstructions = self(image_flattened.float())
            if use_bernoulli_loss:
                loss = self.loss_bernoulli(image_flattened)
                sig = nn.Sigmoid()
                reconstructions = sig(reconstructions)
            else:
                loss = self.loss_VLB(image_flattened)

            self.accum_update(str(dataset_name)+'_loss', loss)
            self.accum_update(str(dataset_name)+'_recon', self.recon_loss)
            if self.variational:
                self.accum_update(str(dataset_name)+'_VLB', self.VLB)
                self.accum_update(str(dataset_name)+'_KL', self.KL)

        head = [dataset_name+'_loss', dataset_name+'_reconstruction']
        ret_log = [self.accum_loss[str(dataset_name)+'_loss'].avg, self.accum_loss[str(dataset_name)+'_recon'].avg]
        if self.variational:
            head.append(str(dataset_name)+'_VLB')
            head.append(str(dataset_name)+'_KL')
            ret_log.append(self.accum_loss[str(dataset_name)+'_VLB'].avg)
            ret_log.append(self.accum_loss[str(dataset_name)+'_KL'].avg)
        self.head = np.append(self.head, head)
        self.ret_log = np.append(self.ret_log, ret_log)

        reconstructions = reconstructions.reshape(cur_batch_size, 1, 28, 28)
        file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reals'
        self.plot_tensor_images(image, num_images=25, nrow=5, show=False, save_as=file_path)
        file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reconstructions'
        self.plot_tensor_images(reconstructions, num_images=25, nrow=5, show=False, save_as=file_path)

    def eval_full_batch(self, loader, epoch, dataset_name='val'):
        self.eval()
        bar = Bar('>>>', fill='>', max=len(loader))
        st = time.time()
        for i, (all_seq) in enumerate(loader):
            bt = time.time()
            cur_batch_size = len(all_seq)

            inputs = all_seq.to(self.device).float()

            mu = self(inputs.float())
            loss = self.loss_VLB(inputs)

            self.accum_update(str(dataset_name)+'_loss', loss)
            self.accum_update(str(dataset_name)+'_recon_loss', self.recon_loss)
            if self.variational:
                self.accum_update(str(dataset_name)+'_KL', self.KL)

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(loader), time.time() - bt,
                                                                                  time.time() - st)
            bar.next()
        bar.finish()

        head = [dataset_name+'_loss', dataset_name+'_reconstruction']
        ret_log = [self.accum_loss[str(dataset_name)+'_loss'].avg, self.accum_loss[str(dataset_name)+'_recon_loss'].avg]
        if self.variational:
            head.append(str(dataset_name)+'_KL')
            ret_log.append(self.accum_loss[str(dataset_name)+'_KL'].avg)
        self.head = np.append(self.head, head)
        self.ret_log = np.append(self.ret_log, ret_log)

        inputs_reshaped = inputs.reshape(cur_batch_size, 1, 12, 8)
        reconstructions = mu.reshape(cur_batch_size, 1, 12, 8)
        diffs = inputs_reshaped - reconstructions

        file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reals'
        self.plot_tensor_images(inputs_reshaped, num_images=25, nrow=5, show=False, save_as=file_path)
        file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reconstructions'
        self.plot_tensor_images(reconstructions, num_images=25, nrow=5, show=False, save_as=file_path)
        file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'diffs'
        self.plot_tensor_images(diffs, num_images=25, nrow=5, show=False, save_as=file_path)
        file_path = self.folder_name + '/poses/' + str(dataset_name) + '_' + str(epoch) + '_' + 'poses_xz'
        self.plot_poses(inputs, mu, num_images=25, azim=0, evl=90, save_as=file_path)
        file_path = self.folder_name + '/poses/' + str(dataset_name) + '_' + str(epoch) + '_' + 'poses_yz'
        self.plot_poses(inputs, mu, num_images=25, azim=0, evl=-0, save_as=file_path)
        file_path = self.folder_name + '/poses/' + str(dataset_name) + '_' + str(epoch) + '_' + 'poses_xy'
        self.plot_poses(inputs, mu, num_images=25, azim=90, evl=90, save_as=file_path)

    def book_keeping(self, encoder_layers, decoder_layers, start_epoch=1, lr=0.001, train_batch_size=100, batch_norm=False, weight_decay=0.0, p_dropout=0.0):
        self.accum_loss = dict()

        if self.variational:
            self.folder_name = self.folder_name+"_VAE"
        else:
            self.folder_name = self.folder_name+"_AE"
        for layer in encoder_layers:
            self.folder_name = self.folder_name+'_'+str(layer)
        for layer in decoder_layers:
            self.folder_name = self.folder_name+'_'+str(layer)
        if batch_norm:
            self.folder_name = self.folder_name+'_bn'
        if p_dropout != 0.0:
            self.folder_name = self.folder_name + '_p_drop='+str(p_dropout)
        if self.output_variance:
            self.folder_name = self.folder_name + '_model-var'
        if self.beta != 1.0:
            self.folder_name = self.folder_name + '_beta=' + str(self.beta)
        if start_epoch==1:
            os.makedirs(os.path.join(self.folder_name, 'checkpoints'))
            os.makedirs(os.path.join(self.folder_name, 'images'))
            os.makedirs(os.path.join(self.folder_name, 'poses'))
            write_type='w'
        else:
            write_type = 'a'

        original_stdout = sys.stdout
        with open(str(self.folder_name)+'/'+'architecture.txt', write_type) as f:
            sys.stdout = f
            if start_epoch:
                print(self)
            print("Start epoch:{}".format(start_epoch))
            print("Learning rate:{}".format(lr))
            print("Training batch size:{}".format(train_batch_size))
            print("BN:{}".format(batch_norm))
            print("Weight Decay (1e-4):{}".format(weight_decay))
            print("p_dropout:{}".format(p_dropout))
            print("Output variance:{}".format(self.output_variance))
            print("Beta(downweight of KL):{}".format(self.beta))
            print("Activation function:{}".format(self.activation))
            sys.stdout = original_stdout

        self.head = []
        self.ret_log = []

    def save_checkpoint_and_csv(self, epoch, lr, optimizer):
        df = pd.DataFrame(np.expand_dims(self.ret_log, axis=0))
        if self.losses_file_exists:
            with open(self.folder_name+'/'+'losses.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        else:
            df.to_csv(self.folder_name+'/'+'losses.csv', header=self.head, index=False)
            self.losses_file_exists = True
        state = {'epoch': epoch + 1,
                         'lr': lr,
                         'err':  self.accum_loss['train_loss'].avg,
                         'state_dict': self.state_dict(),
                         'optimizer': optimizer.state_dict()}
        if epoch % 10 == 0:
            print("Saving checkpoint....")
            file_path = self.folder_name + '/checkpoints/' + 'ckpt_' + str(epoch) + '_weights.path.tar'
            torch.save(state, file_path)
        self.head = []
        self.ret_log = []
        self.accum_reset()

    def plot_tensor_images(self, image_tensor, num_images=25, nrow=5, show=False, save_as=None):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        if show:
            plt.show()
        if not save_as==None:
            plt.savefig(save_as)
        plt.close()

    def plot_poses(self, xyz_gt, xyz_pred, num_images=25, azim=0, evl=0, save_as=None):
        '''
        Function for visualizing poses: saves grid of poses.
        Assumes poses are normalised between 0 and 1
        :param xyz_gt: set of ground truth 3D joint positions (batch_size, 96)
        :param xyz_pred: set of predicted 3D joint positions (batch_size, 96)
        :param num_images: number of poses to plotted from given set (int)
        :param azim: azimuthal angle for viewing (int)
        :param evl: angle of elevation for viewing (int)
        :param save_as: path and name to save (str)
        '''
        xyz_gt = xyz_gt.detach().cpu().numpy()
        xyz_pred = xyz_pred.detach().cpu().numpy()

        xyz_gt = xyz_gt[:num_images].reshape(num_images, 32, 3)
        xyz_pred = xyz_pred[:num_images].reshape(num_images, 32, 3)

        fig = plt.figure()
        if num_images > 4:
            fig = plt.figure(figsize=(20, 20))

        grid_dim_size = np.ceil(np.sqrt(num_images))
        for i in range(num_images):
            ax = fig.add_subplot(grid_dim_size, grid_dim_size, i+1, projection='3d')
            ax.set_xlim3d([0, 1])
            ax.set_ylim3d([0, 1])
            ax.set_zlim3d([0, 1])

            ob = viz_3d.Ax3DPose(ax)
            ob.update(xyz_gt[i], xyz_pred[i])

            ob.ax.set_axis_off()
            ob.ax.view_init(azim, evl)

        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0)

        plt.savefig(save_as)
        plt.close()








