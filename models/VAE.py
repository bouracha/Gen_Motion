#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

import models.utils as utils
from models.encoders import VAE_Encoder
from models.decoders import VAE_Decoder

from progress.bar import Bar
import time
from torch.autograd import Variable
from tqdm.auto import tqdm
import os
import sys
import numpy as np
import pandas as pd

class VAE(nn.Module):
    def __init__(self, input_n=96, encoder_hidden_layers=[100, 50], n_z=2, act_fn=nn.LeakyReLU(0.1), variational=False, output_variance=False, device="cuda", batch_norm=False, p_dropout=0.0):
        """
        :param input_n: num of input feature
        :param encoder_hidden_layers: num of hidden feature, decoder is made symmetric
        :param n_z: latent variable size
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VAE, self).__init__()
        print(">>> creating model")
        self.define_layers(input_n=input_n, encoder_hidden_layers=encoder_hidden_layers, n_z=n_z)

        self.activation = act_fn
        self.variational = variational
        self.output_variance = output_variance
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout

        self.encoder = VAE_Encoder(layers=self.encoder_layers, activation=self.activation, variational=variational, device=device, batch_norm=batch_norm, p_dropout=p_dropout)
        self.decoder = VAE_Decoder(layers=self.decoder_layers, activation=self.activation, output_variance=output_variance, device=device, batch_norm=batch_norm, p_dropout=p_dropout)

        print(self)
        num_parameters = sum(p.numel() for p in self.parameters())
        print(">>> total params: {:.2f}M".format(num_parameters/1000000.0))

        if device == "cuda":
            print("Moving model to GPU")
            self.cuda()
        else:
            print("Using CPU")


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

    def define_layers(self, input_n=96, encoder_hidden_layers=[100, 50], n_z=2):
        encoder_layers = []
        decoder_layers = []
        encoder_layers.append(input_n)
        decoder_layers.append(n_z)
        n_hidden = len(encoder_hidden_layers)
        for i in range(n_hidden):
            encoder_layers.append(encoder_hidden_layers[i])
            decoder_layers.append(encoder_hidden_layers[n_hidden - 1 - i])
        encoder_layers.append(n_z)
        decoder_layers.append(input_n)
        self.n_x = encoder_layers[0]
        self.n_z = encoder_layers[-1]
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

    def initialise(self, start_epoch=1, folder_name="", lr=0.0001, beta=1.0, l2_reg=False, train_batch_size=100, figs_checkpoints_save_freq=10):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_reg)
        self.folder_name = folder_name
        self.lr = lr
        self.beta = beta
        self.clipping_value = 1
        self.figs_checkpoints_save_freq = figs_checkpoints_save_freq
        if start_epoch==1:
            self.losses_file_exists = False
            self.book_keeping(start_epoch=start_epoch, train_batch_size=train_batch_size, l2_reg=l2_reg)
        else:
            self.losses_file_exists = True
            self.book_keeping(start_epoch=start_epoch, train_batch_size=train_batch_size, l2_reg=l2_reg)
            ckpt_path = self.folder_name + '/checkpoints/' + 'ckpt_' + str(start_epoch - 1) + '_weights.path.tar'
            ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))
            self.load_state_dict(ckpt['state_dict'])


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
            self.accum_loss[key] = utils.AccumLoss()
        val = val.cpu().data.numpy()
        self.accum_loss[key].update(val)

    def accum_reset(self):
        for key in self.accum_loss.keys():
            self.accum_loss[key].reset()

    def train_epoch(self, epoch, train_loader):
        self.train()
        bar = Bar('>>>', fill='>', max=len(train_loader))
        st = time.time()
        for i, (all_seq) in enumerate(train_loader):
            bt = time.time()

            inputs = all_seq.to(self.device).float()

            mu = self(inputs.float())
            loss = self.loss_VLB(inputs)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            self.optimizer.step()

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                                  time.time() - st)
            bar.next()

        head = ['Epoch']
        ret_log = [epoch]
        self.head = np.append(self.head, head)
        self.ret_log = np.append(self.ret_log, ret_log)

        bar.finish()

    def train_epoch_mnist(self, epoch, train_loader, use_bernoulli_loss=False):
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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

        if epoch % self.figs_checkpoints_save_freq == 0:
            reconstructions = reconstructions.reshape(cur_batch_size, 1, 28, 28)
            file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reals'
            self.plot_tensor_images(image, max_num_images=25, nrow=5, show=False, save_as=file_path)
            file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reconstructions'
            self.plot_tensor_images(reconstructions, max_num_images=25, nrow=5, show=False, save_as=file_path)

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

        if epoch % self.figs_checkpoints_save_freq == 0:
            file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reals'
            utils.plot_tensor_images(inputs_reshaped.detach().cpu(), max_num_images=25, nrow=5, show=False, save_as=file_path)
            file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reconstructions'
            utils.plot_tensor_images(reconstructions.detach().cpu(), max_num_images=25, nrow=5, show=False, save_as=file_path)
            file_path = self.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'diffs'
            utils.plot_tensor_images(diffs.detach().cpu(), max_num_images=25, nrow=5, show=False, save_as=file_path)
            file_path = self.folder_name + '/poses/' + str(dataset_name) + '_' + str(epoch) + '_' + 'poses_xz'
            utils.plot_poses(inputs.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=25, azim=0, evl=90, save_as=file_path)
            file_path = self.folder_name + '/poses/' + str(dataset_name) + '_' + str(epoch) + '_' + 'poses_yz'
            utils.plot_poses(inputs.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=25, azim=0, evl=-0, save_as=file_path)
            file_path = self.folder_name + '/poses/' + str(dataset_name) + '_' + str(epoch) + '_' + 'poses_xy'
            utils.plot_poses(inputs.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=25, azim=90, evl=90, save_as=file_path)

    def book_keeping(self, start_epoch=1, train_batch_size=100, l2_reg=0.0):
        self.accum_loss = dict()

        if self.variational:
            self.folder_name = self.folder_name+"_VAE"
        else:
            self.folder_name = self.folder_name+"_AE"
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
            if start_epoch==1:
                print(self)
            print("Start epoch:{}".format(start_epoch))
            print("Learning rate:{}".format(self.lr))
            print("Training batch size:{}".format(train_batch_size))
            print("BN:{}".format(self.batch_norm))
            print("l2 Reg (1e-4):{}".format(l2_reg))
            print("p_dropout:{}".format(self.p_dropout))
            print("Output variance:{}".format(self.output_variance))
            print("Beta(downweight of KL):{}".format(self.beta))
            print("Activation function:{}".format(self.activation))
            sys.stdout = original_stdout

        self.head = []
        self.ret_log = []

    def save_checkpoint_and_csv(self, epoch):
        df = pd.DataFrame(np.expand_dims(self.ret_log, axis=0))
        if self.losses_file_exists:
            with open(self.folder_name+'/'+'losses.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        else:
            df.to_csv(self.folder_name+'/'+'losses.csv', header=self.head, index=False)
            self.losses_file_exists = True
        state = {'epoch': epoch + 1,
                         'err':  self.accum_loss['train_loss'].avg,
                         'state_dict': self.state_dict()}
        if epoch % self.figs_checkpoints_save_freq == 0:
            print("Saving checkpoint....")
            file_path = self.folder_name + '/checkpoints/' + 'ckpt_' + str(epoch) + '_weights.path.tar'
            torch.save(state, file_path)
        self.head = []
        self.ret_log = []
        self.accum_reset()










