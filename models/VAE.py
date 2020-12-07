#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

from models.utils import *
from models.encoders import VAE_Encoder
from models.decoders import VAE_Decoder

from progress.bar import Bar
import time
from torch.autograd import Variable
import os
import sys
import numpy as np
import pandas as pd

class VAE(nn.Module):
    def __init__(self, encoder_layers=[48, 100, 50, 2],  decoder_layers = [2, 50, 100, 48], variational=False, device="cuda", ID='default'):
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
        self.losses_file_exists = False

        self.encoder = VAE_Encoder(layers=encoder_layers, variational=variational, device=device)
        self.decoder = VAE_Decoder(layers=decoder_layers, device=device)

        self.book_keeping(encoder_layers, decoder_layers, ID)

    def forward(self, x):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        self.mu, self.z, self.KL = self.encoder(x)
        self.reconstructions_mu = self.decoder(self.z)

        return self.reconstructions_mu

    def generate(self, z):
        """

        :param z: batch of random variables
        :return: batch of generated samples
        """

        reconstructions_mu = self.decoder(z)
        return reconstructions_mu

    def loss_VLB(self, x):
        """

        :param x: batch of inputs
        :return: loss of reconstructions
        """
        b_n, n_x = x.shape
        assert(n_x == self.n_x)

        self.mse = torch.pow((self.reconstructions_mu - x), 2)
        self.gauss_log_lik = torch.mean(torch.sum(self.mse, axis=1))
        if self.variational:
            self.neg_gauss_log_lik = -self.gauss_log_lik
            self.VLB = self.neg_gauss_log_lik - self.KL
            self.loss = -self.VLB
        else:
            self.loss = self.gauss_log_lik
        return self.loss

    def accum_update(self, key, val):
        if key not in self.accum_loss.keys():
            self.accum_loss[key] = AccumLoss()
        val = val.cpu().data.numpy()
        self.accum_loss[key].update(val)

    def accum_reset(self):
        for key in self.accum_loss.keys():
            self.accum_loss[key].reset()

    def train_epoch(self, epoch, lr, train_loader, optimizer):
        self.train()
        bar = Bar('>>>', fill='>', max=len(train_loader))
        st = time.time()
        for i, (all_seq) in enumerate(train_loader):
            bt = time.time()

            inputs = all_seq

            if self.device == "cuda":
              inputs = Variable(inputs.cuda()).float()

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


    def eval_full_batch(self, loader, dataset_name='val'):
        self.eval()
        bar = Bar('>>>', fill='>', max=len(loader))
        st = time.time()
        for i, (all_seq) in enumerate(loader):
            bt = time.time()

            inputs = all_seq

            if self.device == "cuda":
                inputs = Variable(inputs.cuda()).float()

            mu = self(inputs.float())

            loss = self.loss_VLB(inputs)

            self.accum_update(str(dataset_name)+'_loss', loss)
            self.accum_update(str(dataset_name)+'_gauss_log_lik', self.gauss_log_lik)
            if self.variational:
                self.accum_update(str(dataset_name)+'_KL', self.KL)

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(loader), time.time() - bt,
                                                                                  time.time() - st)
            bar.next()
        bar.finish()

        head = [dataset_name+'_loss', dataset_name+'_reconstruction']
        ret_log = [self.accum_loss[str(dataset_name)+'_loss'].avg, self.accum_loss[str(dataset_name)+'_gauss_log_lik'].avg]
        if self.variational:
            head.append(str(dataset_name)+'_KL')
            ret_log.append(self.accum_loss[str(dataset_name)+'_KL'].avg)
        self.head = np.append(self.head, head)
        self.ret_log = np.append(self.ret_log, ret_log)

    def book_keeping(self, encoder_layers, decoder_layers, ID='default'):
        self.accum_loss = dict()

        if self.variational:
            self.folder_name = "VAE"
        else:
            self.folder_name = "AE"
        for layer in encoder_layers:
            self.folder_name = self.folder_name+'_'+str(layer)
        for layer in decoder_layers:
            self.folder_name = self.folder_name+'_'+str(layer)
        self.folder_name = self.folder_name+'_'+str(ID)
        os.makedirs(os.path.join(self.folder_name, 'checkpoints'))

        original_stdout = sys.stdout
        with open(str(self.folder_name)+'/'+'architecture.txt', 'w') as f:
            sys.stdout = f
            print(self)
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
        file_path = self.folder_name + '/checkpoints/' + 'ckpt_' + str(epoch) + '_weights.path.tar'
        torch.save(state, file_path)
        self.head = []
        self.ret_log = []
        self.accum_reset()












