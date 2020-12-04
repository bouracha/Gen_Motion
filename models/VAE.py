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

class VAE(nn.Module):
    def __init__(self, encoder_layers=[48, 100, 50, 2],  decoder_layers = [2, 50, 100, 48], variational=False, device="cuda"):
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
        self.device = device

        self.encoder = VAE_Encoder(layers=encoder_layers, variational=variational, device=device)
        self.decoder = VAE_Decoder(layers=decoder_layers, device=device)

        # Book keeping values
        self.accum_loss = dict()

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
        self.gauss_log_lik = self.mse #0.5 * (self.log_var + np.log(2 * np.pi) + (self.mse / (1e-8 + torch.exp(self.log_var))))
        self.neg_gauss_log_lik = -torch.mean(torch.sum(self.gauss_log_lik, axis=1))
        self.VLB = self.neg_gauss_log_lik - self.KL
        self.loss = -self.VLB

        return self.loss

    def accum_update(self, key, val):
        if key not in self.accum_loss.keys():
            self.accum_loss[key] = AccumLoss()
        val = val.cpu().data.numpy()
        self.accum_loss[key].update(val)

    def accum_reset(self):
        for key in self.accum_loss.keys():
            self.accum_loss[key].reset()

    def train_epoch(self, train_loader, optimizer):
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
            #print("\n", loss.cpu().data.numpy())
            #print(model.neg_gauss_log_lik.cpu().data.numpy())
            #print(model.KL.cpu().data.numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.accum_update('train_loss', loss) #updates accum_loss['train_loss']
            self.accum_update('train_neg_gauss_log_lik', self.neg_gauss_log_lik)
            if self.variational:
                self.accum_update('train_KL', self.KL)
            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                                  time.time() - st)
            bar.next()
        bar.finish()
        print("Train: ")
        print("loss", self.accum_loss['train_loss'].avg)
        print("neg_gauss_log_lik", self.accum_loss['train_neg_gauss_log_lik'].avg)
        print("KL", self.accum_loss['train_KL'].avg)



