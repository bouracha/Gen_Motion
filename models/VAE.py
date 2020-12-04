#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

from models.utils import *
from models.encoders import VAE_Encoder
from models.decoders import VAE_Decoder

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
        b_n, x_n = x.shape
        assert(x_n == self.x_n)

        self.mse = torch.pow((self.mu - x), 2)
        self.gauss_log_lik = self.mse #0.5 * (self.log_var + np.log(2 * np.pi) + (self.mse / (1e-8 + torch.exp(self.log_var))))
        self.neg_gauss_log_lik = -torch.mean(torch.sum(self.gauss_log_lik, axis=(1, 2)))
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