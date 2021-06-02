#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

import models.utils as utils

from models.encoders import EncoderBlock
from models.decoders import VAE_Decoder


class VAE(nn.Module):
    def __init__(self, input_n=96, hidden_layers=[100, 50], n_z=2, act_fn=nn.LeakyReLU(0.1), variational=False, output_variance=False, device="cuda", batch_norm=False, p_dropout=0.0):
        """
        :param input_n: num of input feature
        :param hidden_layers: num of hidden feature, decoder is made symmetric
        :param n_z: latent variable size
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VAE, self).__init__()
        print(">>> creating model")
        self.encoder_layers, self.decoder_layers = self.define_layers(input_n=input_n, hidden_layers=hidden_layers, n_z=n_z)

        self.activation = act_fn
        self.variational = variational
        self.output_variance = output_variance
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout

        self.encoder = EncoderBlock(layers=self.encoder_layers, activation=self.activation, variational=variational, device=device, batch_norm=batch_norm, p_dropout=p_dropout)
        self.decoder = VAE_Decoder(layers=self.decoder_layers, activation=self.activation, output_variance=output_variance, device=device, batch_norm=batch_norm, p_dropout=p_dropout)

        self.num_parameters = utils.num_parameters_and_place_on_device(self)

    def forward(self, x, num_samples=1):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        y2 = self.encoder(x)
        y1 = self.encoder(y2)

        self.z_mu_1, self.z_log_var_1 = self.reparametisation(y1)
        self.KL_1 = utils.kullback_leibler_divergence(self.z_mu_1, self.z_log_var_1, mu_2=0, log_var_2=1)
        self.z_1 = utils.reparametisation_trick(self.z_mu_1, self.z_log_var_1, self.device)

        self.reconstructions_mu = self.decoder(self.z)
        self.reconstructions_log_var = torch.zeros_like(self.reconstructions_mu)

        sig = nn.Sigmoid()
        self.bernoulli_output = sig(self.reconstructions_mu)

        return self.reconstructions_mu

    def define_layers(self, input_n=96, hidden_layers=[100, 50], n_z=2):
        encoder_layers = []
        decoder_layers = []
        encoder_layers.append(input_n)
        decoder_layers.append(n_z)
        n_hidden = len(hidden_layers)
        for i in range(n_hidden):
            encoder_layers.append(hidden_layers[i])
            decoder_layers.append(hidden_layers[n_hidden - 1 - i])
        encoder_layers.append(n_z)
        decoder_layers.append(input_n)
        self.n_x = encoder_layers[0]
        self.n_z = encoder_layers[-1]

        return encoder_layers, decoder_layers

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


    def cal_loss(self, x, distribution='gaussian'):
        """
        :param x: batch of inputs
        :return: loss of reconstructions
        """
        b_n, n_x = x.shape
        assert(n_x == self.n_x)

        if distribution=='gaussian':
            self.log_lik, self.mse = utils.cal_gauss_log_lik(x, self.reconstructions_mu, self.reconstructions_log_var)
            self.recon_loss = self.mse
        elif distribution=='bernoulli':
            self.log_lik = utils.cal_bernoulli_log_lik(x, self.reconstructions_mu)
            self.recon_loss = -self.log_lik

        if self.variational:
            self.VLB = utils.cal_VLB(self.log_lik, self.KL, self.beta)
            self.loss = -self.VLB
        else:
            self.loss = -self.log_lik
        return self.loss











