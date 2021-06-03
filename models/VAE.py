#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

import models.utils as utils

from models.layers import NeuralNetworkBlock
from models.layers import GaussianBlock
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

        self.encoder = NeuralNetworkBlock(layers=self.encoder_layers, activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
        self.reparametisation_latent = GaussianBlock(self.encoder_layers[-1], n_z)
        self.decoder = NeuralNetworkBlock(layers=self.decoder_layers, activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
        self.reparametisation_output = GaussianBlock(self.decoder_layers[-1], input_n)

        self.num_parameters = utils.num_parameters_and_place_on_device(self)

    def forward(self, x, num_samples=1):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        if self.variational:
            y = self.encoder(x)
            self.z_mu, self.z_log_var = self.reparametisation_latent(y)
            self.KL = utils.kullback_leibler_divergence(self.z_mu, self.z_log_var)
        else:
            self.z_mu = self.encoder(x)
        for i in range(num_samples):
            if self.variational:
                self.z = utils.reparametisation_trick(self.z_mu, self.z_log_var, self.device)
            else:
                self.z = self.z_mu
            decoder_output = self.decoder(self.z)
            reconstructions_mu, reconstructions_log_var = self.reparametisation_output(decoder_output)
            if i==0:
                recon_mu_accum = reconstructions_mu
                recon_log_var_accum = reconstructions_log_var
            else:
                recon_mu_accum += reconstructions_mu
                recon_log_var_accum += reconstructions_log_var
        self.reconstructions_mu = recon_mu_accum/(1.0*num_samples)
        self.reconstructions_log_var = recon_log_var_accum/(1.0*num_samples)

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
            if not self.output_variance:
                self.reconstructions_log_var = torch.zeros_like(self.reconstructions_mu)
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











