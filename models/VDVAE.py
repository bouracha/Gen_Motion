#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

import models.utils as utils

from models.layers import NeuralNetworkBlock
from models.layers import GaussianBlock


class VDVAE(nn.Module):
    def __init__(self, input_n=96, hidden_layers=[100, 50], n_z=2, act_fn=nn.LeakyReLU(0.1), variational=False, output_variance=False, device="cuda", batch_norm=False, p_dropout=0.0):
        """
        :param input_n: num of input feature
        :param hidden_layers: num of hidden feature, decoder is made symmetric
        :param n_z: latent variable size
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VDVAE, self).__init__()
        print(">>> creating model")
        self.encoder_layers, self.decoder_layers = self.define_layers(input_n=input_n, hidden_layers=hidden_layers, n_z=n_z)

        self.activation = act_fn
        self.variational = variational
        self.output_variance = output_variance
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout

        #Layers hardcooded for now
        n_z_1 = 5
        n_z_0 = 2
        encoder_block_1_layers=[input_n, 500, 400, 300, 200, 100, n_z_1]

        encoder_block_2_layers=[n_z_1, 10, 5, n_z_0]

        decoder_block_1_layers=[n_z_0, 5, 10, n_z_1]
        decoder_block_2_layers=[n_z_1, 100, 200, 300, 400, 500]


        self.encoder_block_1 = NeuralNetworkBlock(layers=encoder_block_1_layers, activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
        self.encoder_block_2 = NeuralNetworkBlock(layers=encoder_block_2_layers, activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)

        self.reparametisation_latent_0 = GaussianBlock(encoder_block_2_layers[-1], n_z_0)

        self.decoder_block_1 = NeuralNetworkBlock(layers=decoder_block_1_layers, activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
        self.reparametisation_latent_1 = GaussianBlock(decoder_block_1_layers[-1], n_z_1)
        self.prior_decoder_block_1 = NeuralNetworkBlock(layers=decoder_block_1_layers, activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
        self.reparametisation_latent_1_prior = GaussianBlock(decoder_block_1_layers[-1], n_z_1)

        self.decoder_block_2 = NeuralNetworkBlock(layers=decoder_block_2_layers, activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
        self.reparametisation_output = GaussianBlock(decoder_block_2_layers[-1], input_n)

        self.num_parameters = utils.num_parameters_and_place_on_device(self)

    def forward(self, x, num_samples=1):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        #Bottom Up
        encoder_block_1_output = self.encoder_block_1(x)
        encoder_block_2_output = self.encoder_block_2(encoder_block_1_output)

        #Top Down
        self.z_mu_0, self.z_log_var_0 = self.reparametisation_latent_0(encoder_block_2_output)
        self.KL_0 = utils.kullback_leibler_divergence(self.z_mu_0, self.z_log_var_0, mu_2=torch.zeros_like(self.z_mu_0), log_var_2=torch.ones_like(self.z_log_var_0))
        self.z_0 = utils.reparametisation_trick(self.z_mu_0, self.z_log_var_0, self.device)

        decoder_block_1_output = self.decoder_block_1(self.z_0)
        concat_for_latent_1 = decoder_block_1_output + encoder_block_1_output

        prior_decoder_block_1_output = self.prior_decoder_block_1(self.z_0)
        self.prior_z_mu_1, self.prior_z_log_var_1 = self.reparametisation_latent_1_prior(prior_decoder_block_1_output)

        self.z_mu_1, self.z_log_var_1 = self.reparametisation_latent_1(concat_for_latent_1)
        self.KL_1 = utils.kullback_leibler_divergence(self.z_mu_1, self.z_log_var_1, mu_2=self.prior_z_mu_1, log_var_2=self.prior_z_log_var_1)
        self.z_1 = utils.reparametisation_trick(self.z_mu_1, self.z_log_var_1, self.device)

        decoder_output_2 = self.decoder_block_2(self.z_1)
        self.reconstructions_mu, self.reconstructions_log_var = self.reparametisation_output(decoder_output_2)

        #Total KL
        self.KL = self.KL_0 + self.KL_1

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
        decoder_output = self.decoder(self.z)
        reconstructions_mu, reconstructions_log_var = self.reparametisation_output(decoder_output)

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












