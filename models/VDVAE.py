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
        self.feature_hierachies=[input_n, 50, 10, 5, 2]
        encoder_blocks_layers=[]
        for i in range(len(self.feature_hierachies)-1):
            encoder_blocks_layers.append(utils.define_neurons_layers(self.feature_hierachies[i], self.feature_hierachies[i+1], 5))
        decoder_blocks_layers=[]
        for i in range(len(self.feature_hierachies)-2, -1, -1):
            decoder_blocks_layers.append(encoder_blocks_layers[i][::-1])

        #Bottom Up
        self.encoder_blocks = []
        for i in range(len(self.feature_hierachies)-1):
            self.encoder_blocks.append(NeuralNetworkBlock(layers=encoder_blocks_layers[i], activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout))
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.z_mus = {}
        self.z_log_vars = {}
        self.z_prior_mus = {}
        self.z_prior_log_vars = {}
        self.KLs = {}
        self.zs = {}

        #Top Down
        self.reparametisation_latent_0 = GaussianBlock(encoder_blocks_layers[-1][-1], self.feature_hierachies[-1])

        self.decoder_units = []
        for i in range(len(decoder_blocks_layers)):
            self.decoder_block = NeuralNetworkBlock(layers=decoder_blocks_layers[i], activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
            self.reparametisation_latent = GaussianBlock(decoder_blocks_layers[i][-1], self.feature_hierachies[-2-i])
            self.prior_decoder_block = NeuralNetworkBlock(layers=decoder_blocks_layers[i], activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
            self.reparametisation_latent_prior = GaussianBlock(decoder_blocks_layers[i][-1], self.feature_hierachies[-2-i])
            self.decoder_units.append({
                "decoder_net":self.decoder_block,
                "repara_latent_net":self.reparametisation_latent,
                "decoder_prior_net":self.prior_decoder_block,
                "repara_latent_prior_net":self.reparametisation_latent_prior})
            self.decoder_units[i] = nn.ModuleDict(self.decoder_units[i])
        self.decoder_units = nn.ModuleList(self.decoder_units)

        self.decoder_block_final = NeuralNetworkBlock(layers=decoder_blocks_layers[-1][:-1], activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
        self.reparametisation_output = GaussianBlock(decoder_blocks_layers[-1][:-1][-1], input_n)

        self.num_parameters = utils.num_parameters_and_place_on_device(self)

    def forward(self, x):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        #Bottom Up
        self.encoder_outputs = []
        encoder_output = x
        for i in range(len(self.feature_hierachies)-1):
            encoder_output = self.encoder_blocks[i](encoder_output)
            self.encoder_outputs.append(encoder_output)

        #Top Down
        self.z_mus["0"], self.z_log_vars["0"] = self.reparametisation_latent_0(self.encoder_outputs[-1])
        self.KLs["0"] = utils.kullback_leibler_divergence(self.z_mus["0"], self.z_log_vars["0"], mu_2=torch.zeros_like(self.z_mus["0"]), log_var_2=torch.ones_like(self.z_log_vars["0"]))
        self.zs["0"] = utils.reparametisation_trick(self.z_mus["0"], self.z_log_vars["0"], self.device)

        for i in range(len(self.feature_hierachies)-2):
            self.KLs[str(i+1)], self.zs[str(i+1)] = self.top_down_decode(level=i)

        decoder_output_final = self.decoder_block_final(self.zs[str(len(self.feature_hierachies)-2)])
        self.reconstructions_mu, self.reconstructions_log_var = self.reparametisation_output(decoder_output_final)

        self.KL = sum(self.KLs.values())

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

    def top_down_decode(self, level):
        #decoder_block_output = self.decoder_units[level]["decoder_net"](self.zs[str(level)])
        #concat_for_latent = decoder_block_output + self.encoder_outputs[-2-level]
        #self.z_mus[str(level+1)], self.z_log_vars[str(level+1)] = self.decoder_units[level]["repara_latent_net"](concat_for_latent)
        concat_for_latent = self.zs[str(level)] + self.encoder_outputs[-1-level]
        decoder_block_output = self.decoder_units[level]["decoder_net"](concat_for_latent)
        self.z_mus[str(level+1)], self.z_log_vars[str(level+1)] = self.decoder_units[level]["repara_latent_net"](decoder_block_output)

        prior_decoder_block_1_output = self.decoder_units[level]["decoder_prior_net"](self.zs[str(level)])
        self.z_prior_mus[str(level+1)], self.z_prior_log_vars[str(level+1)] = self.decoder_units[level]["repara_latent_prior_net"](prior_decoder_block_1_output)

        self.KLs[str(level+1)] = utils.kullback_leibler_divergence(self.z_mus[str(level+1)], self.z_log_vars[str(level+1)], mu_2=self.z_prior_mus[str(level+1)], log_var_2=self.z_prior_log_vars[str(level+1)])
        self.zs[str(level+1)] = utils.reparametisation_trick(self.z_mus[str(level+1)], self.z_log_vars[str(level+1)], self.device)

        #Residual
        #self.zs[str(level + 1)] = self.zs[str(level+1)] + self.zs[str(level)]

        return self.KLs[str(level+1)], self.zs[str(level+1)]

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












