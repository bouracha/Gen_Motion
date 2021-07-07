#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

import models.utils as utils

from models.layers import NeuralNetworkBlock
from models.layers import GaussianBlock
from models.layers import FullyConnected


class VDVAE(nn.Module):
    def __init__(self, input_n=96, act_fn=nn.GELU(), variational=False, output_variance=False, device="cuda", batch_norm=False, p_dropout=0.0, n_zs=[50, 10, 5, 2]):
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

        self.activation = act_fn
        self.variational = variational
        self.output_variance = output_variance
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout

        self.feature_hierachies=[]
        self.feature_hierachies.append(input_n)
        for n_z in n_zs:
            self.feature_hierachies.append(n_z)
        print("Feature hierachies: ", self.feature_hierachies)
        self.encoder_output_sizes = utils.define_neurons_layers(input_n, n_zs[-1], len(n_zs))
        print(self.encoder_output_sizes)
        encoder_blocks_layers=[]
        for i in range(len(self.encoder_output_sizes)-1):
            encoder_blocks_layers.append(utils.define_neurons_layers(self.encoder_output_sizes[i], self.encoder_output_sizes[i+1], 2))
        print("encoder_block_layers", encoder_blocks_layers)

        #Bottom Up
        self.encoder_blocks = []
        self.encoder_residual_blocks = []
        for i in range(len(self.feature_hierachies)-1):
            self.encoder_blocks.append(NeuralNetworkBlock(layers=encoder_blocks_layers[i], activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout))
            self.encoder_residual_blocks.append(FullyConnected(in_features=self.encoder_output_sizes[i], out_features=self.encoder_output_sizes[i+1], bias=True))
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.encoder_residual_blocks = nn.ModuleList(self.encoder_residual_blocks)

        self.z_mus = {}
        self.z_log_vars = {}
        self.z_posterior_mus = {}
        self.z_posterior_log_vars = {}
        self.z_prior_mus = {}
        self.z_prior_log_vars = {}
        self.KLs = {}
        self.zs = {}
        self.residuals_dict = {}

        #Top Down
        self.reparametisation_latent_0 = GaussianBlock(encoder_blocks_layers[i][-1], self.feature_hierachies[-1])
        self.reshape_z0_linearly = FullyConnected(in_features=self.feature_hierachies[-1], out_features=100, bias=True)

        self.decoder_units = []
        for i in range(len(self.feature_hierachies)-2):
            self.begin_decoder_block = NeuralNetworkBlock(layers=utils.define_neurons_layers(100, 100, 4), activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)

            print("size of encoder used at decoder level {} is {}".format(i, self.encoder_output_sizes[-2-i]))
            self.posterior_decoder_block = NeuralNetworkBlock(layers=utils.define_neurons_layers(self.encoder_output_sizes[-2-i] + 100, 100, 4), activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
            self.reparametisation_posterior = GaussianBlock(100, self.feature_hierachies[-2-i])

            self.prior_decoder_block = NeuralNetworkBlock(layers=utils.define_neurons_layers(100, 100, 4), activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
            self.reparametisation_prior = GaussianBlock(100, self.feature_hierachies[-2-i])

            self.reshape_z_linearly = FullyConnected(in_features=self.feature_hierachies[-2-i], out_features=100, bias=True)

            self.decoder_units.append({
                "decoder_block":self.begin_decoder_block,
                "posterior_decoder_block":self.posterior_decoder_block,
                "reparametisation_posterior":self.reparametisation_posterior,
                "prior_decoder_block":self.prior_decoder_block,
                "reparametisation_prior": self.reparametisation_prior,
                "reshape_z_linearly": self.reshape_z_linearly
            })
            self.decoder_units[i] = nn.ModuleDict(self.decoder_units[i])
        self.decoder_units = nn.ModuleList(self.decoder_units)

        self.decoder_block_final = NeuralNetworkBlock(layers=utils.define_neurons_layers(self.feature_hierachies[1], 100, 4), activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
        self.reparametisation_output = GaussianBlock(100, input_n)

        self.num_parameters = utils.num_parameters_and_place_on_device(self)

    def forward(self, x):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        #Bottom Up
        self.encoder_outputs = []
        y = x
        for i in range(len(self.feature_hierachies)-1):
            encoder_output = self.encoder_blocks[i](y)
            self.encoder_outputs.append(encoder_output)
            res = self.encoder_residual_blocks[i](y)
            y = encoder_output + res

        #Top Down
        self.z_posterior_mus["0"], self.z_posterior_log_vars["0"] = self.reparametisation_latent_0(self.encoder_outputs[-1])
        self.z_prior_mus["0"], self.z_prior_log_vars["0"] = torch.zeros_like(self.z_posterior_mus["0"]), torch.zeros_like(self.z_posterior_log_vars["0"])
        self.z_mus["0"], self.z_log_vars["0"] = self.z_posterior_mus["0"], self.z_posterior_log_vars["0"]

        self.KLs["0"] = utils.kullback_leibler_divergence(self.z_mus["0"], self.z_log_vars["0"], mu_2=torch.zeros_like(self.z_mus["0"]), log_var_2=torch.ones_like(self.z_log_vars["0"]))
        self.zs["0"] = utils.reparametisation_trick(self.z_mus["0"], self.z_log_vars["0"], self.device)

        self.residuals_dict["0"] = self.reshape_z0_linearly(self.zs["0"])

        for i in range(len(self.feature_hierachies)-2):
            self.KLs[str(i+1)], self.zs[str(i+1)] = self.top_down_decode(level=i)

        decoder_output_final = self.decoder_block_final(self.zs[str(len(self.feature_hierachies)-2)])
        self.reconstructions_mu, self.reconstructions_log_var = self.reparametisation_output(decoder_output_final)

        self.KL = sum(self.KLs.values())

        sig = nn.Sigmoid()
        self.bernoulli_output = sig(self.reconstructions_mu)

        return self.reconstructions_mu

    def top_down_decode(self, level, train=True, latent_resolution=999):
        res_block_output = self.decoder_units[level]["decoder_block"](self.residuals_dict[str(level)])
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level)] + res_block_output

        if train:
            concat_for_posterior = torch.cat((self.residuals_dict[str(level+1)], self.encoder_outputs[-2-level]), dim=1)
            posterior_decoder_block_output = self.decoder_units[level]["posterior_decoder_block"](concat_for_posterior)
            self.z_posterior_mus[str(level + 1)], self.z_posterior_log_vars[str(level + 1)] = self.decoder_units[level]["reparametisation_posterior"](posterior_decoder_block_output)

        prior_decoder_block_output = self.decoder_units[level]["prior_decoder_block"](self.residuals_dict[str(level+1)])
        self.z_prior_mus[str(level + 1)], self.z_prior_log_vars[str(level + 1)] = self.decoder_units[level]["reparametisation_prior"](prior_decoder_block_output)
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level+1)] + prior_decoder_block_output

        if train:
            self.z_mus[str(level + 1)] = self.z_posterior_mus[str(level+1)]
            self.z_log_vars[str(level + 1)] = self.z_posterior_log_vars[str(level+1)]
        else:
            self.z_mus[str(level + 1)] = self.z_prior_mus[str(level+1)]
            self.z_log_vars[str(level + 1)] = self.z_prior_log_vars[str(level+1)]

        if level < latent_resolution:
            self.KLs[str(level+1)] = utils.kullback_leibler_divergence(self.z_mus[str(level+1)], self.z_log_vars[str(level+1)], mu_2=self.z_prior_mus[str(level+1)], log_var_2=self.z_prior_log_vars[str(level+1)])
            self.zs[str(level+1)] = utils.reparametisation_trick(self.z_mus[str(level+1)], self.z_log_vars[str(level+1)], self.device)
        else:
            self.zs[str(level + 1)] = self.z_mus[str(level+1)]

        reshaped_z = self.decoder_units[level]["reshape_z_linearly"](self.zs[str(level+1)])
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level+1)] + reshaped_z

        return self.KLs[str(level+1)], self.zs[str(level+1)]

    def generate(self, z, distribution='gaussian', latent_resolution=0, z_prev_level=0):
        """
        :param z: batch of random variables
        :return: batch of generated samples
        """
        if z_prev_level==0 and z!=None:
            self.zs["0"] = z
            self.residuals_dict["0"] = self.reshape_z0_linearly(self.zs["0"])

        for i in range(z_prev_level, len(self.feature_hierachies)-2):
            self.KLs[str(i+1)], self.zs[str(i+1)] = self.top_down_decode(level=i, train=False, latent_resolution=latent_resolution)

        decoder_output_final = self.decoder_block_final(self.zs[str(len(self.feature_hierachies)-2)])
        self.reconstructions_mu, self.reconstructions_log_var = self.reparametisation_output(decoder_output_final)

        sig = nn.Sigmoid()
        self.bernoulli_output = sig(self.reconstructions_mu)

        if distribution=='gaussian':
            return self.reconstructions_mu
        elif distribution=='bernoulli':
            return self.bernoulli_output


    def cal_loss(self, x, distribution='gaussian'):
        """
        :param x: batch of inputs
        :return: loss of reconstructions
        """

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












