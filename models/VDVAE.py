#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

import models.utils as utils

from models.encoders import VDEncoder
from models.decoders import VDDecoder

from models.encoders import GraphVDEncoder
from models.decoders import GraphVDDecoder

from torch.nn.parameter import Parameter
import math

from models.layers import GraphConvolution


class VDVAE(nn.Module):
    def __init__(self, input_n=96, act_fn=nn.GELU(), variational=False, output_variance=False, device="cuda", batch_norm=False, p_dropout=0.0, n_zs=[50, 10, 5, 2], residual_size=200, gen_disc=False):
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
        self.residual_size = residual_size
        self.gen_disc = gen_disc

        #self.encoder = VDEncoder(input_n=input_n, act_fn=self.activation, device=self.device, batch_norm=self.batch_norm, p_dropout=self.p_dropout, n_zs=n_zs, residual_size=self.residual_size)
        self.encoder = GraphVDEncoder(input_n=input_n, act_fn=self.activation, device=self.device, batch_norm=self.batch_norm, p_dropout=self.p_dropout)
        #self.decoder = VDDecoder(input_n=input_n, encoder_activation_sizes=self.encoder.encoder_output_sizes, act_fn=self.activation, device=self.device, batch_norm=self.batch_norm, p_dropout=self.p_dropout, n_zs=n_zs, residual_size=self.residual_size)
        self.decoder = GraphVDDecoder(input_n=input_n, encoder_activation_sizes=self.encoder.level_output_sizes, act_fn=self.activation, device=self.device, batch_norm=self.batch_norm, p_dropout=self.p_dropout, residual_size=self.residual_size, gen_dsic=gen_disc)

        if not self.output_variance:
            self.implicit_logvar_hat = GraphConvolution(input_n[1], input_n[1], bias=True, node_n=input_n[0], out_node_n=input_n[0])

        self.num_parameters = utils.num_parameters_and_place_on_device(self)

    def forward(self, x, z_0=None, one_hot_labels=None):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        #Bottom Up
        encoder_activations = self.encoder.forward(x)

        # Top Down
        self.reconstructions_mu, self.reconstructions_log_var, self.zs, self.KLs = self.decoder(encoder_activations=encoder_activations, one_hot_labels=one_hot_labels)


        return self.reconstructions_mu, self.reconstructions_log_var, self.zs, self.KLs



    def generate(self, z, distribution='gaussian', latent_resolution=0, z_prev_level=0, one_hot_labels=None):
        """
        :param z: batch of random variables
        :return: batch of generated samples
        """
        if z_prev_level==0 and z!=None:
            self.decoder.zs["0"] = z
            if one_hot_labels is not None:
                self.zs["0"] = torch.cat((self.zs["0"], one_hot_labels), dim=2)
            self.decoder.residuals_dict["0"] = self.decoder.resize_conv_0(self.decoder.zs["0"])

        for i in range(z_prev_level, len(self.decoder.encoder_activation_sizes)-2):
            self.decoder.KLs[str(i+1)], self.decoder.zs[str(i+1)] = self.decoder.top_down_decode(level=i, X_supplied=False, latent_resolution=latent_resolution)

        decoder_output_final = self.decoder.decoder_block_final(self.decoder.residuals_dict[str(len(self.decoder.encoder_activation_sizes)-2)])
        self.reconstructions_mu, self.reconstructions_log_var = self.decoder.reparametisation_output(decoder_output_final)

        sig = nn.Sigmoid()
        self.bernoulli_output = sig(self.reconstructions_mu)

        if distribution=='gaussian':
            return self.reconstructions_mu
        elif distribution=='bernoulli':
            return self.bernoulli_output


    def cal_loss(self, x, mu_hat, logvar_hat=None, KLs=None, distribution='gaussian'):
        """
        :param x: batch of inputs
        :return: loss of reconstructions
        """

        if distribution=='gaussian':
            if not self.output_variance:
                b_n, node_n, f_n = mu_hat.shape
                logvar_hat = torch.ones((b_n, node_n, f_n)).to(self.device).float()
                logvar_hat = self.implicit_logvar_hat(logvar_hat)
                logvar_hat = torch.clamp(logvar_hat, min=-20.0, max=3.0)
            self.log_lik, self.mse = utils.cal_gauss_log_lik(x, mu_hat, logvar_hat)
            self.recon_loss = self.mse
        elif distribution=='bernoulli':
            self.log_lik = utils.cal_bernoulli_log_lik(x, mu_hat)
            self.recon_loss = -self.log_lik

        if self.variational:
            self.KL = sum(KLs.values())
            self.VLB = utils.cal_VLB(self.log_lik, self.KL, self.beta)
            self.loss = -self.VLB
        else:
            self.loss = -self.log_lik
        return self.loss












