#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

import models.utils as utils

from models.encoders import VAE_Encoder
from models.decoders import VAE_Decoder

import numpy as np

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

        self.encoder = VAE_Encoder(layers=self.encoder_layers, activation=self.activation, variational=variational, device=device, batch_norm=batch_norm, p_dropout=p_dropout)
        self.decoder = VAE_Decoder(layers=self.decoder_layers, activation=self.activation, output_variance=output_variance, device=device, batch_norm=batch_norm, p_dropout=p_dropout)

        self.num_parameters = utils.num_parameters_and_place_on_device(self)

    def forward(self, x, num_samples=1):
        """
        :param x: batch of samples
        :return: reconstructions and latent value
        """
        if self.variational:
            self.z_mu, self.z_log_var = self.encoder(x)
            self.KL = utils.kullback_leibler_divergence(self.z_mu, self.z_log_var)
        else:
            self.z_mu, _ = self.encoder(x)
        for i in range(num_samples):
            if self.variational:
                self.z = utils.reparametisation_trick(self.z_mu, self.z_log_var, self.device)
            else:
                self.z = self.z_mu
            if self.output_variance:
                reconstructions_mu, reconstructions_log_var = self.decoder(self.z)
            else:
                reconstructions_mu = self.decoder(self.z)
                reconstructions_log_var = torch.zeros_like(reconstructions_mu)
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

    def loss_bernoulli(self, x):
        """
        :param x: batch of inputs
        :return
        """
        b_n, n_x = x.shape
        assert (n_x == self.n_x)

        logits = self.reconstructions_mu
        BCE = torch.maximum(logits, torch.zeros_like(logits)) - torch.multiply(logits, x) + torch.log(1 + torch.exp(-torch.abs(logits)))
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












