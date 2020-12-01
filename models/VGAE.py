#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

from models.utils import *
from model.encoders import VGAE_Encoder
from model.decoders import VGAE_Decoder




class VGAE(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=6, node_n=48, n_z=16, hybrid=True):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VGAE, self).__init__()
        self.num_stage = num_stage
        self.input_feature = input_feature
        self.node_n = node_n
        self.n_z = n_z
        self.hybrid = hybrid

        self.encoder = VGAE_Encoder(input_feature, hidden_feature, p_dropout, num_stage=num_stage, node_n=node_n,
                                    n_z=self.n_z, hybrid=self.hybrid)
        self.decoder = VGAE_Decoder(self.n_z, input_feature, hidden_feature, p_dropout, num_stage=num_stage,
                                    node_n=node_n, hybrid=self.hybrid)

        # Book keeping values
        self.accum_loss = dict()

    def forward(self, x):
        b = x.shape[0]
        x.shape == (b, self.node_n, self.input_feature)

        self.z, self.KL = self.encoder(x)
        self.mu, self.log_var = self.decoder(self.z)

        return self.mu, self.log_var, self.z

    def generate(self, z):
        """

        :param z: batch of random variables
        :return: batch of generated samples
        """
        b = z.shape[0]
        #if self.hybrid:
        #    assert(z.shape == (b, self.n_z))
        #else:
        #    assert(z.shape == (b, self.node_n, self.n_z))

        mu, log_var = self.decoder(z)
        return mu

    def loss_VLB(self, x):
        """

        :param x: batch of inputs
        :return: loss of reconstructions
        """
        b, n, t = x.shape
        assert (t == self.input_feature)
        assert (n == self.node_n)

        self.mse = torch.pow((self.mu - x), 2)
        self.gauss_log_lik = self.mse#0.5 * (self.log_var + np.log(2 * np.pi) + (self.mse / (1e-8 + torch.exp(self.log_var))))
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

