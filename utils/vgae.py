#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np
from torch.autograd import Variable


class FullyConnected(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(FullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.LeakyReLU(0.1)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class VGAE_encoder(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=6, node_n=48, n_z=16):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VGAE_encoder, self).__init__()
        self.num_stage = num_stage
        self.input_feature = input_feature
        self.node_n = node_n

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))
        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc_mu = GraphConvolution(hidden_feature, n_z, node_n=node_n)
        self.gc_sigma = GraphConvolution(hidden_feature, n_z, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.LeakyReLU(0.1)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        mu = self.gc_mu(y)
        gamma = self.gc_sigma(y)
        noise = torch.normal(mean=0, std=1.0, size=gamma.shape).to(torch.device("cuda"))
        z_latent = mu + torch.mul(torch.exp(gamma), noise)

        self.KL = 0.5 * torch.sum(torch.exp(gamma) + torch.pow(mu, 2) - 1 - gamma, axis=(1,2))

        return z_latent


class VGAE_decoder(nn.Module):
    def __init__(self, n_z, output_feature, hidden_feature, p_dropout, num_stage=6, node_n=48):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VGAE_decoder, self).__init__()
        self.num_stage = num_stage
        self.n_z = n_z
        self.node_n = node_n

        self.decoder_gc1 = GraphConvolution(n_z, hidden_feature, node_n=node_n)
        self.decoder_bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.decoder_gcbs = []
        for i in range(num_stage):
            self.decoder_gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))
        self.decoder_gcbs = nn.ModuleList(self.decoder_gcbs)

        self.gc_decoder_mu = GraphConvolution(hidden_feature, output_feature, node_n=node_n)
        self.gc_decoder_sigma = GraphConvolution(hidden_feature, output_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.LeakyReLU(0.1)

    def forward(self, z_latent):
        z = self.decoder_gc1(z_latent)
        b, n, f = z.shape
        z = self.decoder_bn1(z.view(b, -1)).view(b, n, f)
        z = self.act_f(z)
        z = self.do(z)

        for i in range(self.num_stage):
            z = self.decoder_gcbs[i](z)

        recon_mu = self.gc_decoder_mu(z)
        recon_sigma = self.gc_decoder_sigma(z)
        reconstructions_mu = recon_mu
        reconstructions_log_var = torch.clamp(recon_sigma, min=-20.0, max=3.0)

        return reconstructions_mu, reconstructions_log_var




class VGAE(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=6, node_n=48, n_z=16):
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

        self.encoder = VGAE_encoder(input_feature, hidden_feature, p_dropout, num_stage=num_stage, node_n=node_n, n_z=self.n_z)
        self.decoder = VGAE_decoder(self.n_z, input_feature, hidden_feature, p_dropout, num_stage=num_stage, node_n=node_n)

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

    def forward(self, x):
        z = self.encoder(x)
        mu, log_var = self.decoder(x)

        self.KL = self.encoder.KL

        return mu, log_var, z