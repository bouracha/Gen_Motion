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

class AccumLoss(object):
    def __init__(self):
        self.reset()

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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

    def __init__(self, in_features, out_features, bias=True, node_n=48, out_node_n=None):
        super(GraphConvolution, self).__init__()
        if out_node_n is None:
            out_node_n = node_n
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(out_node_n, node_n))
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
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=6, node_n=48, n_z=16, hybrid=True):
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
        self.n_z = n_z
        self.hybrid = hybrid

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))
        self.gcbs = nn.ModuleList(self.gcbs)

        if self.hybrid:
            out_node_n = 24
            out_hidden_feature = 128
            self.gc_down_1 = GraphConvolution(hidden_feature, out_hidden_feature, node_n=node_n, out_node_n=out_node_n)
            self.bn_down_1 = nn.BatchNorm1d(out_node_n * out_hidden_feature)
            node_n = out_node_n
            out_node_n = 8
            hidden_feature = out_hidden_feature
            out_hidden_feature = 64
            self.gc_down_2 = GraphConvolution(hidden_feature, out_hidden_feature, node_n=node_n, out_node_n=out_node_n)
            self.bn_down_2 = nn.BatchNorm1d(out_node_n * out_hidden_feature)
            node_n = out_node_n
            hidden_feature = out_hidden_feature
            #out_hidden_feature = n_z
            #self.gc_mu = GraphConvolution(hidden_feature, out_hidden_feature, node_n=node_n, out_node_n=out_node_n)
            #self.gc_sigma = GraphConvolution(hidden_feature, out_hidden_feature, node_n=node_n, out_node_n=out_node_n)
            self.fc_z_mu = FullyConnected(node_n*hidden_feature, n_z)
            self.fc_z_sigma = FullyConnected(node_n*hidden_feature, n_z)
        else:
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

        if self.hybrid:
            y = self.gc_down_1(y)
            b, n, f = y.shape
            y = self.bn_down_1(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y)
            y = self.do(y)

            y = self.gc_down_2(y)
            b, n, f = y.shape
            y = self.bn_down_2(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y)
            y = self.do(y)

            y = y.view(b, -1)
            mu = self.fc_z_mu(y)
            gamma = self.fc_z_sigma(y)
        else:
            mu = self.gc_mu(y)
            gamma = self.gc_sigma(y)
        gamma = torch.clamp(gamma, min=-5.0, max=5.0)
        noise = torch.normal(mean=0, std=1.0, size=gamma.shape).to(torch.device("cuda"))
        z_latent = mu + torch.mul(torch.exp(gamma/2.0), noise)

        #if self.hybrid:
        #    assert(z_latent.shape == (b, self.n_z))
        KL_per_sample = 0.5 * torch.sum(torch.exp(gamma) + torch.pow(mu, 2) - 1 - gamma, axis=1)
        #else:
        #assert(z_latent.shape == (b, n, self.n_z))
        #KL_per_sample = 0.5 * torch.sum(torch.exp(gamma) + torch.pow(mu, 2) - 1 - gamma, axis=(1, 2))

        KL = torch.mean(KL_per_sample)

        return z_latent, KL


class VGAE_decoder(nn.Module):
    def __init__(self, n_z, output_feature, hidden_feature, p_dropout, num_stage=6, node_n=48, hybrid=True):
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
        self.hybrid = hybrid
        self.hidden_feature = hidden_feature

        if self.hybrid:
            node_n = 2
            out_node_n = 8
            out_hidden_feature = 64
            #self.gc_up_1 = GraphConvolution(n_z, out_hidden_feature, node_n=node_n, out_node_n=out_node_n)
            self.fc_decoder = FullyConnected(n_z, out_node_n * out_hidden_feature)
            self.bn_up_1 = nn.BatchNorm1d(out_node_n * out_hidden_feature)
            node_n = out_node_n
            out_node_n = 24
            hidden_feature = out_hidden_feature
            out_hidden_feature = 128
            self.gc_up_2 = GraphConvolution(hidden_feature, out_hidden_feature, node_n=node_n, out_node_n=out_node_n)
            self.bn_up_2 = nn.BatchNorm1d(out_node_n * out_hidden_feature)
            node_n = out_node_n
            out_node_n = 48
            hidden_feature = out_hidden_feature
            out_hidden_feature = 256
            self.decoder_gc1 = GraphConvolution(hidden_feature, out_hidden_feature, node_n=node_n, out_node_n=out_node_n)
            node_n = out_node_n
            hidden_feature = out_hidden_feature
        else:
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

    def forward(self, z):
        b = z.shape[0]
        if self.hybrid:
            y = self.fc_decoder(z)
            b = y.shape[0]
            y = self.bn_up_1(y.view(b, -1)).view(b, 8, 64)
            y = self.act_f(y)
            y = self.do(y)

            y = self.gc_up_2(y)
            b, n, f = y.shape
            y = self.bn_up_2(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y)
            y = self.do(y)
            y = self.decoder_gc1(y)
        else:
            assert(z.shape == (b, self.node_n, self.hidden_feature))
            y = self.decoder_gc1(z)
        b, n, f = y.shape
        y = self.decoder_bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.decoder_gcbs[i](y)

        recon_mu = self.gc_decoder_mu(y)
        recon_sigma = self.gc_decoder_sigma(y)
        reconstructions_mu = recon_mu
        reconstructions_log_var = torch.clamp(recon_sigma, min=-20.0, max=3.0)

        return reconstructions_mu, reconstructions_log_var


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

        self.encoder = VGAE_encoder(input_feature, hidden_feature, p_dropout, num_stage=num_stage, node_n=node_n,
                                    n_z=self.n_z, hybrid=self.hybrid)
        self.decoder = VGAE_decoder(self.n_z, input_feature, hidden_feature, p_dropout, num_stage=num_stage,
                                    node_n=node_n, hybrid=self.hybrid)

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

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

