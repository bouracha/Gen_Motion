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
        self.act_f = nn.Tanh()

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


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48, variational=False):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.variational = variational
        if variational:
            self.gc_z_mu = GraphConvolution(hidden_feature, hidden_feature, node_n=node_n)
            self.gc_z_sigma = GraphConvolution(hidden_feature, hidden_feature, node_n=node_n)
            # self.gc_z = GraphConvolution(hidden_feature, hidden_feature, node_n=node_n)
            # self.bnz = nn.BatchNorm1d(node_n * hidden_feature)

        self.gc7 = GraphConvolution(hidden_feature, 2*input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        #self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(0.1)
        self.normalised_act_f = nn.Sigmoid()

    def set_normalising_varaiables(self, maximum, minimum):
        self.data_max = maximum
        self.data_min = minimum

    def forward(self, x):
        max_first = 2 * np.sqrt(20) * np.pi
        min_first = -max_first
        max_l = 4 * np.sqrt(20) * np.pi
        min_l = -max_l

        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage // 2):
            y = self.gcbs[i](y)

        self.KL = None
        if self.variational:
            mu = self.gc_z_mu(y)
            gamma = self.gc_z_sigma(y)
            noise = torch.normal(mean=0, std=1.0, size=gamma.shape).to(torch.device("cuda"))
            z = mu + torch.mul(torch.exp(gamma), noise)
            y = z

            self.KL = 0.5 * torch.sum(torch.exp(gamma) + torch.pow(mu, 2) - 1 - gamma, axis=(1, 2))

        for i in range(self.num_stage // 2, self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        logits = y[:,:,:20].clone()
        log_var = torch.clamp(y[:,:,20:], min=-10.0, max=10.0) #torch.log(torch.square(y[:,:,20:]))
        #outputs_scaled = self.normalised_act_f(logits)

        #outputs = outputs_scaled.clone()
        #outputs[:, :, 0] = outputs[:, :, 0] * (max_first - min_first) + min_first
        #outputs[:, :, 1:] = outputs[:, :, 1:] * (max_l - min_l) + min_l

        outputs = logits + x

        return outputs, logits, log_var
