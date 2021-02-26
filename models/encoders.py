import torch.nn as nn
import torch

from models.layers import *

import numpy as np

class VAE_Encoder(nn.Module):
    def __init__(self, layers=[48, 100, 50, 2], activation=nn.LeakyReLU(0.1), variational=False, device="cuda", batch_norm=False, p_dropout=0.0):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VAE_Encoder, self).__init__()
        self.variational = variational
        self.device = device

        self.n_x = layers[0]
        self.n_z = layers[-1]
        self.layers = np.array(layers)
        self.n_layers = self.layers.shape[0] - 1

        self.fc_blocks = []
        for i in range(self.n_layers - 1):
            self.fc_blocks.append(FC_Block(self.layers[i], self.layers[i + 1], activation=activation, batch_norm=batch_norm, p_dropout=p_dropout, bias=True))
        self.fc_blocks = nn.ModuleList(self.fc_blocks)

        self.z_mu_fc = FullyConnected(self.layers[-2], self.n_z)
        if self.variational:
            self.z_log_var_fc = FullyConnected(self.layers[-2], self.n_z)

    def forward(self, x):
        y = x
        for i in range(self.n_layers - 1):
            y = self.fc_blocks[i](y)

        mu = self.z_mu_fc(y)
        if self.variational:
            log_var = self.z_log_var_fc(y)
            log_var = torch.clamp(log_var, min=-20.0, max=3.0)
        else:
            log_var = None

        return mu, log_var


class VGAE_Encoder(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=6, node_n=48, n_z=16, hybrid=True):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VGAE_Encoder, self).__init__()
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
            # out_hidden_feature = n_z
            # self.gc_mu = GraphConvolution(hidden_feature, out_hidden_feature, node_n=node_n, out_node_n=out_node_n)
            # self.gc_sigma = GraphConvolution(hidden_feature, out_hidden_feature, node_n=node_n, out_node_n=out_node_n)
            self.fc_z_mu = FullyConnected(node_n * hidden_feature, n_z)
            self.fc_z_sigma = FullyConnected(node_n * hidden_feature, n_z)
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
        z_latent = mu + torch.mul(torch.exp(gamma / 2.0), noise)

        KL_per_sample = 0.5 * torch.sum(torch.exp(gamma) + torch.pow(mu, 2) - 1 - gamma, axis=1)

        KL = torch.mean(KL_per_sample)

        return z_latent, KL