import torch.nn as nn
import torch

from models.layers import *

class VGAE_Decoder(nn.Module):
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