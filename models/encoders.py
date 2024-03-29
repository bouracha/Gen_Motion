from models.layers import *

import numpy as np

import models.utils as utils

class GraphVDEncoder(nn.Module):
    def __init__(self, input_n=[96, 10], act_fn=nn.GELU(), device="cuda", batch_norm=False, p_dropout=0.0):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GraphVDEncoder, self).__init__()
        self.activation = act_fn
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout
        self.node_n, self.features = input_n[0], input_n[1]

        # input_n -> input_n corresponding to z_bottom -> .... -> N_{z_0} corresponding to z_top
        self.level_output_sizes = [[self.node_n, self.features], [self.node_n, 128], [24, 128], [8, 128], [1, 256]]
        #self.level_output_sizes = [[self.node_n, self.features], [24, 64], [8, 128], [1, 256]]
        #self.level_output_sizes = [[self.node_n, self.features], [96, 8], [48, 16], [32, 32], [16, 64], [8, 128], [4, 256], [2, 256], [1, 256]]
        #self.level_output_sizes = [[self.node_n, self.features], [96, 256], [48, 256], [32, 256], [16, 256], [8, 256], [4, 256], [2, 256], [1, 256]]
        print(self.level_output_sizes)

        #Bottom Up
        self.graphconv_blocks = []
        self.graphconv_reductions = []
        self.graphconv_residual_reductions = []
        self.rezeros = []
        for i in range(len(self.level_output_sizes)-1):
            in_graph_size = self.level_output_sizes[i][0]
            in_feature_size = self.level_output_sizes[i][1]
            out_graph_size = self.level_output_sizes[i+1][0]
            out_feature_size = self.level_output_sizes[i+1][1]
            self.graphconv_reductions.append(GraphConvolution(in_feature_size, out_feature_size, bias=True, node_n=in_graph_size, out_node_n=out_graph_size))
            self.graphconv_blocks.append(GC_Block(out_feature_size, p_dropout, bias=True, node_n=out_graph_size, activation=nn.GELU()))
            self.graphconv_residual_reductions.append(GraphConvolution(in_feature_size, out_feature_size, bias=True, node_n=in_graph_size, out_node_n=out_graph_size))
            self.rezeros.append(ReZero())
        self.graphconv_blocks = nn.ModuleList(self.graphconv_blocks)
        self.graphconv_reductions = nn.ModuleList(self.graphconv_reductions)
        self.graphconv_residual_reductions = nn.ModuleList(self.graphconv_residual_reductions)
        self.rezeros = nn.ModuleList(self.rezeros)

    def forward(self, x):
        #Bottom Up
        self.activations = []
        y = x
        for i in range(len(self.level_output_sizes)-1):
            conv_y = self.graphconv_reductions[i](y)
            encoder_output = self.graphconv_blocks[i](conv_y)

            self.activations.append(encoder_output)

            res = self.graphconv_residual_reductions[i](y)
            y = res + self.rezeros[i](encoder_output)

        return self.activations


class VDEncoder(nn.Module):
    def __init__(self, input_n=96, act_fn=nn.GELU(), device="cuda", batch_norm=False, p_dropout=0.0, n_zs=[50, 10, 5, 2], residual_size=200):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VDEncoder, self).__init__()
        self.activation = act_fn
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout
        self.residual_size = residual_size

        # input_n -> input_n corresponding to z_bottom -> .... -> N_{z_0} corresponding to z_top
        self.encoder_output_sizes = utils.define_neurons_layers(input_n, n_zs[-1], len(n_zs)-1)
        self.encoder_output_sizes.insert(0, input_n)
        print(self.encoder_output_sizes)

        self.encoder_blocks_layers=[]
        for i in range(len(self.encoder_output_sizes)-1):
            self.encoder_blocks_layers.append(utils.define_neurons_layers(self.encoder_output_sizes[i], self.encoder_output_sizes[i+1], 2))
        print("Encoder_block_layers", self.encoder_blocks_layers)

        #Bottom Up
        self.encoder_blocks = []
        self.encoder_reshape_residual_layer = []
        self.encoder_rezero_operation = []
        for i in range(len(self.encoder_output_sizes)-1):
            self.encoder_blocks.append(NeuralNetworkBlock(layers=self.encoder_blocks_layers[i], activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout))
            self.encoder_reshape_residual_layer.append(FullyConnected(in_features=self.encoder_output_sizes[i], out_features=self.encoder_output_sizes[i+1], bias=True))
            self.encoder_rezero_operation.append(ReZero())
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.encoder_reshape_residual_layer = nn.ModuleList(self.encoder_reshape_residual_layer)
        self.encoder_rezero_operation = nn.ModuleList(self.encoder_rezero_operation)

    def forward(self, x):
        #Bottom Up
        self.activations = []
        y = x
        for i in range(len(self.encoder_output_sizes)-1):
            encoder_output = self.encoder_blocks[i](y)
            self.activations.append(encoder_output)
            res = self.encoder_reshape_residual_layer[i](y)
            y = res + self.encoder_rezero_operation[i](encoder_output)

        return self.activations


class EncoderBlock(nn.Module):
    def __init__(self, layers=[48, 100, 50, 2], activation=nn.LeakyReLU(0.1), variational=False, device="cuda", batch_norm=False, p_dropout=0.0):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(EncoderBlock, self).__init__()
        self.variational = variational
        self.device = device

        self.n_x = layers[0]
        self.n_z = layers[-1]
        self.layers = np.array(layers)
        self.n_layers = self.layers.shape[0] - 1

        self.fc_blocks = []
        for i in range(self.n_layers):
            self.fc_blocks.append(FC_Block(self.layers[i], self.layers[i + 1], activation=activation, batch_norm=batch_norm, p_dropout=p_dropout, bias=True))
        self.fc_blocks = nn.ModuleList(self.fc_blocks)

    def forward(self, x):
        y = x
        for i in range(self.n_layers):
            y = self.fc_blocks[i](y)

        return y



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