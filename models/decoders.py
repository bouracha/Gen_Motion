import torch.nn as nn
import torch

from models.layers import *

import numpy as np

import models.utils as utils

class GraphVDDecoder(nn.Module):
    def __init__(self, input_n=[96, 10], encoder_activation_sizes=[[96, 8], [24, 64], [8, 128], [1, 256]], act_fn=nn.GELU(), device="cuda", batch_norm=False, p_dropout=0.0, residual_size=256, gen_dsic=False):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GraphVDDecoder, self).__init__()
        self.activation = act_fn
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout
        self.node_input_n = 96
        self.input_temp_n = encoder_activation_sizes[0][1]
        self.residual_size = residual_size
        self.encoder_activation_sizes = encoder_activation_sizes

        self.gen_dsic = gen_dsic
        if self.gen_dsic:
            self.num_classes = 13
        else:
            self.num_classes = 0

        #self.feature_hierachies=[]
        #self.feature_hierachies.append(input_n)
        #for n_z in n_zs:
        #    self.feature_hierachies.append(n_z)
        #print("Feature hierachies: ", self.feature_hierachies)

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
        n_z_0_nodes = encoder_activation_sizes[-1][0]
        n_z_0_features = encoder_activation_sizes[-1][1]
        print("n_z_0_features: ", n_z_0_features)
        self.reparametisation_latent_0 = GraphGaussianBlock(in_nodes=n_z_0_nodes, in_features=n_z_0_features, n_z_nodes=n_z_0_nodes, n_z_features=n_z_0_features)
        self.resize_conv_0 = GraphConvolution(in_features=n_z_0_features+self.num_classes, out_features=self.residual_size, bias=True, node_n=n_z_0_nodes, out_node_n=self.node_input_n)
        #self.reshape_z0_linearly = FullyConnected(in_features=self.feature_hierachies[-1], out_features=self.residual_size, bias=True)

        self.decoder_units = []
        for i in range(len(self.encoder_activation_sizes)-2):
            rezero1 = ReZero()
            rezero2 = ReZero()
            rezero3 = ReZero()
            in_node_size = encoder_activation_sizes[-1-i][0]
            in_feature_size = encoder_activation_sizes[-1-i][1]
            out_node_size = encoder_activation_sizes[-2-i][0]
            out_feature_size = encoder_activation_sizes[-2-i][1]
            begin_decoder_block = GC_Block(self.residual_size, p_dropout, bias=True, node_n=self.node_input_n, activation=nn.GELU())

            print(encoder_activation_sizes)
            print("size of encoder used at decoder level {} is {}".format(i, encoder_activation_sizes[-2-i]))
            posterior_decoder_resize = GraphConvolution(in_features=self.residual_size, out_features=out_feature_size, bias=True, node_n=self.node_input_n, out_node_n=out_node_size)
            posterior_decoder_block = GC_Block((out_feature_size+out_feature_size), p_dropout, bias=True, node_n=out_node_size, activation=nn.GELU())
            reparametisation_posterior = GraphGaussianBlock(in_nodes=out_node_size, in_features=(out_feature_size+out_feature_size), n_z_nodes=out_node_size, n_z_features=out_feature_size)

            prior_decoder_block = GC_Block(self.residual_size, p_dropout, bias=True, node_n=self.node_input_n, activation=nn.GELU())
            reparametisation_prior = GraphGaussianBlock(in_nodes=self.node_input_n, in_features=self.residual_size, n_z_nodes=out_node_size, n_z_features=out_feature_size)

            reshape_z = GraphConvolution(in_features=out_feature_size, out_features=self.residual_size, bias=True, node_n=out_node_size, out_node_n=self.node_input_n)

            self.decoder_units.append({
                "decoder_block":begin_decoder_block,
                "posterior_decoder_resize":posterior_decoder_resize,
                "posterior_decoder_block":posterior_decoder_block,
                "reparametisation_posterior":reparametisation_posterior,
                "prior_decoder_block":prior_decoder_block,
                "reparametisation_prior":reparametisation_prior,
                "reshape_z":reshape_z,
                "rezero1":rezero1,
                "rezero2":rezero2,
                "rezero3":rezero3
            })
            self.decoder_units[i] = nn.ModuleDict(self.decoder_units[i])
        self.decoder_units = nn.ModuleList(self.decoder_units)

        self.decoder_block_final = GC_Block(self.residual_size, p_dropout, bias=True, node_n=self.node_input_n, activation=nn.GELU())
        self.reparametisation_output = GraphGaussianBlock(in_nodes=self.node_input_n, in_features=self.residual_size, n_z_nodes=self.node_input_n, n_z_features=self.input_temp_n)

    def forward(self, encoder_activations=None, z_0=None, one_hot_labels=None):
        if z_0 is None:
            self.z_posterior_mus["0"], self.z_posterior_log_vars["0"] = self.reparametisation_latent_0(encoder_activations[-1])
            self.z_prior_mus["0"], self.z_prior_log_vars["0"] = torch.zeros_like(self.z_posterior_mus["0"]), torch.zeros_like(self.z_posterior_log_vars["0"])
            self.z_mus["0"], self.z_log_vars["0"] = self.z_posterior_mus["0"], self.z_posterior_log_vars["0"]

            self.KLs["0"] = utils.kullback_leibler_divergence(self.z_mus["0"], self.z_log_vars["0"], mu_2=self.z_prior_mus["0"], log_var_2=self.z_prior_log_vars["0"], graph=True)
            self.zs["0"] = utils.reparametisation_trick(self.z_mus["0"], self.z_log_vars["0"], self.device)
        else:
            self.zs["0"] = z_0

        self.zs["0"] = torch.cat((self.zs["0"], one_hot_labels), dim=2)

        self.residuals_dict["0"] = self.resize_conv_0(self.zs["0"])

        for i in range(len(self.encoder_activation_sizes)-2):
            self.KLs[str(i+1)], self.zs[str(i+1)] = self.top_down_decode(level=i, encoder_activation=encoder_activations[-2-i])

        decoder_output_final = self.decoder_block_final(self.residuals_dict[str(len(self.encoder_activation_sizes)-2)])
        self.reconstructions_mu, self.reconstructions_log_var = self.reparametisation_output(decoder_output_final)

        sig = nn.Sigmoid()
        self.bernoulli_output = sig(self.reconstructions_mu)

        return self.reconstructions_mu, self.reconstructions_log_var, self.zs, self.KLs

    def top_down_decode(self, level, encoder_activation=None, X_supplied=True, latent_resolution=999):
        res_block_output = self.decoder_units[level]["decoder_block"](self.residuals_dict[str(level)])
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level)] + self.decoder_units[level]["rezero1"](res_block_output)

        #Posterior Route
        if X_supplied==True:
            posterior_decoder_resize_output = self.decoder_units[level]["posterior_decoder_resize"](self.residuals_dict[str(level+1)])
            concat_for_posterior = torch.cat((posterior_decoder_resize_output, encoder_activation), dim=2)
            posterior_decoder_block_output = self.decoder_units[level]["posterior_decoder_block"](concat_for_posterior)
            self.z_posterior_mus[str(level + 1)], self.z_posterior_log_vars[str(level + 1)] = self.decoder_units[level]["reparametisation_posterior"](posterior_decoder_block_output)

        #Prior route
        prior_decoder_block_output = self.decoder_units[level]["prior_decoder_block"](self.residuals_dict[str(level+1)])
        self.z_prior_mus[str(level + 1)], self.z_prior_log_vars[str(level + 1)] = self.decoder_units[level]["reparametisation_prior"](prior_decoder_block_output)
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level+1)] + self.decoder_units[level]["rezero2"](prior_decoder_block_output)

        #Sample from the posterior while training, prior while sampling
        if X_supplied==True:
            self.z_mus[str(level + 1)] = self.z_posterior_mus[str(level+1)]
            self.z_log_vars[str(level + 1)] = self.z_posterior_log_vars[str(level+1)]
        else:
            self.z_mus[str(level + 1)] = self.z_prior_mus[str(level+1)]
            self.z_log_vars[str(level + 1)] = self.z_prior_log_vars[str(level+1)]

        #Sample z_level, or take the mean
        if level < latent_resolution:
            self.KLs[str(level+1)] = utils.kullback_leibler_divergence(self.z_mus[str(level+1)], self.z_log_vars[str(level+1)], mu_2=self.z_prior_mus[str(level+1)], log_var_2=self.z_prior_log_vars[str(level+1)], graph=True)
            #print("At level {}, KL={}, kls={}".format((level+1), self.KLs[str(level+1)], self.KLs[str(level+1)]))
            self.zs[str(level+1)] = utils.reparametisation_trick(self.z_mus[str(level+1)], self.z_log_vars[str(level+1)], self.device)
        else:
            self.zs[str(level + 1)] = self.z_mus[str(level+1)]

        reshaped_z = self.decoder_units[level]["reshape_z"](self.zs[str(level+1)])
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level+1)] + self.decoder_units[level]["rezero3"](reshaped_z)

        return self.KLs[str(level+1)], self.zs[str(level+1)]

class VDDecoder(nn.Module):
    def __init__(self, input_n=96, encoder_activation_sizes=[784, 784, 16], act_fn=nn.GELU(), device="cuda", batch_norm=False, p_dropout=0.0, n_zs=[50, 10, 5, 2], residual_size=200):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VDDecoder, self).__init__()
        self.activation = act_fn
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout
        self.residual_size = residual_size

        self.feature_hierachies=[]
        self.feature_hierachies.append(input_n)
        for n_z in n_zs:
            self.feature_hierachies.append(n_z)
        print("Feature hierachies: ", self.feature_hierachies)

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
        self.reparametisation_latent_0 = GaussianBlock(encoder_activation_sizes[-1], self.feature_hierachies[-1])
        self.reshape_z0_linearly = FullyConnected(in_features=self.feature_hierachies[-1], out_features=self.residual_size, bias=True)

        self.decoder_units = []
        for i in range(len(self.feature_hierachies)-2):
            rezero1 = ReZero()
            rezero2 = ReZero()
            rezero3 = ReZero()
            begin_decoder_block = NeuralNetworkBlock(layers=utils.define_neurons_layers(self.residual_size, self.residual_size, 4), activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)

            print(encoder_activation_sizes)
            print("size of encoder used at decoder level {} is {}".format(i, encoder_activation_sizes[-2-i]))
            posterior_decoder_block = NeuralNetworkBlock(layers=utils.define_neurons_layers(encoder_activation_sizes[-2-i] + self.residual_size, self.residual_size, 4), activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
            reparametisation_posterior = GaussianBlock(self.residual_size, self.feature_hierachies[-2-i])

            prior_decoder_block = NeuralNetworkBlock(layers=utils.define_neurons_layers(self.residual_size, self.residual_size, 4), activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
            reparametisation_prior = GaussianBlock(self.residual_size, self.feature_hierachies[-2-i])

            reshape_z_linearly = FullyConnected(in_features=self.feature_hierachies[-2-i], out_features=self.residual_size, bias=True)

            self.decoder_units.append({
                "decoder_block":begin_decoder_block,
                "posterior_decoder_block":posterior_decoder_block,
                "reparametisation_posterior":reparametisation_posterior,
                "prior_decoder_block":prior_decoder_block,
                "reparametisation_prior":reparametisation_prior,
                "reshape_z_linearly":reshape_z_linearly,
                "rezero1":rezero1,
                "rezero2":rezero2,
                "rezero3":rezero3
            })
            self.decoder_units[i] = nn.ModuleDict(self.decoder_units[i])
        self.decoder_units = nn.ModuleList(self.decoder_units)

        self.decoder_block_final = NeuralNetworkBlock(layers=utils.define_neurons_layers(self.residual_size, self.residual_size, 4), activation=self.activation, batch_norm=batch_norm, p_dropout=p_dropout)
        self.reparametisation_output = GaussianBlock(self.residual_size, input_n)

    def forward(self, encoder_activations=None, z_0=None):
        if z_0 is None:
            self.z_posterior_mus["0"], self.z_posterior_log_vars["0"] = self.reparametisation_latent_0(encoder_activations[-1])
            self.z_prior_mus["0"], self.z_prior_log_vars["0"] = torch.zeros_like(self.z_posterior_mus["0"]), torch.zeros_like(self.z_posterior_log_vars["0"])
            self.z_mus["0"], self.z_log_vars["0"] = self.z_posterior_mus["0"], self.z_posterior_log_vars["0"]

            self.KLs["0"] = utils.kullback_leibler_divergence(self.z_mus["0"], self.z_log_vars["0"], mu_2=self.z_prior_mus["0"], log_var_2=self.z_prior_log_vars["0"])
            self.zs["0"] = utils.reparametisation_trick(self.z_mus["0"], self.z_log_vars["0"], self.device)
        else:
            self.zs["0"] = z_0

        self.residuals_dict["0"] = self.reshape_z0_linearly(self.zs["0"])

        for i in range(len(self.feature_hierachies)-2):
            self.KLs[str(i+1)], self.zs[str(i+1)] = self.top_down_decode(level=i, encoder_activation=encoder_activations[-2-i])

        decoder_output_final = self.decoder_block_final(self.residuals_dict[str(len(self.feature_hierachies)-2)])
        self.reconstructions_mu, self.reconstructions_log_var = self.reparametisation_output(decoder_output_final)

        sig = nn.Sigmoid()
        self.bernoulli_output = sig(self.reconstructions_mu)

        return self.reconstructions_mu, self.reconstructions_log_var, self.zs, self.KLs

    def top_down_decode(self, level, encoder_activation=None, X_supplied=True, latent_resolution=999):
        res_block_output = self.decoder_units[level]["decoder_block"](self.residuals_dict[str(level)])
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level)] + self.decoder_units[level]["rezero1"](res_block_output)

        #Posterior Route
        if X_supplied==True:
            concat_for_posterior = torch.cat((self.residuals_dict[str(level+1)], encoder_activation), dim=1)
            posterior_decoder_block_output = self.decoder_units[level]["posterior_decoder_block"](concat_for_posterior)
            self.z_posterior_mus[str(level + 1)], self.z_posterior_log_vars[str(level + 1)] = self.decoder_units[level]["reparametisation_posterior"](posterior_decoder_block_output)

        #Prior route
        prior_decoder_block_output = self.decoder_units[level]["prior_decoder_block"](self.residuals_dict[str(level+1)])
        self.z_prior_mus[str(level + 1)], self.z_prior_log_vars[str(level + 1)] = self.decoder_units[level]["reparametisation_prior"](prior_decoder_block_output)
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level+1)] + self.decoder_units[level]["rezero2"](prior_decoder_block_output)

        #Sample from the posterior while training, prior while sampling
        if X_supplied==True:
            self.z_mus[str(level + 1)] = self.z_posterior_mus[str(level+1)]
            self.z_log_vars[str(level + 1)] = self.z_posterior_log_vars[str(level+1)]
        else:
            self.z_mus[str(level + 1)] = self.z_prior_mus[str(level+1)]
            self.z_log_vars[str(level + 1)] = self.z_prior_log_vars[str(level+1)]

        #Sample z_level, or take the mean
        if level < latent_resolution:
            self.KLs[str(level+1)] = utils.kullback_leibler_divergence(self.z_mus[str(level+1)], self.z_log_vars[str(level+1)], mu_2=self.z_prior_mus[str(level+1)], log_var_2=self.z_prior_log_vars[str(level+1)])
            self.zs[str(level+1)] = utils.reparametisation_trick(self.z_mus[str(level+1)], self.z_log_vars[str(level+1)], self.device)
        else:
            self.zs[str(level + 1)] = self.z_mus[str(level+1)]

        reshaped_z = self.decoder_units[level]["reshape_z_linearly"](self.zs[str(level+1)])
        self.residuals_dict[str(level+1)] = self.residuals_dict[str(level+1)] + self.decoder_units[level]["rezero3"](reshaped_z)

        return self.KLs[str(level+1)], self.zs[str(level+1)]



class VAE_Decoder(nn.Module):
    def __init__(self, layers = [2, 50, 100, 48], activation=nn.LeakyReLU(0.1), output_variance=False, device="cuda", batch_norm=False, p_dropout=0.0):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VAE_Decoder, self).__init__()
        self.device = device

        self.n_x = layers[-1]
        self.n_z = layers[0]
        self.layers = np.array(layers)
        self.n_layers = self.layers.shape[0]-1
        self.output_variance = output_variance

        self.fc_blocks = []
        for i in range(self.n_layers-1):
            self.fc_blocks.append(FC_Block(self.layers[i], self.layers[i + 1], activation=activation, batch_norm=batch_norm, p_dropout=p_dropout, bias=True))
        self.fc_blocks = nn.ModuleList(self.fc_blocks)

        self.reconstructions_mu_fc = FullyConnected(self.layers[-2], self.n_x)
        if self.output_variance:
            self.reconstructions_log_var_fc = FullyConnected(self.layers[-2], self.n_x)

    def forward(self, x):
        y = x
        for i in range(self.n_layers-1):
            y = self.fc_blocks[i](y)

        reconstructions_mu = self.reconstructions_mu_fc(y)
        if self.output_variance:
            reconstructions_log_var = self.reconstructions_log_var_fc(y)
            reconstructions_log_var = torch.clamp(reconstructions_log_var, min=-20.0, max=3.0)
            return reconstructions_mu, reconstructions_log_var
        else:
            return reconstructions_mu



class VGAE_Decoder(nn.Module):
    def __init__(self, n_z, output_feature, hidden_feature, p_dropout, num_stage=6, node_n=48, hybrid=True):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(VGAE_Decoder, self).__init__()
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