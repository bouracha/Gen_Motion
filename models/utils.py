import torch
import sys
import os

import pandas as pd
import numpy as np

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

def accum_update(model, key, val):
    if key not in model.accum_loss.keys():
        model.accum_loss[key] = AccumLoss()
    val = val.cpu().data.numpy()
    model.accum_loss[key].update(val)

def accum_reset(model):
    for key in model.accum_loss.keys():
        model.accum_loss[key].reset()

def num_parameters_and_place_on_device(model):
    print(model)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(">>> total params: {:.2f}M".format(num_parameters / 1000000.0))
    if model.device == "cuda":
        print("Moving model to GPU")
        model.cuda()
    else:
        print("Using CPU")
    return num_parameters

def define_neurons_layers(n_z_pre, n_z_next, num_layers):
    """

    :param n_z_pre: the layer with the higher number of neurons
    :param n_z_next: the layer with the lower number of neurons
    :param num_layers: number of layers desired in between
    :return: list of layer sizes
    """
    nn_layers = np.linspace(n_z_pre, n_z_next, num_layers)
    nn_layers = list(map(int, nn_layers))
    return nn_layers


# ===============================================================
#                     VAE specific functions
# ===============================================================

def reparametisation_trick(mu, log_var, device):
    """

    :param mu: The mean of the latent variable to be formed (nbatch, n_z)
    :param log_var: The log variance of the latent variable to be formed (nbatch, n_z)
    :param device: CPU or GPU
    :return: latent variable (nbatch, n_z)
    """
    noise = torch.normal(mean=0, std=1.0, size=log_var.shape).to(torch.device(device))
    z = mu + torch.mul(torch.exp(log_var / 2.0), noise)

    return z

def kullback_leibler_divergence(mu_1, log_var_1, mu_2=None, log_var_2=None):
    """

    :param mu: The mean of the latent variable to be formed (nbatch, n_z)
    :param log_var: The log variance of the latent variable to be formed (nbatch, n_z)
    :return: gaussian analytical KL divergence for each datapoint averaged across the
    batch between two gaussian distributions p and q where by default q is N(0,1)
    """
    if mu_2 is None and log_var_2 is None:
        mu_2 = torch.zeros_like(mu_1)
        log_var_2 = torch.ones_like(log_var_1)
    KL_per_datapoint = 0.5 * torch.sum(-1 + log_var_2 - log_var_1 + torch.exp(log_var_1) + torch.pow((mu_1 - mu_2), 2)/(torch.exp(log_var_2)), axis=1)
    KL = torch.mean(KL_per_datapoint)

    return KL

# ===============================================================
#                     Probabilities
# ===============================================================

def cal_gauss_log_lik(x, mu, log_var=0.0):
    """
    :param x: batch of inputs (bn X fn)
    :return: gaussian log likelihood, and the mean squared error
    """
    MSE = torch.pow((mu - x), 2)
    gauss_log_lik = -0.5*(log_var + np.log(2*np.pi) + (MSE/(1e-8 + torch.exp(log_var))))
    MSE = torch.mean(torch.sum(MSE, axis=1))
    gauss_log_lik = torch.mean(torch.sum(gauss_log_lik, axis=1))

    return gauss_log_lik, MSE

def cal_bernoulli_log_lik(x, logits):
    """
    :param x: batch of inputs (bn X fn)
    :return: gaussian log likelihood, and the mean squared error (scalar)
    """
    BCE = torch.maximum(logits, torch.zeros_like(logits)) - torch.multiply(logits, x) + torch.log(1 + torch.exp(-torch.abs(logits)))
    BCE_per_sample = torch.sum(BCE, axis=1)
    BCE_avg_for_batch = torch.mean(BCE_per_sample)
    bernoulli_log_lik = -BCE_avg_for_batch

    return bernoulli_log_lik


def cal_VLB(p_log_x, KL, beta=1.0):
    """
    :param x: batch of inputs
    :return: Variational Lower Bound
    """
    VLB = p_log_x - beta*KL

    return VLB

# ===============================================================
#                     Generic bookkeeping functions
# ===============================================================

def book_keeping(model, start_epoch=1, train_batch_size=100, l2_reg=1e-4):
    model.accum_loss = dict()

    if model.variational:
        model.folder_name = model.folder_name+"_VAE"
    else:
        model.folder_name = model.folder_name+"_AE"
    if start_epoch==1:
        os.makedirs(os.path.join(model.folder_name, 'checkpoints'))
        os.makedirs(os.path.join(model.folder_name, 'images'))
        os.makedirs(os.path.join(model.folder_name, 'poses'))
        write_type='w'
    else:
        write_type = 'a'

    original_stdout = sys.stdout
    with open(str(model.folder_name)+'/'+'architecture.txt', write_type) as f:
        sys.stdout = f
        if start_epoch==1:
            print(model)
        print("Start epoch:{}".format(start_epoch))
        print("Learning rate:{}".format(model.lr))
        print("Training batch size:{}".format(train_batch_size))
        print("Clipping value:{}".format(model.clipping_value))
        print("BN:{}".format(model.batch_norm))
        print("l2 Reg (1e-4):{}".format(l2_reg))
        print("p_dropout:{}".format(model.p_dropout))
        print("Output variance:{}".format(model.output_variance))
        print("Beta(downweight of KL):{}".format(model.beta))
        print("Activation function:{}".format(model.activation))
        print("Num_parameters:{}".format(model.num_parameters))
        sys.stdout = original_stdout

    model.head = []
    model.ret_log = []

def save_checkpoint_and_csv(model, epoch):
    df = pd.DataFrame(np.expand_dims(model.ret_log, axis=0))
    if model.losses_file_exists:
        with open(model.folder_name + '/' + 'losses.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)
    else:
        df.to_csv(model.folder_name + '/' + 'losses.csv', header=model.head, index=False)
        model.losses_file_exists = True
    state = {'epoch': epoch + 1,
             'err': model.accum_loss['train_loss'].avg,
             'state_dict': model.state_dict()}
    if epoch % model.figs_checkpoints_save_freq == 0:
        print("Saving checkpoint....")
        file_path = model.folder_name + '/checkpoints/' + 'ckpt_' + str(epoch) + '_weights.path.tar'
        torch.save(state, file_path)
    model.head = []
    model.ret_log = []
    accum_reset(model)