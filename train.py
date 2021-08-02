#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

import experiments.utils as experiment_utils

from tqdm.auto import tqdm

import models.utils as model_utils

from utils import supervised_data

def initialise(model, start_epoch=1, folder_name="", lr=0.0001, beta=1.0, l2_reg=1e-4, train_batch_size=100,
                figs_checkpoints_save_freq=10, warmup_time=0, beta_final=1.0):
    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    model.folder_name = folder_name
    model.lr = lr
    model.l2_reg = 1e-4
    model.beta = beta
    model.train_batch_size = train_batch_size
    model.clipping_value = 100.0
    model.figs_checkpoints_save_freq = figs_checkpoints_save_freq
    model.epoch_cur = start_epoch
    model.warmup_time = warmup_time
    model.beta_final = beta_final
    if start_epoch == 1:
        model.losses_file_exists = False
        model.kls_file_exists = False
        model_utils.book_keeping(model, start_epoch=start_epoch)
    else:
        model.losses_file_exists = True
        model.kls_file_exists = True
        model_utils.book_keeping(model, start_epoch=start_epoch)
        ckpt_path = model.folder_name + '/checkpoints/' + 'ckpt_' + str(start_epoch - 1) + '_weights.path.tar'
        ckpt = torch.load(ckpt_path, map_location=torch.device(model.device))
        model.load_state_dict(ckpt['state_dict'])

def train_epoch_mnist(model, train_loader, use_bernoulli_loss=False):
    model.train()
    i = 0
    for image, labels in tqdm(train_loader):
        i += 1
        cur_batch_size = len(image)

        image_flattened = image.reshape(cur_batch_size, -1).to(model.device).float()

        mu_hat, logvar_hat, zs, kls = model(image_flattened.float())
        if use_bernoulli_loss:
            loss = model.cal_loss(image_flattened, mu_hat, logvar_hat, kls, 'bernoulli')
        else:
            loss = model.cal_loss(image_flattened, mu_hat, logvar_hat, kls, 'gaussian')

        model.optimizer.zero_grad()
        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.clipping_value)
        model.writer.add_scalar("Gradients/total_gradient_norm", total_norm, model.epoch_cur)
        #if (total_norm < 150) or (model.epoch_cur < 50):
        model.optimizer.step()

    model.beta = model_utils.warmup(model, model.beta, warmup_time=model.warmup_time, beta_final=model.beta_final)


def eval_full_batch_mnist(model, loader, dataset_name='val', use_bernoulli_loss=False):
    with torch.no_grad():
        model.eval()
        i = 0
        for image, labels in tqdm(loader):
            i += 1
            cur_batch_size = len(image)

            image_flattened = image.reshape(cur_batch_size, -1).to(model.device).float()

            mu_hat, logvar_hat, zs, kls = model(image_flattened.float())
            if use_bernoulli_loss:
                loss = model.cal_loss(image_flattened, mu_hat, logvar_hat, kls, 'bernoulli')
                sig = nn.Sigmoid()
                reconstructions = sig(mu_hat)
            else:
                loss = model.cal_loss(image_flattened, mu_hat, logvar_hat, kls, 'gaussian')

            model_utils.accum_update(model, str(dataset_name) + '_loss', loss)
            model_utils.accum_update(model, str(dataset_name) + '_recon', model.recon_loss)
            if model.variational:
                model_utils.accum_update(model, str(dataset_name) + '_VLB', model.VLB)
                model_utils.accum_update(model, str(dataset_name) + '_KL', model.KL)
                for key, value in model.KLs.items():
                    model_utils.accum_update(model, str(dataset_name) + '_KL_' + str(key), value)


        model_utils.log_epoch_values(model, dataset_name)

        if model.epoch_cur % model.figs_checkpoints_save_freq == 0:
            experiment_utils.mnist_reconstructions(model, image, reconstructions, dataset_name, cur_batch_size)


def train_epoch(model, train_loader):
    model.train()
    for i, (all_seq) in enumerate(tqdm(train_loader)):
        inputs = all_seq.to(model.device).float()

        _ = model(inputs.float())
        loss = model.cal_loss(inputs, 'gaussian')

        model.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.clipping_value)
        model.writer.add_scalar("Gradients/total_gradient_norm", total_norm, model.epoch_cur)
        #if (total_norm < 150) or (model.epoch_cur < 50):
        model.optimizer.step()

    model.beta = model_utils.warmup(model, model.beta, warmup_time=model.warmup_time, beta_final=model.beta_final)


def eval_full_batch(model, loader, dataset_name='val'):
    with torch.no_grad():
        model.eval()
        for i, (all_seq) in enumerate(tqdm(loader)):
            cur_batch_size = len(all_seq)

            inputs = all_seq.to(model.device).float()

            mu = model(inputs.float())
            loss = model.cal_loss(inputs, 'gaussian')

            model_utils.accum_update(model, str(dataset_name)+'_loss', loss)
            model_utils.accum_update(model, str(dataset_name)+'_recon', model.recon_loss)
            if model.variational:
                model_utils.accum_update(model, str(dataset_name)+'_KL', model.KL)
                for key, value in model.KLs.items():
                    model_utils.accum_update(model, str(dataset_name) + '_KL_' + str(key), value)

        model_utils.log_epoch_values(model, dataset_name)

        if model.epoch_cur % model.figs_checkpoints_save_freq == 0:
            experiment_utils.poses_visualisations(model, inputs, mu, dataset_name, cur_batch_size)


def train_motion_epoch(model, train_loader, use_dct=False):
    model.train()
    for i, (all_seq, labels) in enumerate(tqdm(train_loader)):

        print(len(labels))
        print(labels[0], labels[2], labels[-1])
        print(all_seq.shape)

        class2idx, idx2class, num_classes = supervised_data.initialise_motion_class_and_index_map()
        labels = pd.DataFrame(labels)
        labels.replace(class2idx, inplace=True)
        labels = torch.from_numpy(np.array(labels)).long()
        labels = F.one_hot(labels, num_classes=num_classes)

        print(len(labels))
        print(labels)
        print(all_seq.shape)

        b_n, f_n, t_n = all_seq.shape
        model.b_n, model.f_n, model.t_n = b_n, f_n, t_n

        inputs = all_seq.to(model.device).float()
        if use_dct:
            inputs_dct = model_utils.dct(model, inputs)

            mu_hat, logvar_hat, zs, kls = model(inputs_dct.float())

            inputs_hat = model_utils.dct(model, mu_hat, inverse=True)
        else:
            inputs = inputs.reshape(b_n, f_n * t_n)
            inputs_hat, logvar_hat, zs, kls = model(inputs.float())
        loss = model.cal_loss(inputs, inputs_hat, logvar_hat, kls, 'gaussian')

        model.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.clipping_value)
        model.writer.add_scalar("Gradients/total_gradient_norm", total_norm, model.epoch_cur)
        #if (total_norm < 150) or (model.epoch_cur < 50):
        model.optimizer.step()

    model.beta = model_utils.warmup(model, model.beta, warmup_time=model.warmup_time, beta_final=model.beta_final)

def eval_motion_batch(model, loader, dataset_name='val', use_dct=False):
    with torch.no_grad():
        model.eval()
        for i, (all_seq, labels) in enumerate(tqdm(loader)):

            b_n, f_n, t_n = all_seq.shape

            inputs = all_seq.to(model.device).float()
            if use_dct:
                inputs_dct = model_utils.dct(model, inputs)

                mu_hat, logvar_hat, zs, kls = model(inputs_dct.float())

                inputs_hat = model_utils.dct(model, mu_hat, inverse=True)
            else:
                inputs = inputs.reshape(b_n, f_n * t_n)
                inputs_hat, logvar_hat, zs, kls = model(inputs.float())
            loss = model.cal_loss(inputs, inputs_hat, logvar_hat, kls, 'gaussian')

            model_utils.accum_update(model, str(dataset_name)+'_loss', loss)
            model_utils.accum_update(model, str(dataset_name)+'_recon', model.recon_loss)
            if model.variational:
                model_utils.accum_update(model, str(dataset_name)+'_KL', model.KL)
                for key, value in model.KLs.items():
                    model_utils.accum_update(model, str(dataset_name) + '_KL_' + str(key), value)

        model_utils.log_epoch_values(model, dataset_name)

        if model.epoch_cur % model.figs_checkpoints_save_freq == 0:
            inputs = inputs.reshape(b_n, f_n, t_n)
            inputs_hat = inputs_hat.reshape(b_n, f_n, t_n)
            file_path = model.folder_name + '/poses/' + str(dataset_name) + '_latest_' + 'poses_yz'
            experiment_utils.plot_motion(inputs.detach().cpu().numpy(), inputs_hat.detach().cpu().numpy(), azim=0, evl=-0, save_as=file_path)
            file_path = model.folder_name + '/poses/' + str(dataset_name) + '_latest_' + 'poses_xz'
            experiment_utils.plot_motion(inputs.detach().cpu().numpy(), inputs_hat.detach().cpu().numpy(), azim=0, evl=-90, save_as=file_path)
            file_path = model.folder_name + '/poses/' + str(dataset_name) + '_latest_' + 'poses_xy'
            experiment_utils.plot_motion(inputs.detach().cpu().numpy(), inputs_hat.detach().cpu().numpy(), azim=90, evl=-90, save_as=file_path)

