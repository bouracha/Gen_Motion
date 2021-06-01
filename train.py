#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch

import models.utils as utils
import experiments.utils as experiment_utils
from models.encoders import VAE_Encoder
from models.decoders import VAE_Decoder

from progress.bar import Bar
import time
from torch.autograd import Variable
from tqdm.auto import tqdm
import os
import sys
import numpy as np
import pandas as pd



def train_epoch_mnist(model, epoch, train_loader, use_bernoulli_loss=False):
    model.train()
    i = 0
    for image, labels in tqdm(train_loader):
        i += 1
        cur_batch_size = len(image)

        image_flattened = image.reshape(cur_batch_size, -1).to(model.device).float()

        _ = model(image_flattened.float())
        if use_bernoulli_loss:
            loss = model.loss_bernoulli(image_flattened)
        else:
            loss = model.loss_VLB(image_flattened)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    head = ['Epoch']
    ret_log = [epoch]
    model.head = np.append(model.head, head)
    model.ret_log = np.append(model.ret_log, ret_log)


def eval_full_batch_mnist(model, loader, epoch, dataset_name='val', use_bernoulli_loss=False):
    with torch.no_grad():
        model.eval()
        i = 0
        for image, labels in tqdm(loader):
            i += 1
            cur_batch_size = len(image)

            image_flattened = image.reshape(cur_batch_size, -1).to(model.device).float()

            reconstructions = model(image_flattened.float())
            if use_bernoulli_loss:
                loss = model.loss_bernoulli(image_flattened)
                sig = nn.Sigmoid()
                reconstructions = sig(reconstructions)
            else:
                loss = model.loss_VLB(image_flattened)

            model.accum_update(str(dataset_name) + '_loss', loss)
            model.accum_update(str(dataset_name) + '_recon', model.recon_loss)
            if model.variational:
                model.accum_update(str(dataset_name) + '_VLB', model.VLB)
                model.accum_update(str(dataset_name) + '_KL', model.KL)

        head = [dataset_name + '_loss', dataset_name + '_reconstruction']
        ret_log = [model.accum_loss[str(dataset_name) + '_loss'].avg, model.accum_loss[str(dataset_name) + '_recon'].avg]
        if model.variational:
            head.append(str(dataset_name) + '_VLB')
            head.append(str(dataset_name) + '_KL')
            ret_log.append(model.accum_loss[str(dataset_name) + '_VLB'].avg)
            ret_log.append(model.accum_loss[str(dataset_name) + '_KL'].avg)
        model.head = np.append(model.head, head)
        model.ret_log = np.append(model.ret_log, ret_log)

        if epoch % model.figs_checkpoints_save_freq == 0:
            reconstructions = reconstructions.reshape(cur_batch_size, 1, 28, 28)
            file_path = model.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reals'
            experiment_utils.plot_tensor_images(image, max_num_images=25, nrow=5, show=False, save_as=file_path)
            file_path = model.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reconstructions'
            experiment_utils.plot_tensor_images(reconstructions, max_num_images=25, nrow=5, show=False,
                                                save_as=file_path)

def train_epoch(model, epoch, train_loader):
    model.train()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    st = time.time()
    for i, (all_seq) in enumerate(train_loader):
        bt = time.time()

        inputs = all_seq.to(model.device).float()

        _ = model(inputs.float())
        loss = model.loss_VLB(inputs)

        model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.clipping_value)
        model.optimizer.step()

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt, time.time() - st)
        bar.next()

    head = ['Epoch']
    ret_log = [epoch]
    model.head = np.append(model.head, head)
    model.ret_log = np.append(model.ret_log, ret_log)

    bar.finish()


def eval_full_batch(model, loader, epoch, dataset_name='val'):
    with torch.no_grad():
        model.eval()
        bar = Bar('>>>', fill='>', max=len(loader))
        st = time.time()
        for i, (all_seq) in enumerate(loader):
            bt = time.time()
            cur_batch_size = len(all_seq)

            inputs = all_seq.to(model.device).float()

            mu = model(inputs.float())
            loss = model.loss_VLB(inputs)

            model.accum_update(str(dataset_name)+'_loss', loss)
            model.accum_update(str(dataset_name)+'_recon_loss', model.recon_loss)
            if model.variational:
                model.accum_update(str(dataset_name)+'_KL', model.KL)

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(loader), time.time() - bt,
                                                                                      time.time() - st)
            bar.next()
        bar.finish()

        head = [dataset_name+'_loss', dataset_name+'_reconstruction']
        ret_log = [model.accum_loss[str(dataset_name)+'_loss'].avg, model.accum_loss[str(dataset_name)+'_recon_loss'].avg]
        if model.variational:
            head.append(str(dataset_name)+'_KL')
            ret_log.append(model.accum_loss[str(dataset_name)+'_KL'].avg)
        model.head = np.append(model.head, head)
        model.ret_log = np.append(model.ret_log, ret_log)

        inputs_reshaped = inputs.reshape(cur_batch_size, 1, 12, 8)
        reconstructions = mu.reshape(cur_batch_size, 1, 12, 8)
        diffs = inputs_reshaped - reconstructions

        if epoch % model.figs_checkpoints_save_freq == 0:
            file_path = model.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reals'
            experiment_utils.plot_tensor_images(inputs_reshaped.detach().cpu(), max_num_images=25, nrow=5, show=False, save_as=file_path)
            file_path = model.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'reconstructions'
            experiment_utils.plot_tensor_images(reconstructions.detach().cpu(), max_num_images=25, nrow=5, show=False, save_as=file_path)
            file_path = model.folder_name + '/images/' + str(dataset_name) + '_' + str(epoch) + '_' + 'diffs'
            experiment_utils.plot_tensor_images(diffs.detach().cpu(), max_num_images=25, nrow=5, show=False, save_as=file_path)
            file_path = model.folder_name + '/poses/' + str(dataset_name) + '_' + str(epoch) + '_' + 'poses_xz'
            experiment_utils.plot_poses(inputs.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=25, azim=0, evl=90, save_as=file_path)
            file_path = model.folder_name + '/poses/' + str(dataset_name) + '_' + str(epoch) + '_' + 'poses_yz'
            experiment_utils.plot_poses(inputs.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=25, azim=0, evl=-0, save_as=file_path)
            file_path = model.folder_name + '/poses/' + str(dataset_name) + '_' + str(epoch) + '_' + 'poses_xy'
            experiment_utils.plot_poses(inputs.detach().cpu().numpy(), mu.detach().cpu().numpy(), max_num_images=25, azim=90, evl=90, save_as=file_path)