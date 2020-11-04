#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""overall code framework is adapped from https://github.com/weigq/3d_pose_baseline_pytorch"""
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim

from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options

import utils.GCN_Architecture as nnmodel
import utils.data_utils as data_utils

from data import DATA

from model_methods import MODEL_METHODS

def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    print(">>> loading data")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    sample_rate = opt.sample_rate

    #####################################################
    # Define script name
    #####################################################
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + "_{}_in{:d}_out{:d}_dctn{:d}_dropout_{}".format(str(opt.dataset), opt.input_n, opt.output_n, opt.dct_n, str(opt.dropout))
    if out_of_distribution:
        script_name = script_name + "_OoD_{}_".format(str(opt.out_of_distribution))
    if opt.variational:
        script_name = script_name + "_var_lambda_{}_nz_{}_lr_{}_n_layers_{}".format(str(opt.lambda_), str(opt.n_z), str(opt.lr), str(opt.num_decoder_stage))


    #####################################################
    # Load data
    #####################################################
    data = DATA(opt.dataset, opt.data_dir)
    out_of_distribution = data.get_dct_and_sequences(input_n, output_n, sample_rate, dct_n, opt.out_of_distribution)
    train_loader, val_loader, OoD_val_loader, test_loaders = data.get_dataloaders(opt.train_batch, opt.test_batch, opt.job)
    print(">>> data loaded !")
    print(">>> train data {}".format(data.train_dataset.__len__()))
    if opt.dataset=='h3.6m':
      print(">>> validation data {}".format(data.val_dataset.__len__()))

    ##################################################################
    # Instantiate model, and methods used fro training and valdation
    ##################################################################
    print(">>> creating model")
    model = nnmodel.GCN(input_feature=dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=data.node_n, variational=opt.variational, n_z=opt.n_z, num_decoder_stage=opt.num_decoder_stage)
    methods = MODEL_METHODS(model, is_cuda)
    if opt.is_load:
      start_epoch, err_best, lr_now = methods.load_weights(opt.load_path)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    methods.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(start_epoch, opt.epochs):
        #####################################################################################################################################################
        # Training step
        #####################################################################################################################################################
        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(methods.optimizer, lr_now, opt.lr_gamma)
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l, t_l_joint, t_l_vlb, t_l_latent, t_e, t_3d = methods.train(train_loader, dataset=opt.dataset, input_n=input_n,
                                                              lr_now=lr_now, cartesian=data.cartesian, lambda_=opt.lambda_,max_norm=opt.max_norm,
                                                              dim_used=data.train_dataset.dim_used, dct_n=dct_n)
        ret_log = np.append(ret_log, [lr_now, t_l, t_l_joint, t_l_vlb, t_l_latent, t_e, t_3d])
        head = np.append(head, ['lr', 't_l', 't_l_joint', 't_l_vlb', 't_l_latent', 't_e', 't_3d'])

        #####################################################################################################################################################
        # Evaluate on validation set; Keep track of best, either via val set, OoD val set (in the case of OoD), or train set in the case of the CMU dataset
        #####################################################################################################################################################
        if opt.dataset == 'h3.6m':
          v_e, v_3d = methods.val(val_loader, input_n=input_n, dim_used=data.train_dataset.dim_used,
                          dct_n=dct_n)
          ret_log = np.append(ret_log, [v_e, v_3d])
          head = np.append(head, ['v_e', 'v_3d'])

          is_best, err_best = utils.check_is_best(v_e, err_best)
          if out_of_distribution:
              OoD_v_e, OoD_v_3d = methods.val(OoD_val_loader, input_n=input_n, dim_used=data.train_dataset.dim_used,
                          dct_n=dct_n)
              ret_log = np.append(ret_log, [OoD_v_e, OoD_v_3d])
              head = np.append(head, ['OoD_v_e', 'OoD_v_3d'])
        else:
          is_best, err_best = utils.check_is_best(t_e, err_best)

        #####################################################
        # Evaluate on test set
        #####################################################
        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts_test:
            test_e, test_3d = methods.test(test_loaders[act], dataset=opt.dataset, input_n=input_n, output_n=output_n, cartesian=data.cartesian, dim_used=data.train_dataset.dim_used, dct_n=dct_n)
            ret_log = np.append(ret_log, test_e)
            test_3d_temp = np.append(test_3d_temp, test_3d)
            test_3d_head = np.append(test_3d_head,
                                     [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            head = np.append(head, [act + '80', act + '160', act + '320', act + '400'])
            if output_n > 10:
                head = np.append(head, [act + '560', act + '1000'])
                test_3d_head = np.append(test_3d_head,
                                         [act + '3d560', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)

        #####################################################
        # Update log file and save checkpoint
        #####################################################
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_e[0],
                         'state_dict': model.state_dict(),
                         'optimizer': methods.optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)


if __name__ == "__main__":
    option = Options().parse()
    main(option)
