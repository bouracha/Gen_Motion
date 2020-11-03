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

from model import MODEL_METHODS

def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    if opt.out_of_distribution != None:
        out_of_distribution = True
        acts_train = data_utils.define_actions(opt.out_of_distribution, opt.dataset, out_of_distribution=False)
        acts_OoD = data_utils.define_actions(opt.out_of_distribution, opt.dataset, out_of_distribution=True)
        acts_test = data_utils.define_actions('all', opt.dataset, out_of_distribution=False)
    else:
        out_of_distribution = False
        acts_train = data_utils.define_actions('all', opt.dataset, out_of_distribution=False)
        acts_OoD = None
        acts_test = data_utils.define_actions('all', opt.dataset, out_of_distribution=False)

    # define log csv file
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + "_{}_in{:d}_out{:d}_dctn{:d}_dropout_{}".format(str(opt.dataset), opt.input_n, opt.output_n, opt.dct_n, str(opt.dropout))
    if out_of_distribution:
        script_name = script_name + "_OoD_{}_".format(str(opt.out_of_distribution))
    if opt.variational:
        script_name = script_name + "_var_lambda_{}_nz_{}_lr_{}_n_layers_{}".format(str(opt.lambda_), str(opt.n_z), str(opt.lr), str(opt.num_decoder_stage))

    # data loading
    print(">>> loading data")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    sample_rate = opt.sample_rate

    data = DATA(opt.dataset, opt.data_dir)
    data.get_dct_and_sequences(acts_train, input_n, output_n, sample_rate, dct_n, out_of_distribution, acts_OoD, acts_test)
    train_loader, val_loader, OoD_val_loader, test_loaders = data.get_dataloaders(opt.train_batch, opt.test_batch, acts_test, opt.job)
    print(">>> data loaded !")
    print(">>> train data {}".format(data.train_dataset.__len__()))
    if opt.dataset=='h3.6m':
      print(">>> validation data {}".format(data.val_dataset.__len__()))


    print(">>> creating model")
    model = nnmodel.GCN(input_feature=dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=data.node_n, variational=opt.variational, n_z=opt.n_z, num_decoder_stage=opt.num_decoder_stage)
    methods = MODEL_METHODS(model)

    if is_cuda:
        model.cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if opt.is_load:
        model_path_len = 'checkpoint/test/ckpt_main_in10_out10_dctn20_var__last.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))







    for epoch in range(start_epoch, opt.epochs):

        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l, t_l_joint, t_l_vlb, t_l_latent, t_e, t_3d = methods.train(train_loader, optimizer, dataset=opt.dataset, input_n=input_n,
                                                              lr_now=lr_now, cartesian=data.cartesian, lambda_=opt.lambda_,max_norm=opt.max_norm, is_cuda=is_cuda,
                                                              dim_used=data.train_dataset.dim_used, dct_n=dct_n)
        ret_log = np.append(ret_log, [lr_now, t_l, t_l_joint, t_l_vlb, t_l_latent, t_e, t_3d])
        head = np.append(head, ['lr', 't_l', 't_l_joint', 't_l_vlb', 't_l_latent', 't_e', 't_3d'])

        if opt.dataset=='h3.6m':
          v_e, v_3d = methods.val(val_loader, input_n=input_n, is_cuda=is_cuda, dim_used=train_dataset.dim_used,
                          dct_n=dct_n)
          ret_log = np.append(ret_log, [v_e, v_3d])
          head = np.append(head, ['v_e', 'v_3d'])
          if not np.isnan(v_e):
            is_best = v_e < err_best
            err_best = min(v_e, err_best)
          else:
            is_best = False

          if out_of_distribution:
              OoD_v_e, OoD_v_3d = methods.val(OoD_val_loader, input_n=input_n, is_cuda=is_cuda, dim_used=train_dataset.dim_used,
                          dct_n=dct_n)
              ret_log = np.append(ret_log, [OoD_v_e, OoD_v_3d])
              head = np.append(head, ['OoD_v_e', 'OoD_v_3d'])
        # If not h3.6 dataset, select best on train error
        else:
          if not np.isnan(t_e):
            is_best = t_e < err_best
            err_best = min(t_e, err_best)
          else:
            is_best = False


        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts_test:
            test_e, test_3d = methods.test(test_loaders[act], dataset=opt.dataset, input_n=input_n, output_n=output_n, cartesian=cartesian, is_cuda=is_cuda, dim_used=train_dataset.dim_used, dct_n=dct_n)
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

        # update log file and save checkpoint
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
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)




if __name__ == "__main__":
    option = Options().parse()
    main(option)
