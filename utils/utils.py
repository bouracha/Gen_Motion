#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
import numpy as np

import matplotlib.pyplot as plt

def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_ckpt(state, ckpt_path, is_best=True, file_name=['ckpt_best.pth.tar', 'ckpt_last.pth.tar']):
    file_path = os.path.join(ckpt_path, file_name[1])
    torch.save(state, file_path)
    if is_best:
        file_path = os.path.join(ckpt_path, file_name[0])
        torch.save(state, file_path)

def check_is_best(err, err_best):
    if not np.isnan(err):
        is_best = err < err_best
        err_best = min(err, err_best)
    else:
        is_best = False
    return is_best, err_best