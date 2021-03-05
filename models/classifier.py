import torch.nn as nn
import torch

import models.utils as utils
from models.layers import *

import numpy as np
import pandas as pd
import os, sys

import time
from tqdm.auto import tqdm

class Classifier(nn.Module):
    def __init__(self, num_features=96, hidden_layers=[1024, 512, 128], num_classes=15, device="cuda", act_fn=nn.LeakyReLU(0.1), batch_norm=False, p_dropout=0.0):
        super(Classifier, self).__init__()
        self.activation = act_fn
        self.device = device
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout

        self.layers = self.define_layers(num_features, hidden_layers, num_classes)
        self.layers = np.array(self.layers)
        self.n_layers = self.layers.shape[0] - 1

        assert(self.layers[0] == num_features)
        assert(self.layers[-1] == num_classes)

        self.fc_blocks = []
        for i in range(self.n_layers - 1):
            self.fc_blocks.append(FC_Block(self.layers[i], self.layers[i + 1], activation=self.activation, batch_norm=self.batch_norm, p_dropout=self.p_dropout, bias=True))
        self.fc_blocks = nn.ModuleList(self.fc_blocks)

        self.layer_out = FullyConnected(self.layers[-2], self.layers[-1])

        self.num_parameters = utils.num_parameters_and_place_on_device(self)

    def forward(self, x):
        y = x
        for i in range(self.n_layers-1):
            y = self.fc_blocks[i](y)
        logits = self.layer_out(y)
        return logits

    def define_layers(self, num_features=96, hidden_layers=[100, 50], num_classes=2):
        layers = []
        layers.append(num_features)
        n_hidden = len(hidden_layers)
        for i in range(n_hidden):
            layers.append(hidden_layers[i])
        layers.append(num_classes)
        return layers

    def _initialise(self, start_epoch=1, folder_name="", lr=0.0001, l2_reg=False, train_batch_size=100, figs_checkpoints_save_freq=10):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_reg)
        self.folder_name = folder_name
        self.lr = lr
        self.clipping_value = 1
        self.figs_checkpoints_save_freq = figs_checkpoints_save_freq
        if start_epoch==1:
            self.losses_file_exists = False
            self.book_keeping(start_epoch=start_epoch, train_batch_size=train_batch_size, l2_reg=l2_reg)
        else:
            self.losses_file_exists = True
            self.book_keeping(start_epoch=start_epoch, train_batch_size=train_batch_size, l2_reg=l2_reg)
            ckpt_path = self.folder_name + '/checkpoints/' + 'ckpt_' + str(start_epoch - 1) + '_weights.path.tar'
            ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))
            self.load_state_dict(ckpt['state_dict'])

    def accum_update(self, key, val):
        if key not in self.accum_loss.keys():
            self.accum_loss[key] = utils.AccumLoss()
        val = val.cpu().data.numpy()
        self.accum_loss[key].update(val)

    def accum_reset(self):
        for key in self.accum_loss.keys():
            self.accum_loss[key].reset()

    def book_keeping(self, start_epoch=1, train_batch_size=100, l2_reg=0.0):
        self.accum_loss = dict()

        self.folder_name = self.folder_name+"_classifier"
        if start_epoch==1:
            os.makedirs(os.path.join(self.folder_name, 'checkpoints'))
            write_type='w'
        else:
            write_type = 'a'

        original_stdout = sys.stdout
        with open(str(self.folder_name)+'/'+'architecture.txt', write_type) as f:
            sys.stdout = f
            if start_epoch==1:
                print(self)
            print("Start epoch:{}".format(start_epoch))
            print("Learning rate:{}".format(self.lr))
            print("Training batch size:{}".format(train_batch_size))
            print("BN:{}".format(self.batch_norm))
            print("l2 Reg (1e-4):{}".format(l2_reg))
            print("p_dropout:{}".format(self.p_dropout))
            print("Activation function:{}".format(self.activation))
            sys.stdout = original_stdout

        self.head = []
        self.ret_log = []

    def loss_bernoulli(self, logits, labels):
        """

        :param logits: outputs of the classifier, sigmoid not applied (batch_size, num_classes)
        :param labels: ground truth labels (batch_size)
        :return: Cross-Entropy Loss (float)
        """
        cross_entropy_fn = nn.CrossEntropyLoss()
        loss = cross_entropy_fn(logits, labels)

        return loss

    def metric_accuracy(self, y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum()/(1.0*len(correct_pred))

        acc = acc * 100.0

        return acc

    # ===============================================================
    # Training and validation methods
    # ===============================================================

    def save_checkpoint_and_csv(self, epoch):
        df = pd.DataFrame(np.expand_dims(self.ret_log, axis=0))
        if self.losses_file_exists:
            with open(self.folder_name+'/'+'losses.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        else:
            df.to_csv(self.folder_name+'/'+'losses.csv', header=self.head, index=False)
            self.losses_file_exists = True
        state = {'epoch': epoch + 1,
                         'err':  self.accum_loss['train_loss'].avg,
                         'state_dict': self.state_dict()}
        if epoch % self.figs_checkpoints_save_freq == 0:
            print("Saving checkpoint....")
            file_path = self.folder_name + '/checkpoints/' + 'ckpt_' + str(epoch) + '_weights.path.tar'
            torch.save(state, file_path)
        self.head = []
        self.ret_log = []
        self.accum_reset()

    def train_epoch(self, epoch, train_loader):
        self.train()
        for X_batch, y_batch in tqdm(train_loader):
            cur_batch_size = len(X_batch)

            X_batch = X_batch.reshape(cur_batch_size, -1)
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            y_batch_pred = self(X_batch)

            loss = self.loss_bernoulli(y_batch_pred, y_batch)
            #acc = self.metric_accuracy(y_batch_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            self.optimizer.step()

        head = ['Epoch']
        ret_log = [epoch]
        self.head = np.append(self.head, head)
        self.ret_log = np.append(self.ret_log, ret_log)

    def eval_full_batch(self, loader, epoch, dataset_name='val'):
        with torch.no_grad():
            self.eval()
            for X_batch, y_batch in tqdm(loader):
                cur_batch_size = len(X_batch)

                X_batch = X_batch.reshape(cur_batch_size, -1)
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                y_batch_pred = self(X_batch)

                loss = self.loss_bernoulli(y_batch_pred, y_batch)
                acc = self.metric_accuracy(y_batch_pred, y_batch)

                self.accum_update(str(dataset_name)+'_loss', loss)
                self.accum_update(str(dataset_name)+'_acc', acc)

            head = [dataset_name+'_loss', dataset_name+'_accuracy']
            ret_log = [self.accum_loss[str(dataset_name)+'_loss'].avg, self.accum_loss[str(dataset_name)+'_acc'].avg]
            self.head = np.append(self.head, head)
            self.ret_log = np.append(self.ret_log, ret_log)
