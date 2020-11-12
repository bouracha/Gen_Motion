from utils import loss_funcs, utils as utils
from utils.opt import Options

import os
import time
import torch
import torch.nn as nn

from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd

import utils.data_utils as data_utils

class MODEL_METHODS():
    def __init__(self, architecture, is_cuda):
        self.model = architecture
        self.is_cuda = is_cuda
        clipping_value = 1
        torch.nn.utils.clip_grad_norm(self.model.parameters(), clipping_value)
        if is_cuda:
            self.model.cuda()

    def load_weights(self, model_path_len = 'checkpoint/test/ckpt_main_in10_out10_dctn20_var__last.pth.tar'):
        """ Loads weights from specified model path.

        :param model_path_len: path to pretrained model, has a default
        :return: tuple (start_epoch, err_best, lr_now)
            WHERE
            start_epoch is the next epoch after previous training ended
            err_best is the best error that was reached
            lr_now is the current learning rate when learning was terminated
        """
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if self.is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

        return start_epoch, err_best, lr_now


    def train(self, train_loader, dataset='h3.6m', input_n=20, dct_n=20, lr_now=None, cartesian=False,
              lambda_=0.01, max_norm=True, dim_used=[]):
        t_l = utils.AccumLoss()
        t_l_joint = utils.AccumLoss()
        t_l_vlb = utils.AccumLoss()
        t_l_latent = utils.AccumLoss()
        t_e = utils.AccumLoss()
        t_3d = utils.AccumLoss()

        self.model.train()
        st = time.time()
        bar = Bar('>>>', fill='>', max=len(train_loader))
        for i, (inputs, targets, all_seq) in enumerate(train_loader):

            # skip the last batch if only have one sample for batch_norm layers
            batch_size = inputs.shape[0]
            if batch_size == 1:
                continue

            bt = time.time()
            if self.is_cuda:
                inputs = Variable(inputs.cuda()).float()
                targets = Variable(targets.cuda(non_blocking=True)).float()
                all_seq = Variable(all_seq.cuda(non_blocking=True)).float()

            outputs, reconstructions, log_var, z = self.model(inputs.float())
            KL = self.model.KL
            n = outputs.shape[0]
            outputs = outputs.view(n, -1)

            loss, joint_loss, vlb, latent_loss = loss_funcs.sen_loss(outputs, all_seq, dim_used, dct_n, inputs,
                                                                     cartesian, lambda_, KL, reconstructions, log_var)

            # Print losses for epoch
            ret_log = np.array([i, loss.cpu().data.numpy(), joint_loss.cpu().data.numpy(), vlb.cpu().data.numpy(),
                                latent_loss.cpu().data.numpy()])
            df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
            if i == 0:
                head = ['iteration', 'loss', 'joint_loss', 'vlb', 'latent_loss']
                df.to_csv('losses.csv', header=head, index=False)
            with open('losses.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)

            # calculate loss and backward
            self.optimizer.zero_grad()
            loss.backward()
            if max_norm:
                nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            n, _, _ = all_seq.data.shape

            if dataset == 'h3.6m':
                # 3d error
                m_err = loss_funcs.mpjpe_error(outputs, all_seq, input_n, dim_used, dct_n)
                # angle space error
                e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used, dct_n)
            elif dataset == 'cmu_mocap':
                m_err = loss_funcs.mpjpe_error_cmu(outputs, all_seq, input_n, dim_used=dim_used, dct_n=dct_n)
                e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used=dim_used, dct_n=dct_n)
            elif dataset == 'cmu_mocap_3d':
                m_err = loss
                e_err = loss

            # update the training loss
            t_l.update(loss.cpu().data.numpy() * n, n)
            t_l_joint.update(joint_loss.cpu().data.numpy() * n, n)
            t_l_vlb.update(vlb.cpu().data.numpy() * n, n)
            t_l_latent.update(latent_loss.cpu().data.numpy() * n, n)
            t_e.update(e_err.cpu().data.numpy() * n, n)
            t_3d.update(m_err.cpu().data.numpy() * n, n)

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                             time.time() - st)
            bar.next()
        bar.finish()
        print("\nJoint loss: ", t_l_joint.avg)
        print("vlb: ", t_l_vlb.avg)
        print("Latent loss: ", t_l_latent.avg)
        print("loss: ", t_l.avg)
        return lr_now, t_l.avg, t_l_joint.avg, t_l_vlb.avg, t_l_latent.avg, t_e.avg, t_3d.avg

    def val(self, train_loader, input_n=20, dct_n=20, dim_used=[]):
        # t_l = utils.AccumLoss()
        t_e = utils.AccumLoss()
        t_3d = utils.AccumLoss()

        self.model.eval()
        st = time.time()
        bar = Bar('>>>', fill='>', max=len(train_loader))
        for i, (inputs, targets, all_seq) in enumerate(train_loader):
            bt = time.time()

            if self.is_cuda:
                inputs = Variable(inputs.cuda()).float()
                # targets = Variable(targets.cuda(async=True)).float()
                all_seq = Variable(all_seq.cuda(non_blocking=True)).float()

            outputs, reconstructions, log_var, z = self.model(inputs.float())
            n = outputs.shape[0]
            outputs = outputs.view(n, -1)
            # targets = targets.view(n, -1)

            # loss = loss_funcs.sen_loss(outputs, all_seq, dim_used)

            n, _, _ = all_seq.data.shape
            m_err = loss_funcs.mpjpe_error(outputs, all_seq, input_n, dim_used, dct_n)
            e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used, dct_n)

            # t_l.update(loss.cpu().data.numpy()[0] * n, n)
            t_e.update(e_err.cpu().data.numpy() * n, n)
            t_3d.update(m_err.cpu().data.numpy() * n, n)

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                             time.time() - st)
            bar.next()
        bar.finish()
        return t_e.avg, t_3d.avg

    def test(self, train_loader, dataset='h3.6m', input_n=20, output_n=50, dct_n=20, cartesian=False,
             dim_used=[]):
        N = 0
        # t_l = 0
        if output_n >= 25:
            eval_frame = [1, 3, 7, 9, 13, 24]
        elif output_n == 10:
            eval_frame = [1, 3, 7, 9]

        t_e = np.zeros(len(eval_frame))
        t_3d = np.zeros(len(eval_frame))

        self.model.eval()
        st = time.time()
        bar = Bar('>>>', fill='>', max=len(train_loader))
        for i, (inputs, targets, all_seq) in enumerate(train_loader):
            bt = time.time()

            if self.is_cuda:
                inputs = Variable(inputs.cuda()).float()
                all_seq = Variable(all_seq.cuda(non_blocking=True)).float()

            outputs, reconstructions, log_var, z = self.model(inputs.float())
            n = outputs.shape[0]

            n, seq_len, dim_full_len = all_seq.data.shape
            dim_used_len = len(dim_used)
            all_seq[:, :, 0:6] = 0

            # inverse dct transformation
            _, idct_m = data_utils.get_dct_matrix(seq_len)
            idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
            outputs_t = outputs.view(-1, dct_n).transpose(0, 1)

            if cartesian == False:
                outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1,
                                                                                                           dim_used_len,
                                                                                                           seq_len).transpose(
                    1, 2)
                pred_expmap = all_seq.clone()
                dim_used = np.array(dim_used)
                pred_expmap[:, :, dim_used] = outputs_exp
                pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len)
                targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)

                pred_expmap[:, 0:6] = 0
                targ_expmap[:, 0:6] = 0
                pred_expmap = pred_expmap.view(-1, 3)
                targ_expmap = targ_expmap.view(-1, 3)

                # get euler angles from expmap
                pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap))
                pred_eul = pred_eul.view(-1, dim_full_len).view(-1, output_n, dim_full_len)
                targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap))
                targ_eul = targ_eul.view(-1, dim_full_len).view(-1, output_n, dim_full_len)
                if dataset == 'h3.6m':
                    # get 3d coordinates
                    targ_p3d = data_utils.expmap2xyz_torch(targ_expmap.view(-1, dim_full_len)).view(n, output_n, -1, 3)
                    pred_p3d = data_utils.expmap2xyz_torch(pred_expmap.view(-1, dim_full_len)).view(n, output_n, -1, 3)
                elif dataset == 'cmu_mocap':
                    # get 3d coordinates
                    targ_p3d = data_utils.expmap2xyz_torch_cmu(targ_expmap.view(-1, dim_full_len)).view(n, output_n, -1,
                                                                                                        3)
                    pred_p3d = data_utils.expmap2xyz_torch_cmu(pred_expmap.view(-1, dim_full_len)).view(n, output_n, -1,
                                                                                                        3)
                for k in np.arange(0, len(eval_frame)):
                    j = eval_frame[k]
                    t_e[k] += torch.mean(torch.norm(pred_eul[:, j, :] - targ_eul[:, j, :], 2, 1)).cpu().data.numpy() * n
                    t_3d[k] += torch.mean(torch.norm(
                        targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                        2, 1)).cpu().data.numpy() * n

            elif cartesian:
                outputs_3d = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1,
                                                                                                          dim_used_len,
                                                                                                          seq_len).transpose(
                    1, 2)
                pred_3d = all_seq.clone()
                dim_used = np.array(dim_used)

                # deal with joints at same position
                joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
                index_to_ignore = np.concatenate(
                    (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
                joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
                index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

                pred_3d[:, :, dim_used] = outputs_3d
                pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
                pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
                targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
                for k in np.arange(0, len(eval_frame)):
                    j = eval_frame[k]
                    t_e[k] += torch.mean(torch.norm(
                        targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                        2, 1)).cpu().data.numpy()[0] * n
                    t_3d[k] += torch.mean(torch.norm(
                        targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                        2, 1)).cpu().data.numpy()[0] * n

            N += n

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                             time.time() - st)
            bar.next()
        bar.finish()
        return t_e / N, t_3d / N


