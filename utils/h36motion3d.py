from torch.utils.data import Dataset
import numpy as np
from utils import data_utils

class H36motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, dct_used=15, split=0, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_used = dct_used

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        acts = data_utils.define_actions(actions)

        subjs = subs[split]

        labels = []
        all_seqs = np.array([])
        #print(acts)
        for act in acts:
            #print(act)
            action_seq, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, [act], sample_rate, input_n)
            #print(action_seq.shape)
            try:
                all_seqs = np.concatenate((all_seqs, action_seq), axis=0)
            except:
                all_seqs = action_seq
            label = [str(act)] * action_seq.shape[0]
            labels = labels + label
            #print(len(labels))
            #print(all_seqs.shape)
            #print(labels[0], labels[-1])
        #all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate, input_n )

        self.labels = labels
        self.all_seqs = all_seqs
        t_n = input_n

        b_n, f_n = self.all_seqs.shape[0], 96

        self.all_seqs = self.all_seqs.reshape(b_n, f_n, t_n)


        #normalise to unit cube
        max_per_pose = np.amax(self.all_seqs, axis=(1,2))
        max_per_pose = np.repeat(max_per_pose.reshape(b_n, 1, 1), 96, axis=1)
        max_per_pose = np.repeat(max_per_pose.reshape(b_n, 96, 1), t_n, axis=2)
        min_per_pose = np.amin(self.all_seqs, axis=(1,2))
        min_per_pose = np.repeat(min_per_pose.reshape(b_n, 1, 1), 96, axis=1)
        min_per_pose = np.repeat(min_per_pose.reshape(b_n, 96, 1), t_n, axis=2)
        self.all_seqs = (self.all_seqs - min_per_pose)/(max_per_pose - min_per_pose)


    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        return self.all_seqs[item], self.labels[item]

class H36motion3D_original(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_used = dct_used

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        acts = data_utils.define_actions(actions)

        # subs = np.array([[1], [5], [11]])
        # acts = ['walking']

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                 input_n + output_n)
        self.all_seqs = all_seqs
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used]
        all_seqs = all_seqs.transpose(0, 2, 1)
        all_seqs = all_seqs.reshape(-1, input_n + output_n)
        all_seqs = all_seqs.transpose()

        dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
        dct_m_out, _ = data_utils.get_dct_matrix(input_n + output_n)
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_dct_seq = np.matmul(dct_m_in[0:dct_used, :], all_seqs[i_idx, :])
        input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
        # input_dct_seq = input_dct_seq.reshape(-1, len(dim_used) * dct_used)

        output_dct_seq = np.matmul(dct_m_out[0:dct_used, :], all_seqs)
        output_dct_seq = output_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
        # output_dct_seq = output_dct_seq.reshape(-1, len(dim_used) * dct_used)

        self.input_dct_seq = input_dct_seq
        self.output_dct_seq = output_dct_seq

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]


class H36motion3D_pose(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_used = dct_used

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        acts = data_utils.define_actions(actions)

        # subs = np.array([[1], [5], [11]])
        # acts = ['walking']

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate, 20)
        self.all_seqs = all_seqs
        self.dim_used = dim_used

        m = self.all_seqs.shape[0]
        n = self.all_seqs.shape[-1]
        #average pose over 20 contiguous timesteps
        self.all_seqs = np.mean(self.all_seqs, axis=1)
        #normalise to unit cube
        self.all_seqs = self.all_seqs.reshape((m, n))

        max_per_pose = np.amax(self.all_seqs, -1)
        max_per_pose = np.repeat(max_per_pose.reshape(m, 1), 96, axis=1)
        min_per_pose = np.amin(self.all_seqs, -1)
        min_per_pose = np.repeat(min_per_pose.reshape(m, 1), 96, axis=1)
        self.all_seqs = (self.all_seqs - min_per_pose)/(max_per_pose - min_per_pose)

        #print("all seq_shape ", self.all_seqs.shape)
        #self.all_seqs = self.all_seqs[:, dim_used]

    def __len__(self):
        return self.all_seqs.shape[0]

    def __getitem__(self, item):
        return self.all_seqs[item]