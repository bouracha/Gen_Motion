from torch.utils.data import DataLoader

from utils.h36motion3d import H36motion3D, H36motion3D_pose

import utils.data_utils as data_utils

class DATA():
    def __init__(self, dataset, data_dir):
        self.dataset = dataset
        self.data_dir = data_dir

    def get_dct_and_sequences(self, input_n, output_n, sample_rate, dct_n, out_of_distribution_action=None):
        if out_of_distribution_action != None:
            self.out_of_distribution = True
            acts_train = data_utils.define_actions(out_of_distribution_action, self.dataset, out_of_distribution=False)
            self.acts_OoD = data_utils.define_actions(out_of_distribution_action, self.dataset, out_of_distribution=True)
            self.acts_test = data_utils.define_actions('all', self.dataset, out_of_distribution=False)
        else:
            self.out_of_distribution = False
            acts_train = data_utils.define_actions('all', self.dataset, out_of_distribution=False)
            self.acts_OoD = None
            self.acts_test = data_utils.define_actions('all', self.dataset, out_of_distribution=False)

        if self.dataset == 'h3.6m':
            self.cartesian = False
            self.node_n=48
            self.train_dataset = H36motion(path_to_data=self.data_dir, actions=acts_train, input_n=input_n, output_n=output_n,
                                          split=0, sample_rate=sample_rate, dct_n=dct_n)
            self.data_std = self.train_dataset.data_std
            self.data_mean = self.train_dataset.data_mean
            self.dim_used = self.train_dataset.dim_used
            self.val_dataset = H36motion(path_to_data=self.data_dir, actions=acts_train, input_n=input_n, output_n=output_n,
                                        split=2, sample_rate=sample_rate, data_mean=self.data_mean, data_std=self.data_std, dct_n=dct_n)
            if self.out_of_distribution:
                  self.OoD_val_dataset = H36motion(path_to_data=self.data_dir, actions=self.acts_OoD, input_n=input_n, output_n=output_n,
                                          split=2, sample_rate=sample_rate, data_mean=self.data_mean, data_std=self.data_std, dct_n=dct_n)
            else:
                  self.OoD_val_dataset = None
            self.test_dataset = dict()
            for act in self.acts_test:
                self.test_dataset[act] = H36motion(path_to_data=self.data_dir, actions=act, input_n=input_n, output_n=output_n,
                                         split=1, sample_rate=sample_rate, data_mean=self.data_mean, data_std=self.data_std, dct_n=dct_n)
        elif self.dataset == 'h3.6m_3d':
            self.cartesian = True
            self.node_n=96
            self.train_dataset = H36motion3D(path_to_data=self.data_dir, actions=acts_train, input_n=input_n, output_n=output_n,
                                          split=0, sample_rate=sample_rate, dct_used=dct_n)
            self.val_dataset = H36motion3D(path_to_data=self.data_dir, actions=acts_train, input_n=input_n, output_n=output_n,
                                        split=2, sample_rate=sample_rate, dct_used=dct_n)
            if self.out_of_distribution:
                  self.OoD_val_dataset = H36motion3D(path_to_data=self.data_dir, actions=self.acts_OoD, input_n=input_n, output_n=output_n,
                                          split=2, sample_rate=sample_rate, dct_used=dct_n)
            else:
                  self.OoD_val_dataset = None
            self.test_dataset = dict()
            for act in self.acts_test:
                self.test_dataset[act] = H36motion3D(path_to_data=self.data_dir, actions=act, input_n=input_n, output_n=output_n,
                                         split=1, sample_rate=sample_rate, dct_used=dct_n)
        else:
            raise Exception("Dataset name ({}) is not valid!".format(self.dataset))
        return self.out_of_distribution

    def get_dataloaders(self, train_batch, test_batch, job=0, val_categorise=False):
        # load dadasets for training
        if val_categorise:
            train_loader = dict()
            for act in self.acts_train:
                train_loader[act] = DataLoader(
                    dataset=self.train_dataset[act],
                    batch_size=train_batch,
                    shuffle=True,
                    num_workers=job,
                    pin_memory=True)
        else:
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=train_batch,
                shuffle=True,
                num_workers=job,
                pin_memory=True)
        # We only use validation for the h3.6M dataset
        if self.dataset == 'h3.6m' or 'h3.6m_3d':
            if val_categorise:
                val_loader = dict()
                for act in self.acts_train:
                    val_loader[act] = DataLoader(
                        dataset=self.val_dataset[act],
                        batch_size=test_batch,
                        shuffle=True,
                        num_workers=job,
                        pin_memory=True)
            else:
                val_loader = DataLoader(
                    dataset=self.val_dataset,
                    batch_size=test_batch,
                    shuffle=True,
                    num_workers=job,
                    pin_memory=True)
            if self.out_of_distribution:
                OoD_val_loader = DataLoader(
                    dataset=self.OoD_val_dataset,
                    batch_size=test_batch,
                    shuffle=True,
                    num_workers=job,
                    pin_memory=True)
            else:
                OoD_val_loader = None
        else:
            val_loader = None
            OoD_val_loader = None
        test_loaders = dict()
        for act in self.acts_test:
            test_loaders[act] = DataLoader(
                dataset=self.test_dataset[act],
                batch_size=test_batch,
                shuffle=False,
                num_workers=job,
                pin_memory=True)
        print(">>> train data {}".format(self.train_dataset.__len__()))
        print(">>> validation data {}".format(self.val_dataset.__len__()))
        return train_loader, val_loader, OoD_val_loader, test_loaders

    def get_poses(self, input_n, output_n, sample_rate, dct_n, out_of_distribution_action=None, val_categorise=False):
        if out_of_distribution_action != None:
            self.out_of_distribution = True
            self.acts_train = data_utils.define_actions(out_of_distribution_action, self.dataset, out_of_distribution=False)
            self.acts_OoD = data_utils.define_actions(out_of_distribution_action, self.dataset, out_of_distribution=True)
            self.acts_test = data_utils.define_actions('all', self.dataset, out_of_distribution=False)
        else:
            self.out_of_distribution = False
            self.acts_train = data_utils.define_actions('all', self.dataset, out_of_distribution=False)
            self.acts_OoD = None
            self.acts_test = data_utils.define_actions('all', self.dataset, out_of_distribution=False)

        if self.dataset == 'h3.6m':
            self.cartesian = False
            self.node_n=48
            self.train_dataset = H36motion_pose(path_to_data=self.data_dir, actions=self.acts_train, input_n=input_n, output_n=output_n,
                                          split=0, sample_rate=sample_rate, dct_n=dct_n)
            self.data_std = self.train_dataset.data_std
            self.data_mean = self.train_dataset.data_mean
            self.dim_used = self.train_dataset.dim_used
            self.val_dataset = H36motion_pose(path_to_data=self.data_dir, actions=self.acts_train, input_n=input_n, output_n=output_n,
                                        split=2, sample_rate=sample_rate, data_mean=self.data_mean, data_std=self.data_std, dct_n=dct_n)
            if self.out_of_distribution:
                  self.OoD_val_dataset = H36motion_pose(path_to_data=self.data_dir, actions=self.acts_OoD, input_n=input_n, output_n=output_n,
                                          split=2, sample_rate=sample_rate, data_mean=self.data_mean, data_std=self.data_std, dct_n=dct_n)
            else:
                  self.OoD_val_dataset = None
            self.test_dataset = dict()
            for act in self.acts_test:
                self.test_dataset[act] = H36motion_pose(path_to_data=self.data_dir, actions=act, input_n=input_n, output_n=output_n,
                                         split=1, sample_rate=sample_rate, data_mean=self.data_mean, data_std=self.data_std, dct_n=dct_n)
        elif self.dataset == 'h3.6m_3d':
            self.cartesian = False
            self.node_n=96
            if val_categorise:
                self.train_dataset = dict()
                self.val_dataset = dict()
                for act in self.acts_train:
                    self.train_dataset[act] = H36motion3D_pose(path_to_data=self.data_dir, actions=act, input_n=input_n, output_n=output_n, split=0, sample_rate=sample_rate, dct_used=dct_n)
                    self.val_dataset[act] = H36motion3D_pose(path_to_data=self.data_dir, actions=act, input_n=input_n, output_n=output_n, split=2, sample_rate=sample_rate, dct_used=dct_n)
            else:
                self.train_dataset = H36motion3D_pose(path_to_data=self.data_dir, actions=self.acts_train, input_n=input_n, output_n=output_n, split=0, sample_rate=sample_rate, dct_used=dct_n)
                self.val_dataset = H36motion3D_pose(path_to_data=self.data_dir, actions=self.acts_train, input_n=input_n, output_n=output_n, split=2, sample_rate=sample_rate, dct_used=dct_n)
            if self.out_of_distribution:
                  self.OoD_val_dataset = H36motion3D_pose(path_to_data=self.data_dir, actions=self.acts_OoD, input_n=input_n, output_n=output_n,
                                          split=2, sample_rate=sample_rate, dct_used=dct_n)
            else:
                  self.OoD_val_dataset = None
            self.test_dataset = dict()
            for act in self.acts_test:
                self.test_dataset[act] = H36motion3D_pose(path_to_data=self.data_dir, actions=act, input_n=input_n, output_n=output_n,
                                         split=1, sample_rate=sample_rate, dct_used=dct_n)
        else:
            raise Exception("Dataset name ({}) is not valid!".format(dataset))
        return self.out_of_distribution