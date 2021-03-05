from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd


class ClassifierDataset(Dataset):
    def __init__(self, path="debug_inputs/all_train.csv"):
        print("Initialising data from '{}'...".format(path))
        self.class2idx, self.indx2class, self.num_classes = self._initialise_class_and_index_map()

        df = pd.read_csv(path, header=None)

        X = df.iloc[:, 0:-1]
        y = df.iloc[:, -1]

        y.replace(self.class2idx, inplace=True)

        self.X = torch.from_numpy(np.array(X)).float()
        self.y = torch.from_numpy(np.array(y)).long()

        self.m, self.n = X.shape

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    def _initialise_class_and_index_map(self):
        class2idx = {
            'walking': 0,
            'eating': 1,
            'smoking': 2,
            'directions': 3,
            'greeting': 4,
            'posing': 5,
            'purchases': 6,
            'sitting': 7,
            'sittingdown': 8,
            'takingphoto': 9,
            'waiting': 10,
            'walkingdog': 11,
            'walkingtogether': 12
        }
        idx2class = {v: k for k, v in class2idx.items()}
        num_classes = len(class2idx)
        return class2idx, idx2class, num_classes