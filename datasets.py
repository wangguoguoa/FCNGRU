import os
import h5py
import os.path as osp
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data

__all__ = ['EPIDataSetTrain', 'EPIDataSetTest']



class EPIDataSetTrain(data.Dataset):
    def __init__(self, data_tr, label_tr, denselabel_tr):
        super(EPIDataSetTrain, self).__init__()
        self.data = data_tr
        self.label = label_tr
        self.denselabel = denselabel_tr

        assert len(self.data) == len(self.label) and len(self.data) == len(self.denselabel), \
            "the number of sequences and labels must be consistent."

        print("The number of data is {}".format(len(self.label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]
        denselabel_one = self.denselabel[index]

        return {"data": data_one, "label": label_one, "denselabel": denselabel_one}


class EPIDataSetTest(data.Dataset):
    def __init__(self, data_te, label_te, denselabel_te):
        super(EPIDataSetTest, self).__init__()
        self.data = data_te
        self.label = label_te
        self.denselabel = denselabel_te

        assert len(self.data) == len(self.label) and len(self.data) == len(self.denselabel), \
            "the number of sequences and labels must be consistent."
        print("The number of data is {}".format(len(self.label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]
        denselabel_one = self.denselabel[index]

        return {"data": data_one, "label": label_one, "denselabel": denselabel_one}


