from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset

import utils
import settings


class ProcessedRNNInputDataset(Dataset):

    def __init__(self, entity_type, role):
        data_dir = join(settings.OUT_DIR, entity_type, "rnn")
        fname = "{}_rnn_{}.pkl".format(entity_type, role)
        data_dict = utils.load_large_obj(data_dir, fname)
        self.x1_seq1 = np.array(data_dict["x1_seq1"], dtype=int)
        self.x1_seq2 = np.array(data_dict["x1_seq2"], dtype=int)
        self.x2_seq1 = np.array(data_dict["x2_seq1"], dtype=int)
        self.x2_seq2 = np.array(data_dict["x2_seq2"], dtype=int)
        self.y = np.array(data_dict["y"], dtype=int)

        self.N = len(self.y)

        self.x1_seq1 = torch.from_numpy(self.x1_seq1)
        self.x1_seq2 = torch.from_numpy(self.x1_seq2)
        self.x2_seq1 = torch.from_numpy(self.x2_seq1)
        self.x2_seq2 = torch.from_numpy(self.x2_seq2)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x1_seq1[idx], self.x2_seq1[idx], self.x1_seq2[idx], self.x2_seq2[idx], self.y[idx]


class ProcessedCNNInputDataset(Dataset):

    def __init__(self, entity_type, role):
        data_dir = join(settings.OUT_DIR, entity_type, "cnn")
        fname = "{}_cnn_{}.pkl".format(entity_type, role)
        data_dict = utils.load_large_obj(data_dir, fname)
        self.x1 = np.array(data_dict["x1"], dtype="float32")
        self.x2 = np.array(data_dict["x2"], dtype="float32")
        self.y = np.array(data_dict["y"], dtype=int)

        self.N = len(self.y)

        self.x1 = torch.from_numpy(self.x1)
        self.x2 = torch.from_numpy(self.x2)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]
