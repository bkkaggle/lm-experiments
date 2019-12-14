import random
import pickle

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, "rb") as handle:
            self.batches = pickle.load(handle)

        print(f'Dataset length: {len(self.batches)}')

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index])
