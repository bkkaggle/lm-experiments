import random
import pickle

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, preprocessed_path):
        with open(preprocessed_path, "rb") as handle:
            self.batches = pickle.load(handle)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index])

class MultiDataset(Dataset):
    def __init__(self, dataset_1_path, dataset_2_path):
        dataset_1_batches = []
        dataset_2_batches = []

        with open(dataset_1_path, "rb") as handle:
            dataset_1_batches = pickle.load(handle)

        with open(dataset_2_path, "rb") as handle:
            dataset_2_batches = pickle.load(handle)

        dataset_2_batches = np.repeat(dataset_2_batches, 30)

        print(f'dataset 1 len: {len(dataset_1_batches)}')
        print(f'dataset 2 len: {len(dataset_2_batches)}')

        batches = dataset_1_batches + dataset_2_batches
        random.shuffle(batches)

        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index])
