import pickle

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
        return torch.tensor(self.batches[index]).float()
