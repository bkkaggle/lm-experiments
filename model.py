import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

        self.linear = nn.Linear(256, 256)

    def forward(self, x, labels):
        x = self.linear(x)

        loss = F.cross_entropy(x, torch.argmax(labels, dim=-1).view(-1))

        return (loss, x)

    def save_pretrained(self, dir):
        filename = os.path.join(dir, "checkpoint.pt")
        torch.save(self.state_dict(), filename)