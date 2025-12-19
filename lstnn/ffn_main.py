import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class FFN(nn.Module):
    def __init__(self, hidden_size=160, input_size=80, output_size=4):
        super(FFN, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden_size = hidden_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            torch.nn.LogSoftmax(dim=-1))

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out
