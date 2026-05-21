"""LSTM baseline model for the Latin Squares Task.

A 4-layer stacked LSTM that processes flattened puzzle grids row by row.
Used by run_simple_models.py as a recurrent baseline for comparison.
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=160, num_layers=4,
                 output_size=4, bidirectional=False, device='cpu'):
        super(LSTM, self).__init__()
        self.bidirectional = bidirectional
        self.device = device
        self.num_layers = num_layers
        self.lstm_row = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional)

        if self.bidirectional:
            self.hidden_size = hidden_size*2
        else:
            self.hidden_size = hidden_size

        self.final_layer = nn.Sequential(nn.Linear(self.hidden_size,
                                                   output_size),
                                         torch.nn.LogSoftmax(dim=-1))

    def forward(self, x):
        self.batch_size = x.size(0)
        x_row = x.flatten(start_dim=1, end_dim=2)

        if self.bidirectional:
            h_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size,
                                       int(self.hidden_size/2))
                                       ).to(self.device)
            c_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size,
                                       int(self.hidden_size/2))
                                       ).to(self.device)
        else:
            h_0 = Variable(torch.zeros(self.num_layers, self.batch_size,
                                       self.hidden_size)).to(self.device)
            c_0 = Variable(torch.zeros(self.num_layers, self.batch_size,
                                       self.hidden_size)).to(self.device)

        lstm_out, hidden = self.lstm_row(x_row, (h_0, c_0))

        out = self.final_layer(lstm_out[:, -1, :])
        return out
