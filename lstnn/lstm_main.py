'''

see https://github.com/charlesakin/sudoku/blob/master/3LayerLSTMwithConfusionMatrix.ipynb
for inspiration behind LSTM models

for weight init
https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
import math
'''

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


# class LSTM_combined(nn.Module):
#     def __init__(self, input_size=6, hidden_size=160, num_layers=1,
#                  output_size=4, bidirectional=True, device='cpu'):
#         super(LSTM_combined, self).__init__()
#         self.bidirectional = bidirectional
#         self.device = device
#         self.num_layers = num_layers
#         self.lstm_row = nn.LSTM(
#             input_size, hidden_size, num_layers,
#             batch_first=True, bidirectional=bidirectional)
#         self.lstm_col = nn.LSTM(
#             input_size, hidden_size, num_layers,
#             batch_first=True, bidirectional=bidirectional)

#         if self.bidirectional:
#             self.hidden_size = hidden_size*2
#         else:
#             self.hidden_size = hidden_size

#         self.final_layer = torch.nn.Linear(self.hidden_size, output_size)

#     def forward(self, x):
#         self.batch_size = x.size(0)

#         x_row = x.flatten(start_dim=1, end_dim=2)

#         # pad inputs
#         x_row = torch.nn.functional.pad(x_row, (0, 1, 0, 1), 'constant')
#         x_row[:, -1, -1] = 1

#         h_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size,
#                        int(self.hidden_size/2))).to(self.device)
#         c_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size,
#                        int(self.hidden_size/2))).to(self.device)
#         lstm_row_out, hidden_row = self.lstm_row(x_row, (h_0, c_0))

#         # run "column" LSTM
#         x_col = x.transpose(1, 2)
#         x_col = x_col.flatten(start_dim=1, end_dim=2)

#         # pad inputs
#         x_col = torch.nn.functional.pad(x_col, (0, 1, 0, 1), 'constant')
#         x_col[:, -1, -1] = 1

#         h_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size,
#                        int(self.hidden_size/2))).to(self.device)
#         c_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size,
#                        int(self.hidden_size/2))).to(self.device)
#         lstm_col_out, hidden_col = self.lstm_col(x_col, (h_0, c_0))

#         # combine the two
#         combined = lstm_row_out + lstm_col_out
#         out = self.final_layer(combined[:, -1, :])
#         #out = self.final_layer(lstm_row_out[:, -1, :])
#         return out


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

        #self.final_layer = torch.nn.Linear(self.hidden_size, output_size)
        self.final_layer = nn.Sequential(nn.Linear(hidden_size, output_size),
                                         torch.nn.LogSoftmax(dim=-1))

    def forward(self, x):
        self.batch_size = x.size(0)
        x_row = x.flatten(start_dim=1, end_dim=2)

        if self.bidirectional:
            h_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size,
                                       int(self.hidden_size/2))).to(self.device)
            c_0 = Variable(torch.zeros(self.num_layers*2, self.batch_size,
                                       int(self.hidden_size/2))).to(self.device)
        else:
            h_0 = Variable(torch.zeros(self.num_layers, self.batch_size,
                                       self.hidden_size)).to(self.device)
            c_0 = Variable(torch.zeros(self.num_layers, self.batch_size,
                                       self.hidden_size)).to(self.device)

        lstm_out, hidden = self.lstm_row(x_row, (h_0, c_0))

        # combine the two
        out = self.final_layer(lstm_out[:, -1, :])
        return out
