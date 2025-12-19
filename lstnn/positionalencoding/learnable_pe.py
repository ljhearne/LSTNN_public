import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, emb_dim, learnable=True, init=1.0):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(max_len, emb_dim)*init)
        if learnable is False:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        positional_encoding = self.positional_encoding[:seq_len, :]
        return x + positional_encoding

class LearnablePositionalEncodingUniform(nn.Module):
    def __init__(self, max_len, emb_dim, learnable=True, randinit=True):
        super(LearnablePositionalEncodingUniform, self).__init__()
        if randinit:
            self.positional_encoding = nn.Parameter(torch.rand(max_len, emb_dim))
        else:
            self.positional_encoding = nn.Parameter(torch.rand(max_len, emb_dim)/10)
        if learnable is False:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        positional_encoding = self.positional_encoding[:seq_len, :]
        return x + positional_encoding
