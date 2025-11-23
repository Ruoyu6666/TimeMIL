from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F



class StartConv(nn.Module):
    def __init__(self, d_in, d_out, hidden=128):
        super().__init__()

        # Conv1D scans over time only
        self.conv = nn.Conv1d(
            in_channels=d_in,      # feature dims
            out_channels=hidden,   # new feature channels
            kernel_size=3,
            padding=1
        )
        self.d_out = d_out
        self.fc = nn.Linear(hidden, d_out)  # Final projection to output size D

    def forward(self, x):
        B, N, T, D = x.shape                    
        x = x.reshape(B * N, T, D).permute(0, 2, 1) # (B * N, D, T)
        x = torch.relu(self.conv(x))         # (B*N, hidden, T)
        x = x.mean(dim=-1)                   # Global average pooling over time: (B*N, hidden)
        x = self.fc(x)                       # Project to output dimension: (B*N, d_out)

        # reshape back
        return x.reshape(B, N, self.d_out)


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)