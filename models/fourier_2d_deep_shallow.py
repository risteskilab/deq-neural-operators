import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import torch.autograd as autograd

import sys
import os

sys.path.append("lib/")
sys.path.append("../")

import operator
from functools import reduce
from functools import partial
from utils.utilities3 import *
from lib.solvers import anderson, broyden

from models.commons import SpectralConv2d, MLP2d
from models.fourier_2d_deq_shallow import StackedBasicBlock

#### Note: This code works only for Darcy flow
class FNO2dDeepSmall(nn.Module):
    def __init__(self, modes1, modes2, 
                    width, 
                    block_depth,
                    add_mlp=False):

        super(FNO2dDeepSmall, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2 
        self.width = width
        self.add_mlp = add_mlp

        self.padding = 9 # pad the domain if input is non-periodic
        self.fc = nn.Linear(3, self.width) # input channel is 2: (a(x), x)
        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        if self.add_mlp:
            self.mlp = MLP2d(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)

        self.deq_block = StackedBasicBlock(self.modes1, self.modes2, self.width, depth=block_depth, add_mlp=add_mlp)

        self.q = MLP2d(self.width, 1, self.width * 4)

    def forward(self, x, **kwargs):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        ### Use this for testing non-weight tied, unrolled + backprop networks
        z1 = torch.zeros_like(x).cuda() 
        next_z1= self.deq_block(z1, x)

        x1 = self.conv(next_z1)
        if self.add_mlp:
            x1 = self.mlp(x1)
        x2 = self.w0(next_z1)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)