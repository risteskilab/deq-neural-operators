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

class BasicBlock(nn.Module):
  # Note: parametrizing with a single layer-- might have to use more layers if this doesn't work well
  def __init__(self, modes1, modes2, width, add_mlp=False, normalize=False, activation=F.gelu):
    super(BasicBlock, self).__init__()

    self.modes1 = modes1
    self.modes2 = modes2
    self.width = width
    self.add_mlp = add_mlp
    self.normalize = normalize
    self.act = activation

    self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
    self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
    self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
    
    if self.add_mlp:
      self.mlp0 = MLP2d(self.width, self.width, self.width)
      self.mlp1 = MLP2d(self.width, self.width, self.width)
      self.mlp2 = MLP2d(self.width, self.width, self.width)
    
    if normalize:
      self.norm = nn.InstanceNorm2d(self.width)

    self.w0 = nn.Conv2d(self.width, self.width, 1)
    self.w1 = nn.Conv2d(self.width, self.width, 1)
    self.w2 = nn.Conv2d(self.width, self.width, 1)

  def forward(self, x):
    x = self.norm(x) if self.normalize else x

    x1 = self.conv0(x)
    x1 = self.norm(x1) if self.normalize else x1

    if self.add_mlp:
      x1 = self.mlp0(x1)
    x1 = self.norm(x1) if self.normalize else x1

    x2 = self.w0(x)
    x2 = self.norm(x2) if self.normalize else x2

    x = x1 + x2 
    x = self.act(x)

    x1 = self.conv1(x)
    x1 = self.norm(x1) if self.normalize else x1

    if self.add_mlp:
      x1 = self.mlp1(x1)
    x1 = self.norm(x1) if self.normalize else x1
    
    x2 = self.w1(x)
    x2 = self.norm(x2) if self.normalize else x2

    x = x1 + x2 
    x = self.act(x)

    x1 = self.conv2(x)
    x1 = self.norm(x1) if self.normalize else x1

    if self.add_mlp:
      x1 = self.mlp2(x1)
    x1 = self.norm(x1) if self.normalize else x1
    
    x2 = self.w2(x)
    x2 = self.norm(x2) if self.normalize else x2

    x = x1 + x2 
    x = self.act(x)
    return x

class StackedBasicBlock(nn.Module):
  def __init__(self, modes1, modes2, width, depth=1, add_mlp=False):
    super(StackedBasicBlock, self).__init__()

    self.modes1 = modes1
    self.modes2 = modes2
    self.width = width

    self.depth = depth
    
    blocks = []
    for _ in range(depth):
      blocks.append(BasicBlock(self.modes1, self.modes2, self.width, add_mlp=add_mlp))
    
    self.deq_block = nn.ModuleList(blocks)

  def forward(self, x):
    for idx in range(self.depth):
      x = self.deq_block[idx](x)
    return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, 
                 width, 
                 block_depth,
                 add_mlp=False,
                 normalize=False,
                 in_channels=1,
                 out_channels=1,
                 padding=9,):

        super(FNO2d, self).__init__()

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
        self.add_mlp = add_mlp
        self.modes1 = modes1
        self.modes2 = modes2 
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding # pad the domain if input is non-periodic
        self.fc = nn.Linear(self.in_channels + 2, self.width) # input channel is 2: (a(x), x)
        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.normalize = normalize

        if normalize:
          self.norm = nn.InstanceNorm2d(self.width)
        
        if self.add_mlp:
          self.mlp = MLP2d(self.width, self.width, self.width)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)

        self.deq_block = StackedBasicBlock(self.modes1, self.modes2, self.width, depth=block_depth, add_mlp=add_mlp)

        self.q = MLP2d(self.width, self.out_channels, self.width * 4)

    def forward(self, x, grid=None, **kwargs):
        if grid is None:
          grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        next_z1= self.deq_block(x)

        if self.normalize:
          next_z1 = self.norm(next_z1)

        x1 = self.conv(next_z1)
        x1 = self.norm(x1) if self.normalize else x1

        if self.add_mlp:
          x1 = self.mlp(x1)
        x1 = self.norm(x1) if self.normalize else x1

        x2 = self.w0(next_z1)
        x2 = self.norm(x2) if self.normalize else x2
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
