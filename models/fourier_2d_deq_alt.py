from random import sample
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# import matplotlib.pyplot as plt
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
from models.fourier_2d_deq import StackedBasicBlock

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm

import wandb 

class FNO2dDEQ(nn.Module):
  def __init__(self, modes1, modes2, width, args, 
                block_depth=1,
                in_channels=2,
                out_channels=1,
                padding=9,
                add_mlp=False,
                normalize=False):

        super(FNO2dDEQ, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.add_mlp = add_mlp

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding # pad the domain if input is non-periodic

        self.fc = nn.Linear(self.in_channels + 2, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.normalize = normalize
        if normalize:
          self.norm = nn.InstanceNorm2d(self.width)

        if self.add_mlp:
          self.mlp = MLP2d(self.width, self.width, self.width)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        
        self.deq_block = StackedBasicBlock(self.modes1, self.modes2, self.width, 
                                    depth=block_depth, 
                                    add_mlp=add_mlp, 
                                    normalize=normalize)

        self.q = MLP2d(self.width, self.out_channels, self.width * 4)

        self.deq = get_deq(args.deq_args)
        apply_norm(self.deq_block, args=args)



  def forward(self, x, grid=None, train_step=-1, iters=-1, eps=1e-3, f_thres=0, wandb_log=False):
    if grid is None:
      grid = self.get_grid(x.shape, x.device)

    x = torch.cat((x, grid), dim=-1)
    x = self.fc(x)
    x = x.permute(0, 3, 1, 2)
    x = F.pad(x, [0,self.padding, 0,self.padding])

    reset_norm(self.deq_block)

    func = lambda z: self.deq_block(z, x)

    z1 = torch.zeros_like(x).cuda()

    output, info = self.deq(func, z1)
    next_z1 = output[-1]
          
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
