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

import wandb 

class BasicBlock(nn.Module):
  # Note: parametrizing with a single layer-- might have to use more layers if this doesn't work well
  def __init__(self, modes1, modes2, width, add_mlp=False,  normalize=False, activation=F.gelu):
    super(BasicBlock, self).__init__()

    self.modes1 = modes1
    self.modes2 = modes2
    self.width = width
    self.add_mlp = add_mlp

    self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
    self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
    self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
    
    self.normalize = normalize    
    self.act = activation

    if add_mlp:
      self.mlp0 = MLP2d(self.width, self.width, self.width)
      self.mlp1 = MLP2d(self.width, self.width, self.width)
      self.mlp2 = MLP2d(self.width, self.width, self.width)

    if normalize:
      self.norm = nn.InstanceNorm2d(self.width)

    self.w0 = nn.Conv2d(self.width, self.width, 1)
    self.w1 = nn.Conv2d(self.width, self.width, 1)
    self.w2 = nn.Conv2d(self.width, self.width, 1)

  def forward(self, x, injection=None):
    x = self.norm(x) if self.normalize else x

    x1 = self.conv0(x)
    x1 = self.norm(x1) if self.normalize else x1

    if self.add_mlp:
      x1 = self.mlp0(x1)
      
    x1 = self.norm(x1) if self.normalize else x1

    x2 = self.w0(x)
    x2 = self.norm(x2) if self.normalize else x2
    x = x1 + x2 + injection
    x = self.act(x)

    x1 = self.conv1(x)
    x1 = self.norm(x1) if self.normalize else x1

    if self.add_mlp:
      x1 = self.mlp1(x1)

    x1 = self.norm(x1) if self.normalize else x1

    x2 = self.w1(x)
    x2 = self.norm(x2) if self.normalize else x2
    x = x1 + x2 + injection
    x = self.act(x)

    x1 = self.conv2(x)
    x1 = self.norm(x1) if self.normalize else x1
    if self.add_mlp:
      x1 = self.mlp2(x1)
    x1 = self.norm(x1) if self.normalize else x1

    x2 = self.w2(x)
    x2 = self.norm(x2) if self.normalize else x2
    x = x1 + x2 + injection
    x = self.act(x)
    return x

class StackedBasicBlock(nn.Module):
  def __init__(self, modes1, modes2, width, depth=1, add_mlp=False, normalize=False):
    super(StackedBasicBlock, self).__init__()

    self.modes1 = modes1
    self.modes2 = modes2
    self.width = width
    self.add_mlp = add_mlp
    self.depth = depth
    
    blocks = []
    for _ in range(depth):
      blocks.append(BasicBlock(self.modes1, self.modes2, self.width, add_mlp=add_mlp, normalize=normalize))
    
    self.deq_block = nn.ModuleList(blocks)

  def forward(self, x, injection=None):
    for idx in range(self.depth):
      x = self.deq_block[idx](x, injection)
    return x

class FNO2dDEQ(nn.Module):
  def __init__(self, modes1, modes2, width, 
                pretrain_steps=300,
                f_thres=32,
                b_thres=32,
                f_solver='anderson', 
                b_solver='anderson',
                stop_mode='abs',
                block_depth=1,
                add_mlp=False,
                use_pg=False,
                tau=0.5,
                pg_steps=1,
                pretrain_iter_steps=8,
                in_channels=2,
                out_channels=1,
                padding=9,
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
        
        self.pretrain_steps = pretrain_steps
        self.deq_block = StackedBasicBlock(self.modes1, self.modes2, self.width, 
                                    depth=block_depth, 
                                    add_mlp=add_mlp, 
                                    normalize=normalize)

        self.q = MLP2d(self.width, self.out_channels, self.width * 4)

        # DEQ solver related parameters/configuration
        self.f_solver = eval(f_solver)
        self.b_solver = eval(b_solver)

        self.f_thres = f_thres
        self.b_thres = b_thres

        self.stop_mode = stop_mode

        self.use_pg = use_pg
        self.tau = tau
        self.pg_steps = pg_steps

        self.pretrain_iter_steps = pretrain_iter_steps

  def forward(self, x, grid=None, train_step=-1, iters=-1, eps=1e-3, f_thres=0, wandb_log=False):
    if grid is None:
      grid = self.get_grid(x.shape, x.device)

    x = torch.cat((x, grid), dim=-1)
    x = self.fc(x)
    x = x.permute(0, 3, 1, 2)
    x = F.pad(x, [0,self.padding, 0,self.padding])

    deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)
    func = lambda z: self.deq_block(z, x)
    
    ### Use this for testing unrolled + backprop networks
    if iters > 0:
      z1 = torch.zeros_like(x).cuda() 
      for _ in range(iters):
        next_x = func(z1)
        abs_diff = (next_x - x).norm().item()
        rel_diff = abs_diff / (1e-5 + x.norm().item())
        # wandb.log({'abs ||f(x,z) - z||': abs_diff, 'relative ||f(x,z) - z||': rel_diff})
        z1 = next_x
      next_z1 = z1
    else:
      ## Regular DEQ mode with pretraining
      z1 = torch.zeros_like(x).cuda()
      if not deq_mode:
        print("Not in DEQ mode!!!")
        fp_iters = self.pretrain_iter_steps
        for _ in range(fp_iters):
          next_z1 = func(z1)
          abs_diff = (next_z1 - z1).norm().item()
          rel_diff = abs_diff / (1e-5 + z1.norm().item())
          z1 = next_z1
        next_z1 = z1
      else:
        with torch.no_grad():
          if f_thres > 0:
            self.f_thres = f_thres
          result = self.f_solver(
            func, z1, threshold=self.f_thres, 
            stop_mode=self.stop_mode, name="forward", eps=eps)

          z1 = result['result']
          # if wandb_log:
          #   wandb.init(
          #       project=f"Steady-State-PDE-DEQ-Darcy-flow-residuals", reinit=True
          #   )
          #   for i in range(len(result['abs_trace'])):
          #       wandb.log({'abs ||f(x,z) - z||': result['abs_trace'][i], 'relative ||f(x,z) - z||': result['rel_trace'][i]})

          if train_step == -1:
            print(f"[Forward] {train_step} total steps: {len(result['abs_trace'])} Steps: {result['nstep']} Abs diff {min(result['abs_trace'])} Rel diff {min(result['rel_trace'])}")
        
        next_z1 = z1

        if self.training:
          if self.use_pg:
            z1.requires_grad_()
            for _ in range(self.pg_steps):
                z1 = (1 - self.tau) * z1 + self.tau * func(z1)
            next_z1 = z1
          else:
            next_z1 = func(z1.requires_grad_())
            def backward_hook(grad):
              if self.hook is not None:
                self.hook.remove()
                torch.cuda.synchronize()
              result = self.b_solver(
                  lambda y: autograd.grad(next_z1, z1, y, retain_graph=True)[0] + grad, 
                  torch.zeros_like(grad),
                  # lam=self.anderson_lam,
                  # m=self.anderson_m,
                  threshold=self.b_thres,
                  stop_mode=self.stop_mode,
                  eps=eps,
                  name="backward")
              # if train_step % 500 == 0:
                # print(f"[Back] {train_step} Steps: {result['nstep']} Abs diff {min(result['abs_trace'])} Rel diff {min(result['rel_trace'])}")
              return result['result']

            self.hook = next_z1.register_hook(backward_hook)
          
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
