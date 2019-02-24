# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:20:51 2019

@author: orrivlin
"""

import torch
import torch.nn.functional as F


class Policy(torch.nn.Module):
    def __init__(self,N,K):
        super(Policy, self).__init__()
        self.N = N
        self.K = K
        self.fc1 = torch.nn.Linear(self.N,128)
        self.fc2 = torch.nn.Linear(128,self.K)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
