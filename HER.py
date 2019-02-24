# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:42:57 2019

@author: Or
"""
from collections import deque
import numpy as np
import copy


class HER:
    def __init__(self,N):
        self.buffer = deque()
        self.N = N
        
    def reset(self):
        self.buffer = deque()
        
    def keep(self,item):
        self.buffer.append(item)
        
    def backward(self):

        new_buffer = copy.deepcopy(self.buffer)
        num = len(new_buffer)
        goal = self.buffer[-1][-2][0:self.N]
        for i in range(num):
            new_buffer[-1-i][2] = -1.0
            new_buffer[-1-i][-2][self.N:] = goal
            new_buffer[-1-i][0][self.N:] = goal
            new_buffer[-1-i][4] = False
            if (np.sum(np.abs((new_buffer[-1-i][-2][self.N:] - goal))) == 0):
                new_buffer[-1-i][2] = 0.0
                new_buffer[-1-i][4] = True
        return new_buffer
        
        