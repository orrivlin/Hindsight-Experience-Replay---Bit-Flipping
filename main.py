# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:23:01 2019

@author: Or
"""

import numpy as np
import matplotlib.pyplot as plt
from smooth_signal import smooth
from BFP import BFP
from dqn_HER import DQN_HER


N = 10
env = BFP(N)
gamma = 0.99
buffer_size = int(1e6)
cuda_flag = False
alg = DQN_HER(env, gamma, buffer_size, cuda_flag)
epochs = 5000
start_obs, done = env.reset()
for i in range(epochs):
    log = alg.run_epoch(start_obs)
    print('Done: {} of {}. loss: {}. return: {}'.format(i,epochs,np.round(log.get_current('avg_loss'),2),np.round(log.get_current('tot_return'),2)))
    if i == 2000:
        for param_group in alg.optimizer.param_groups:
            param_group['lr'] = 0.0001
    if i == 4500:
        for param_group in alg.optimizer.param_groups:
            param_group['lr'] = 0.00005

Y = np.asarray(log.get_log('tot_return'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('episode return')

Y = np.asarray(log.get_log('avg_loss'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('average loss')

Y = np.asarray(log.get_log('final_dist'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('minimum distance')

Y = np.asarray(log.get_log('final_dist'))
Y[Y > 1] = 1.0
K = 100
Z = Y.reshape(int(epochs/K),K)
T = 1 - np.mean(Z,axis=1)
x = np.linspace(0, len(T), len(T))*K
fig4 = plt.figure()
ax4 = plt.axes()
ax4.plot(x, T)
plt.xlabel('episodes')
plt.ylabel('sucess rate')
