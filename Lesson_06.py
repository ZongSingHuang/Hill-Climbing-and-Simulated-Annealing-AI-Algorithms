# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:19:48 2021

@author: zongsing.huang
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# HC/SA 是屬於一種概念型的最佳化演算法，因此X沒有特定的更新方式
# HC/SA 不屬於群體智能算法，因此P固定為1
# 講師採用機器人爬山作為例子，機器人每個次代都會嘗試四種方向前進，最後取適應值高的作為本次代的更新
# 本題為最大化問題求解，並且有四種測試函數供選擇
# =============================================================================

def fitness(X, species=0):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    if species==0:
        o = -10 * np.cos( 2 * (X[:, 0]**2+X[:, 1]**2)**0.5 ) * np.exp( -0.5 * ( (X[:, 0]+1)**2 + (X[:, 1]-1)**2)**0.5 ) + 5.1
    if species==1:
         o = -( 3*(1-X[:, 0])**2*np.exp(-(X[:, 0]**2) - (X[:, 1]+1)**2) - 10*(X[:, 0]/5 - X[:, 0]**3 - X[:, 1]**5)*np.exp(-X[:, 0]**2-X[:, 1]**2) - 1/3*np.exp(-(X[:, 0]+1)**2 - X[:, 1]**2)  ) + 8.5
    if species==2:
        o =-1*( 0.2 + X[:, 0]**2 + X[:, 1]**2 - 0.1*np.cos(6*np.pi*X[:, 0]) - 0.1*np.cos(6*np.pi*X[:, 1]) )
    if species==3:
        o = -1 * np.sum(X**2, axis=1)
    
    return o

def getNeighbours(X, step_size):
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    new_X1 = X + [1, 0]*step_size
    new_X2 = X + [0, 1]*step_size
    new_X3 = X + [-1, 0]*step_size
    new_X4 = X + [0, -1]*step_size
    new_X = np.vstack([new_X1, new_X2, new_X3, new_X4])
    
    return new_X

#%% 參數設定
species = 3
P = 1 # 固定
D = 2
G = None
ub = 1*np.ones([P, D])
lb = -1*np.ones([P, D])
k = 0.05

#%% 初始化
X = np.random.uniform(low=lb, high=ub, size=[P, D])
step_size = k*np.ones([P, D])
found_the_optimum = False
loss_curve = []
early_stopping = 0
g = 0

#%% 迭代
F = fitness(X, species)[0]
loss_curve.append(F)

while True:
    # Step1. 找尋周圍的可行解
    new_X = getNeighbours(X, step_size)
    # 邊界處理
    mask1 = new_X>ub
    mask2 = new_X<lb
    new_X[mask1] = ( ub*np.ones([4, D]) )[mask1]
    new_X[mask2] = ( lb*np.ones([4, D]) )[mask1]
    
    # Step2. 適應值計算
    new_F = fitness(new_X, species)
    
    # Step3. 更新X
    for i in range(len(new_F)):
       if new_F[i]>F:
           F = new_F[i]
           X = new_X[i]
           found_the_optimum = True
           early_stopping = 0
    
    # 收斂曲線
    if found_the_optimum==True:
        loss_curve.append(F)
        g = g + 1
        print(g)
    
    # 收斂率<0.1%
    if np.abs(loss_curve[-1]-loss_curve[-2]) / np.abs(loss_curve[-1])<=1e-3:
        early_stopping = early_stopping + 1

    # 如果本次沒有順利更新/收斂太慢，則強制結束
    if found_the_optimum==False or early_stopping>=5:
        loss_curve = np.array(loss_curve)
        break

#%% 作圖
plt.figure()
plt.plot(loss_curve)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Fitness')