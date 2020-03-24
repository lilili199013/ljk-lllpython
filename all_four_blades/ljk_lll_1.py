# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:31:26 2020

@author: lilili
"""

#!user/bin/env python
# _*_ coding: utf-8 _*_
 
# @Version :   1.0
# @Author  :   liujunkong
# @Email   :   1196689756@qq.com
# @Time    :   2020/03/20 09:22:57
#Description:
 
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import math,time


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/9/11
# @Author  : github.com/guofei9987

import numpy as np
import random


class ACA_TSP:
    def __init__(self, func, n_dim,
                 size_pop=10, max_iter=20,
                 distance_matrix=None,
                 alpha=1, beta=2, rho=0.1,
                 ):
        self.func = func
        self.n_dim = n_dim  # 城市数量
        self.size_pop = size_pop  # 蚂蚁数量
        self.max_iter = max_iter  # 迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 适应度的重要程度
        self.rho = rho  # 信息素挥发速度

        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))  # 避免除零错误

        self.Tau = np.ones((n_dim, n_dim))  # 信息素矩阵，每次迭代都会更新
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int)  # 某一代每个蚂蚁的爬行路径
        self.y = None  # 某一代每个蚂蚁的爬行总距离
        self.x_best_history, self.y_best_history = [], []  # 记录各代的最佳情况
        self.best_x, self.best_y = None, None

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):  # 对每次迭代
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta  # 转移概率，无须归一化。
            for j in range(self.size_pop):  # 对每个蚂蚁
  ##########################################################################################
                self.Table[j, 0] = random.randint(0,self.n_dim-1)  # start point，其实可以随机，但没什么区别
                for k in range(self.n_dim - 1):  # 蚂蚁到达的每个节点
                    taboo_set = set(self.Table[j, :k + 1])  # 已经经过的点和当前点，不能再次经过
                    allow_list = list(set(range(self.n_dim)) - taboo_set)  # 在这些点中做选择
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()  # 概率归一化
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            # 计算距离
            y = np.array([self.func(i) for i in self.Table])

            # 顺便记录历史最好情况
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.x_best_history.append(x_best)
            self.y_best_history.append(y_best)

            # 计算需要新涂抹的信息素
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):  # 每个蚂蚁
                for k in range(self.n_dim - 1):  # 每个节点
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # 蚂蚁从n1节点爬到n2节点
                    delta_tau[n1, n2] += 1 / y[j]  # 涂抹的信息素
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]  # 蚂蚁从最后一个节点爬回到第一个节点
                delta_tau[n1, n2] += 1 / y[j]  # 涂抹信息素

            # 信息素飘散+信息素涂抹
            self.Tau = (1 - self.rho) * self.Tau + delta_tau

        best_generation = np.array(self.y_best_history).argmin()
        self.best_x = self.x_best_history[best_generation]
        self.best_y = self.y_best_history[best_generation]
        return self.best_x, self.best_y

    fit = run




##################################################
L_1=[61, 31, 4, 101, 22, 43, 98, 76, 102, 34, 54, 14, 79, 19, 110, 96, 
    114, 63, 90, 62, 41, 105, 46, 77, 40, 39, 69, 18, 47, 88, 71, 16, 36, 
    80, 44, 91, 100, 78, 87, 94, 70, 67, 116, 50, 1, 103, 12, 26, 111, 5, 
    65, 68, 38, 25, 86, 112, 66, 27, 56, 13, 117, 104, 60, 55, 23, 99, 35, 
    51, 17, 57, 93, 113, 84, 3, 85, 107, 81, 89, 82, 6, 97, 83, 28, 53, 9, 
    49, 64, 30, 74, 11, 29, 8, 59, 32, 73, 42, 21, 109, 10, 45, 108, 2, 95, 
    7, 75, 52, 37, 33, 106, 48, 72, 24, 92, 20, 15, 115, 58, 118, 119, 120, 
    121, 122, 123, 124, 125, 126]
L_2=[3.504, 3.489, 3.501, 3.498, 3.499, 3.5, 3.496, 3.496, 3.5, 3.503, 3.502, 
    3.501, 3.496, 3.502, 3.502, 3.486, 3.496, 3.499, 3.499, 3.499, 3.491, 3.488, 
    3.501, 3.495, 3.5, 3.5, 3.486, 3.49, 3.484, 3.487, 3.49, 3.492, 3.491, 3.499, 
    3.491, 3.488, 3.494, 3.493, 3.493, 3.486, 3.495, 3.489, 3.485, 3.481, 3.483, 
    3.49, 3.487, 3.49, 3.5, 3.482, 3.49, 3.48, 3.485, 3.48, 3.487, 3.49, 3.487, 
    3.487, 3.475, 3.477, 3.484, 3.493, 3.479, 3.494, 3.482, 3.479, 3.492, 3.483, 
    3.487, 3.49, 3.491, 3.479, 3.482, 3.485, 3.487, 3.475, 3.481, 3.486, 3.481, 
    3.481, 3.486, 3.485, 3.487, 3.492, 3.475, 3.487, 3.487, 3.496, 3.487, 3.488, 
    3.484, 3.489, 3.501, 3.497, 3.49, 3.494, 3.495, 3.491, 3.498, 3.502, 3.5, 3.496, 
    3.499, 3.504, 3.495, 3.499, 3.496, 3.503, 3.499, 3.504, 3.503, 3.495, 3.501, 
    3.486, 3.493, 3.495, 3.5, 3.498, 3.497, 3.502, 3.496, 3.498, 3.492, 3.494, 3.463, 3.218]
L=zip(L_1,L_2)
L=dict(L)
num_points = len(L_1)
sita=[i*2*math.pi/num_points for i in range(num_points)]


#第一个参数为蚂蚁数量，第二个参数为迭代次数，第三个参数为初始信息素浓度,第四个for循环系数。
####<<****************************参数设置************

num_pop,num_iters,info_org,num_for=50,200,0.001,10

###************************************************>>


def blade_list(each_list):
    each_list=[L[i] for i in each_list]
    return each_list

def list_add_one(each_list):
    each_list=[i+1 for i in each_list]
    return each_list


def cal_total_distance(routine):
    routine_x,routine_y,count=0,0,0
    for each in routine:
        routine_x+=math.sin(sita[count])*L[each+1]
        routine_y+=math.cos(sita[count])*L[each+1]
        count+=1
    goal=routine_x**2+routine_y**2
    return goal**0.5

distance_matrix=np.ones((num_points,num_points))*info_org-np.eye(num_points,num_points)*info_org
##############################################

'''
def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

'''
# %% Do ACA
out_list=[]
out_value=[]
time_start=time.time()
for i in range(num_for):
  aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                size_pop=num_pop, max_iter=num_iters,
                distance_matrix=distance_matrix)########size_pop 为蚂蚁数量，max_iter为迭代次数

  best_x, best_y = aca.run()
  out_list.extend([list(best_x)])
  out_value.append(best_y)
time_cost=time.time()-time_start
out_list=list(map(list_add_one,out_list))
blade_list=list(map(blade_list,out_list))
# %% Plot
print('运行时间:%s,蚂蚁个数:%s,迭代次数:%s,初始信息素浓度:%s,for循环次数:%s.'%(time_cost,num_pop,num_iters,info_org,num_for))
print('out_value:%s'%out_value)
print('out_list:%s'%out_list)
print('blade_list:%s'%blade_list)
