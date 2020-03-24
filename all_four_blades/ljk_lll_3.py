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
L_1=[60, 23, 7, 79, 24, 21, 69, 62, 50, 57, 41, 67, 82, 16, 17, 64, 34, 27, 1, 74, 44, 
    39, 76, 32, 26, 38, 63, 51, 35, 53, 56, 68, 75, 45, 18, 65, 49, 59, 31, 11, 61, 22, 48, 
    54, 28, 20, 43, 8, 42, 14, 78, 37, 12, 40, 29, 77, 30, 15, 81, 4, 52, 2, 47, 73, 10, 36, 
    3, 70, 80, 19, 46, 58, 55, 25, 5, 9, 33, 66, 72, 71, 6, 13]
L_2=[18081.277, 18163.373, 18254.33, 18089.252, 18141.826, 18312.807, 18171.732, 18152.018, 
    18118.658, 18112.373, 18183.064, 18211.59, 18208.184, 18095.563, 18152.445, 18134.135, 
    18213.354, 18138.158, 18133.852, 18071.467, 18122.969, 18137.896, 18176.447, 18225.566, 
    18071.133, 18137.135, 18115.564, 18100.682, 18183.803, 18150.445, 18266.281, 18108.041, 
    17975.154,18160.23, 18158.564, 18135.588, 18201.494, 18241.066, 18151.611, 18222.852, 
    18183.137, 18194.352, 18081.016, 18179.684, 18095.658, 18122.896, 18226.02, 18135.396, 
    18143.184, 18249.449, 18067.539, 18172.781, 18174.922, 18127.803, 18120.49, 18183.23, 
    18100.135, 18154.373, 18128.443, 18241.566, 18166.422, 18146.637, 18088.275, 18162.637, 
    18213.781, 17993.346, 18112.348, 18144.063, 18148.564, 18145.967, 18085.943, 18556.672, 
    18111.516, 18039.941, 18039.514, 18162.994, 18165.898, 18227.543, 18035.871, 18155.469, 
    18121.039, 18174.113]
#82叶片
L=zip(L_1,L_2)
L=dict(L)
num_points = len(L_1)
sita=[i*2*math.pi/num_points for i in range(num_points)]


#第一个参数为蚂蚁数量，第二个参数为迭代次数，第三个参数为初始信息素浓度,第四个for循环系数。
####<<****************************参数设置************

num_pop,num_iters,info_org,num_for=500,1500,0.001,10

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
