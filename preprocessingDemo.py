#coding:utf-8

import numpy as np
from StableDog import preprocessing

data = np.array([[ 3, -1.5,  2, -5.4],
                 [ 0,  4,  -0.3, 2.1],
                 [ 1,  3.3, -1.9, -4.3]])

# 均值移除
# data_standardized = preprocessing.meanRemoval(data, True)
data_standardized = preprocessing.meanRemoval(data, True)
print("均值移除后数据:\n", data_standardized)

# 缩放数据
data_scaled = preprocessing.scaling(data, -1, 1) # 缩放数据到[-1,1]
print("\n缩放后数据:\n", data_scaled)

# 归一化数据
data_normalized = preprocessing.normalization(data, 'l1') # 独立归一化每一个样本
print("\nL1归一化后数据(独立归一化每一个样本):\n", data_normalized) 
data_normalized = preprocessing.normalization(data, 'l1', 0) # 归一化每一维特征
print("\nL1归一化后数据(归一化每一维特征):\n", data_normalized)

# 二值化数据
data_binarized = preprocessing.binarization(data, 0.0) # 以0为界限划分
print("\n二值化后数据:\n", data_binarized)

# 独热编码
train_data = np.array([[0, 2, 1, 12], 
                       [1, 3, 5, 3], 
                       [2, 3, 2, 12], 
                       [1, 2, 4, 3]])
one_hot_encoder = preprocessing.OneHotEncoder(train_data)
encoded_vector = preprocessing.oneHotEncoding([[2, 3, 4, 3]], one_hot_encoder)
print("\n独热编码后数据:\n", encoded_vector)