#coding:utf-8

import numpy as np
from StableDog import preprocessing

data = np.array([[ 3, -1.5,  2, -5.4],
                 [ 0,  4,  -0.3, 2.1],
                 [ 1,  3.3, -1.9, -4.3]])
print(data)

# 均值移除
# data_standardized = preprocessing.meanRemoval(data, True)
data_standardized = preprocessing.meanRemoval(data, True)
print("\n均值移除后数据:\n", data_standardized)

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
one_hot_encoder = preprocessing.OneHotEncoder(train_data) # 构建编码器
data = [[2, 3, 4, 3]]
encoded_vector = preprocessing.oneHotEncoding(data, one_hot_encoder)
print("\n训练数据:\n", data)
print("\n独热编码后数据:\n", encoded_vector)

# 便签编码
label_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
label_encoder = preprocessing.LabelEncoder(label_classes)

labels = ['toyota', 'ford', 'audi', 'ford']
encoded_labels = preprocessing.labelEncoding(labels, label_encoder)
print("\n原始标签 =", labels)
print("编码后的标签 =", encoded_labels)

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = preprocessing.labelDecoding(encoded_labels, label_encoder)
print("\n编码后的标签 =", encoded_labels)
print("解码后的标签 =", decoded_labels)