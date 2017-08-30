#coding:utf-8

import numpy as np
from sklearn import preprocessing

def meanRemoval(data, log=False):
    """
    均值移除
    - data: 待处理数据
    - log: 打印信息
    return: 均值移除后的数据
    """
    data_standardized = preprocessing.scale(data)
    if log:
        print('mean removal success!')
        print('Mean =', data_standardized.mean(axis=0))
        print('Std deviation =', data_standardized.std(axis=0))
    return data_standardized

def scaling(data, min, max):
    """
    缩放数据
    - data: 待处理数据
    - min: 缩放下界
    - max: 缩放上界
    return: 缩放后的数据
    """
    data_scaler = preprocessing.MinMaxScaler(feature_range=(min, max))
    data_scaled = data_scaler.fit_transform(data)
    return data_scaled

def normalization(data, norm='l1', axis=1):
    """
    归一化数据
    - data: 待处理数据
    - norm: 'l1', 'l2', or 'max'.(默认'l1')
    - axis: 取1, 独立地归一化每一个样本; 取0, 归一化每一维特征.(默认1)
    return: 归一化后的数据
    """
    data_normalized = preprocessing.normalize(data, norm, axis)
    return data_normalized

def binarization(data, threshold=0.0):
    """
    二值化数据
    - data: 待处理数据
    - threshold: 划分界限, 小于等于界限->0, 大于界限->1.(默认0.0)
    return: 二值化后的数据
    """
    binarizer = preprocessing.Binarizer(threshold)
    data_binarized = binarizer.transform(data)
    return data_binarized

def OneHotEncoder(train_data):
    """
    独热编码器
    - train_data: 训练数据
    return: 独热编码器
    """
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(train_data)
    return encoder

def oneHotEncoding(data, one_hot_encoder):
    """
    独热编码
    - data: 待编码数据
    - one_hot_encoder: 独热编码器
    return: 独热编码后的数据
    """
    encoded_vector = one_hot_encoder.transform(data).toarray()
    return encoded_vector

def LabelEncoder(label_classes, log=True):
    """
    标签编码器
    - label_classes: 标签列表
    - log: 打印信息(默认打印)
    return: 标签编码器
    """
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(label_classes)
    if log:
        print("\nClass mapping:")
        for i, item in enumerate(label_encoder.classes_):
            print(item, '-->', i)
    return label_encoder

def labelEncoding(labels, label_encoder):
    """
    标签编码
    - labels: 标签列表
    - label_encoder: 标签编码器
    return: 编码后的标签列表
    """
    encoded_labels = label_encoder.transform(labels)
    return list(encoded_labels)

def labelDecoding(encoded_labels, label_encoder):
    """
    标签解码
    - encoded_labels: 编码后的标签列表
    - label_encoder: 标签编码器
    return: 原始标签列表
    """
    decoded_labels = label_encoder.inverse_transform(encoded_labels)
    return list(decoded_labels)
