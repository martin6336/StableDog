#coding:utf-8

import numpy as np

class FileReader:
    """
    数据文件读取器
    """
    def __init__(self, filename, split=' ', include_label=True, log=False):
        """
        构造函数
        - filename: 文件名称
        - split: 数据分隔符
        - include_label: 是否包含标签(默认True)
        - log: 是否打印日志(默认True)
        """
        self.features = list()
        self.labels = list()
        with open(filename, 'r') as f:
            for line in f.readlines():
                items = line.strip().split(split)
                if include_label:
                    x_vector = [float(x) for x in items[:-1]]
                    self.features.append(x_vector)
                    self.labels.append(items[-1])
                else:
                    x_vector = [float(x) for x in items]
                    self.features.append(x_vector)
        if log:
            if len(self.features) == 0:
                print("file is empty!")
            else:
                print("Find", len(self.features), "samples.")
                print("Each sample has", len(self.features[0]), "features.")

    def getSamples(self, split=False, training_prop=0.8):
        """
        获得样本
        - split: 是否划分训练测试集
        - training_prop: 训练集比例
        return: 
        (不划分)特征集, 目标集;
        (划分)训练特征集, 训练目标集, 测试特征集, 测试目标集
        """

        if not split:
            X = np.array(self.features)
            y = np.array(self.labels)
            return X, y

        num_training = int(training_prop * len(self.labels))
        num_test = len(self.labels) - num_training
        
        X_train = np.array(self.features[:num_training])
        y_train = np.array(self.labels[:num_training])
        X_test = np.array(self.features[num_training:])
        y_test = np.array(self.labels[num_training:])
        return X_train, y_train, X_test, y_test
