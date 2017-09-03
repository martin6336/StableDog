#coding:utf-8

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import utils

class FileReader:
    """
    数据文件读取器
    """
    def __init__(self, filename, csv_file=False, delimiter=' ', include_label=True, log=False):
        """
        构造函数
        - filename: 文件名称
        - csv_file: 是否是CSV文件
        - delimiter: 数据分隔符(如果是CSV文件忽略该属性)
        - include_label: 是否包含标签(默认True)
        - log: 是否打印日志(默认True)
        """
        self.features = list()
        self.labels = list()
        self.feature_name = None
        if csv_file:
            file_reader = csv.reader(open(filename, 'r'), delimiter=',')
            for row in file_reader:
                if include_label:
                    self.features.append(row[:-1])
                    self.labels.append(row[-1])
                else:
                    self.features.append(row)
            self.feature_name = np.array(self.features[0])
            if include_label:
                self.features = self.features[1:]
                self.labels = self.labels[1:]
            else:
                self.features = self.features[1:]
        else:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    items = line.strip().split(delimiter)
                    if include_label:
                        self.features.append(items[:-1])
                        self.labels.append(items[-1])
                    else:
                        self.features.append(items)
        if log:
            if len(self.features) == 0:
                print("file is empty!")
            else:
                print("Find", len(self.features), "samples.")
                print("Each sample has", len(self.features[0]), "features.")

    def getSamples(self, feature_name=False, split=False, training_prop=0.8, shuffle=False, random_seed=7):
        """
        获得样本
        - feature_name: 是否包含特征名称
        - split: 是否划分训练测试集
        - training_prop: 训练集比例
        - shuffle: 是否打乱样本
        - random_seed: 打乱时的随机数种子
        return: 
        (不划分)特征集, 目标集;
        (划分)训练特征集, 训练目标集, 测试特征集, 测试目标集
        如果 feature_name=True, 还将返回多返回一个特征名称
        """
        X = np.array(self.features)
        y = np.array(self.labels)
        if shuffle:
            X, y = utils.shuffle(X, y, random_state=random_seed)
        if not split:
            if feature_name:
                return X, y, self.feature_name
            else:
                return X, y

        num_training = int(training_prop * len(self.labels))
        X_train = X[:num_training]
        y_train = y[:num_training]
        X_test = X[num_training:]
        y_test = y[num_training:]
        if feature_name:
            return X_train, y_train, X_test, y_test, self.feature_name
        else:
            return X_train, y_train, X_test, y_test

def plot_feature_importances(feature_importances, title, feature_names):
    """
    绘制特征的相对重要性
    - feature_importances: 特征的相对重要性
    - title: 图片标题
    - feature_names: 特征名称
    """
    # 将重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # 将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))
    # 让X坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5
    # 画条形图
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()