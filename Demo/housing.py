#coding:utf-8

import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import sys
sys.path.append('../')
import StableDog
from StableDog.regressor import DecisionTreeRegressor, AdaBoostRegressor
from StableDog.data import plot_feature_importances

# 加载房屋价格数据
housing_data = datasets.load_boston()
# 打乱数据顺序
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)
# 切分训练数据集和测试数据集 (80% for training, 20% for testing)
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 决策树回归器
dt_regressor = DecisionTreeRegressor(depth=4)
dt_regressor.train(X_train, y_train)
print("\nDecision Tree performance:\n")
dt_regressor.test(X_test, y_test)

# AdaBoost决策树回归器
ab_regressor = AdaBoostRegressor(depth=4, max_estimators=400, random_seed=7)
ab_regressor.train(X_train, y_train)
print("\AdaBoost performance:\n")
ab_regressor.test(X_test, y_test)

# 绘制特征的相对重要性
plot_feature_importances(dt_regressor.getFeatureImportances(), 
        'Decision Tree regressor', housing_data.feature_names)
plot_feature_importances(ab_regressor.getFeatureImportances(), 
        'AdaBoost regressor', housing_data.feature_names)
