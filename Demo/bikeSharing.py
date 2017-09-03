#coding:utf-8

import sys
sys.path.append('../')
import StableDog
from StableDog.data import FileReader
from StableDog.data import plot_feature_importances
from StableDog.regressor import RandomForestRegressor
import numpy as np

# 加载数据文件
fileReader = FileReader('../testData/bike_day.csv', csv_file=True)
# 按 0.8 切分训练集、测试集
X_train, y_train, X_test, y_test, feature_names = fileReader.getSamples(feature_name=True, split=True, 
                                                                        training_prop=0.8, shuffle=True, random_seed=7)
X_train = np.array([X[2:13] for X in X_train])
X_test = np.array([X[2:13] for X in X_test])
# 转换数据类型
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# 随机森林回归器
rf_regressor = RandomForestRegressor(tree_num=1000, depth=10, min_split=2)
rf_regressor.train(X_train, y_train)
rf_regressor.test(X_test, y_test)

# 绘制特征的相对重要性
plot_feature_importances(rf_regressor.getFeatureImportances(), 'Random Forest regressor', feature_names)

