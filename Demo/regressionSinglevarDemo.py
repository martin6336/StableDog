#coding:utf-8

import sys
sys.path.append('../')
import StableDog
from StableDog.data import FileReader
from StableDog.regressor import LinearRegressor
import numpy as np

fileReader = FileReader('../testData/data_singlevar.txt', csv_file=False, delimiter=',')

# 按 0.8 切分训练集、测试集
X_train, y_train, X_test, y_test = fileReader.getSamples(feature_name=False, split=True, training_prop=0.8)

# 转换数据类型
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# 创建线性回归器
linear_regressor = LinearRegressor()

# 训练
linear_regressor.train(X_train, y_train)

# 预测
y_test_pred = linear_regressor.predict(X_test)
print('Predict the output:\n', y_test_pred)

# 测试
linear_regressor.test(X_test, y_test)

# 可视化拟合效果
import matplotlib.pyplot as plt

# 训练集拟合可视化
y_train_pred = linear_regressor.predict(X_train)
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.xticks(())
plt.yticks(())
plt.show()

# 测试集拟合可视化
y_test_pred = linear_regressor.predict(X_test)
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.xticks(())
plt.yticks(())
plt.show()

# 模型存储
linear_regressor.saveModel('model_linear_regr.pkl')

# 读取模型
linear_regressor.loadModel('model_linear_regr.pkl')
linear_regressor.test(X_test, y_test)
