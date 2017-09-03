#coding:utf-8

import sys
sys.path.append('../')
import StableDog
from StableDog.data import FileReader
from StableDog.regressor import LinearRegressor, RidgeRegressor
from StableDog.regressor import PolynomialRegressor
from StableDog.regressor import SGDRegressor

fileReader = FileReader('../testData/data_multivar.txt',',')

# 按 0.8 切分训练集、测试集
X_train, y_train, X_test, y_test = fileReader.getSamples(True, training_prop=0.8)

# 将 y 转换为 float
y_train = [float(y) for y in y_train]
y_test = [float(y) for y in y_test]


# 创建线性回归器和岭回归器
linear_regressor = LinearRegressor()
ridge_regressor = RidgeRegressor(alpha=100)
# 训练
linear_regressor.train(X_train, y_train)
ridge_regressor.train(X_train, y_train)
# 预测
y_test_pred = linear_regressor.predict(X_test)
print('\nLINEAR Predict the output:\n', y_test_pred)
y_test_pred_ridge = ridge_regressor.predict(X_test)
print('\nRIDGE Predict the output:\n', y_test_pred_ridge)
# 测试
print('\nLINEAR:')
linear_regressor.test(X_test, y_test)
print('\nRIDGE:')
ridge_regressor.test(X_test, y_test)


# 创建多项式回归器
polynomial_regressor = PolynomialRegressor(degree=3)
# 训练
polynomial_regressor.train(X_train, y_train)
# 预测
y_test_pred_pol = polynomial_regressor.predict(X_test)
print('\nPOLYNOMIAL Predict the output:\n', y_test_pred_pol)
# 测试
print('\nPOLYNOMIAL:')
polynomial_regressor.test(X_test, y_test)


# 创建SGD回归器
sgd_regressor = SGDRegressor(loss='squared_loss', epoch=100)
# 训练
sgd_regressor.train(X_train, y_train)
# 预测
y_test_pred_sgd = sgd_regressor.predict(X_test)
print('\nSGD Predict the output:\n', y_test_pred_sgd)
# 测试
print('\nSGD:')
sgd_regressor.test(X_test, y_test)