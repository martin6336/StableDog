# regressor 模块

**回归器模块**

回归是估计输入数据与连续值输出数据之间关系的过程。数据通常是实数形式的，我们的目标是估计满足输入到输出映射关系的基本函数。

该模块提供了从简单的 **线性回归器(LinearRegressor)**、**多项式回归器(PolynomialRegressor)**、**决策树回归器(DecisionTreeRegressor)**到更为实用的 **AdaBoost决策树回归器(AdaBoostRegressor)**、**随机森林回归器(RandomForestRegressor)** 等各种回归器。

## 索引

- 回归器共有方法
  - 训练模型
  - 测试模型
  - 预测
  - 存储模型
  - 加载模型
- 线性回归器(LinearRegressor)
- 岭回归器(RidgeRegressor)
- 多项式回归器(PolynomialRegressor)
- SGD回归器(SGDRegressor)
- 决策树回归器(DecisionTreeRegressor)
- AdaBoost决策树回归器(AdaBoostRegressor)
- 随机森林回归器(RandomForestRegressor)

## 1. 回归器共有方法

以下这些方法为所有回归器所共有，后续不再重复说明。

### 1.1 训练模型

```
Regressor.train(X_train, y_train)
```

- **X_train**: 训练特征集
- **y_train**: 训练目标集

### 1.2 测试模型

```
Regressor.test(X_test, y_test)
```

- **X_test**: 测试特征集
- **y_test**: 测试目标集

return: 预测目标

注：`test()` 会计算并打印包括**平均绝对误差(mean absolute error)**、**均方误差(mean squared error)**、**中位数绝对误差(median absolute error)**、**解释方差分(explained variance score)**、**R方得分(R2 score)**在内的回归器各项评价指标。通常的做法是尽量保证均方误差最低，而且解释方差分最高。

### 1.3 预测

```
Regressor.predict(X_test)
```

- **X_test**: 测试特征集

### 1.4 存储模型

```
Regressor.saveModel(output_model_file)
```

- **output_model_file**: 模型名称

### 1.5 加载模型

```
Regressor.loadModel(output_model_file)
```

- **output_model_file**: 模型名称

## 2. 线性回归器(LinearRegressor)

### 2.1 初始化模型

```
StableDog.regressor.LinearRegressor()
```

线性回归器是最简单的回归器，所以不需要设置参数

示例代码：

```python
# 创建线性回归器
linear_regressor = LinearRegressor()
# 训练
linear_regressor.train(X_train, y_train)
# 预测
y_test_pred = linear_regressor.predict(X_test)
print('Predict the output:\n', y_test_pred)
# 测试
linear_regressor.test(X_test, y_test)
# 模型存储
linear_regressor.saveModel('model_linear_regr.pkl')
# 读取模型
linear_regressor.loadModel('model_linear_regr.pkl')
linear_regressor.test(X_test, y_test)
```

## 3. 岭回归器(RidgeRegressor)

### 3.1 初始化模型 

```
StableDog.regressor.RidgeRegressor(alpha=1.0, max_iter_num=10000)
```

- **alpha**: 正则化强度
- **max_iter_num**: 最大迭代次数

示例代码：

```python
# 创建岭回归器
ridge_regressor = RidgeRegressor(alpha=100)
# 训练
ridge_regressor.train(X_train, y_train)
# 预测
y_test_pred_ridge = ridge_regressor.predict(X_test)
print('\nRIDGE Predict the output:\n', y_test_pred_ridge)
# 测试
print('\nRIDGE:')
ridge_regressor.test(X_test, y_test)
```

## 4. 多项式回归器(PolynomialRegressor)

### 4.1 初始化模型

```
StableDog.regressor.PolynomialRegressor(degree=3)
```

- **degree**: 多项式的次数

示例代码：

```python
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
```

## 5. SGD回归器(SGDRegressor)

### 5.1 初始化模型

```
StableDog.regressor.SGDRegressor(loss='squared_loss', epoch=50)
```

- **loss**: 损失函数,'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
- **epoch**: 迭代次数

示例代码：

```python
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
```

## 6. 决策树回归器(DecisionTreeRegressor)

### 6.1 初始化模型

```
StableDog.regressor.DecisionTreeRegressor(depth=4)
```

- **depth**: 树最大深度

示例代码：

```python
# 决策树回归器
dt_regressor = DecisionTreeRegressor(depth=4)
# 训练
dt_regressor.train(X_train, y_train)
# 测试
print("\nDecision Tree performance:\n")
dt_regressor.test(X_test, y_test)
# 绘制特征的相对重要性
plot_feature_importances(dt_regressor.getFeatureImportances(), 'Decision Tree regressor', housing_data.feature_names)
```

## 7. AdaBoost决策树回归器(AdaBoostRegressor)

### 7.1 初始化模型

```
StableDog.regressor.AdaBoostRegressor(depth=4, max_estimators=400, random_seed=None)
```

- **depth**: 树最大深度
- **max_estimators**: 最大容忍误差, 一旦某个基学习器误差超过该值, 学习过程提前终止
- **random_seed**: 随机数种子

### 7.2 获得特征的相对重要性

```
AdaBoostRegressor.getFeatureImportances()
```

**return**: 特征的相对重要性

示例代码：

```python
# AdaBoost决策树回归器
ab_regressor = AdaBoostRegressor(depth=4, max_estimators=400, random_seed=7)
# 训练
ab_regressor.train(X_train, y_train)
# 测试
print("\AdaBoost performance:\n")
ab_regressor.test(X_test, y_test)
# 绘制特征的相对重要性
plot_feature_importances(ab_regressor.getFeatureImportances(), 'AdaBoost regressor', housing_data.feature_names)
```

### 8. 随机森林回归器(RandomForestRegressor)

### 8.1 初始化模型

```
StableDog.regressor.RandomForestRegressor(tree_num=1000, depth=10, min_split=2)
```

- **tree_num**: 森林中树的数量
- **depth**: 树的最大深度
- **min_split**: 分割内部节点所需的最小样本数量

### 8.2 获得特征的相对重要性 

```
RandomForestRegressor.getFeatureImportances()
```

**return**: 特征的相对重要性

示例代码：

```python
# 随机森林回归器
rf_regressor = RandomForestRegressor(tree_num=1000, depth=10, min_split=2)
# 训练
rf_regressor.train(X_train, y_train)
# 测试
rf_regressor.test(X_test, y_test)
# 绘制特征的相对重要性
plot_feature_importances(rf_regressor.getFeatureImportances(), 'Random Forest regressor', feature_names)
```

## 9. Demo

- [regressionSinglevarDemo.py](https://github.com/jsksxs360/StableDog/blob/master/Demo/regressionSinglevarDemo.py)
- [regressionMultivarDemo.py](https://github.com/jsksxs360/StableDog/blob/master/Demo/regressionMultivarDemo.py)
- [housing.py](https://github.com/jsksxs360/StableDog/blob/master/Demo/housing.py)
- [bikeSharing.py](https://github.com/jsksxs360/StableDog/blob/master/Demo/bikeSharing.py)