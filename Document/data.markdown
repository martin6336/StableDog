# data 模块

**文件数据处理模块**

提供了从数据文件(文本、CSV格式)读取数据并构建数据集合的函数，并且包含了随机打乱数据、划分训练集测试集等功能，简化了数据处理的过程。

## 索引

- [StableDog.data.FileReader 类](#1-stabledogdatafilereader-类)
  - [构造函数](#11-构造函数)
  - [获得样本](#12-获得样本)
- [绘制特征的相对重要性](#2-绘制特征的相对重要性)

## 1. StableDog.data.FileReader 类

### 1.1 构造函数

```
StableDog.data.FileReader(filename, csv_file=False, delimiter=' ', include_label=True, log=False)
```

- **filename**: 文件名称
- **csv_file**: 是否是CSV文件
- **delimiter**: 数据分隔符(如果是CSV文件忽略该属性)
- **include_label**: 是否包含标签(默认True)
- **log**: 是否打印日志(默认True)

### 1.2 获得样本

```
FileReader.getSamples(feature_name=False, split=False, training_prop=0.8, shuffle=False, random_seed=7)
```

- **feature_name**: 是否包含特征名称
- **split**: 是否划分训练测试集
- **training_prop**: 训练集比例
- **shuffle**: 是否打乱样本
- **random_seed**: 打乱时的随机数种子

**return**:   
(不划分)特征集, 目标集;  
(划分)训练特征集, 训练目标集, 测试特征集, 测试目标集  
如果 feature_name=True, 还将返回多返回一个特征名称

示例代码：

```python
# 读取文本文件
# 每行包含多个特征值和一个目标值，使用','分隔
fileReader = FileReader('../testData/data_singlevar.txt', csv_file=False, delimiter=',')
# 按 0.8 切分训练集、测试集
X_train, y_train, X_test, y_test = fileReader.getSamples(feature_name=False, split=True, training_prop=0.8)

# 读取CSV文件
fileReader = FileReader('../testData/bike_day.csv', csv_file=True)
# 按 0.8 切分训练集、测试集
X_train, y_train, X_test, y_test, feature_names = fileReader.getSamples(feature_name=True, split=True, training_prop=0.8, shuffle=True, random_seed=7)
```

示例代码(线性回归)：

```python
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
```

## 2. 绘制特征的相对重要性

```
StableDog.data.plot_feature_importances(feature_importances, title, feature_names)
```

- **feature_importances**: 特征的相对重要性
- **title**: 图片标题
- **feature_names**: 特征名称

注：feature_importances 通常由回归器的 `getFeatureImportances()` 获得。

示例代码：

```python
import StableDog
from StableDog.data import FileReader
from StableDog.data import plot_feature_importances
from StableDog.regressor import RandomForestRegressor
import numpy as np

# 加载数据文件
fileReader = FileReader('../testData/bike_day.csv', csv_file=True)
# 按 0.8 切分训练集、测试集
X_train, y_train, X_test, y_test, feature_names = fileReader.getSamples(feature_name=True, split=True, training_prop=0.8, shuffle=True, random_seed=7)
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
```

