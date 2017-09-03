# preprocessing 模块

**数据预处理模块**

包含常用的一些数据预处理技术，包括`均值移除(Mean removal)`、`范围缩放(Scaling)`、`归一化(Normalization)`、`二值化(Binarization)`等。

## 索引

- [均值移除(Mean removal)](#1-均值移除mean-removal)
- [缩放数据(Scaling)](#2-缩放数据scaling)
- [归一化数据(Normalization)](#3-归一化数据normalization)
- [二值化数据(Binarization)](#4-二值化数据binarization)
- [独热编码](#5-独热编码)
  - [独热编码器](#51-独热编码器)
  - [独热编码](#52-独热编码)
- [标签编码](#6-标签编码)
  - [标签编码器](#61-标签编码器)
  - [标签编码](#62-标签编码)
  - [标签解码](#63-标签解码)
- [Demo](#demo)

## 1. 均值移除(Mean removal)

```python
StableDog.preprocessing.meanRemoval(data, log=False)
```

- **data**: 待处理数据
- **log**: 打印信息

**return**: 均值移除后的数据

示例代码：

```python
>>> data = np.array([[ 3, -1.5,  2, -5.4],
                     [ 0,  4,  -0.3, 2.1],
                     [ 1,  3.3, -1.9, -4.3]])

>>> data_standardized = preprocessing.meanRemoval(data, True)
>>> print("\n均值移除后数据:\n", data_standardized)
```

输出：

```
mean removal success!
Mean = [  5.55111512e-17  -1.11022302e-16  -7.40148683e-17  -7.40148683e-17]
Std deviation = [ 1.  1.  1.  1.]
均值移除后数据:
 [[ 1.33630621 -1.40451644  1.29110641 -0.86687558]
 [-1.06904497  0.84543708 -0.14577008  1.40111286]
 [-0.26726124  0.55907936 -1.14533633 -0.53423728]]
```

## 2. 缩放数据(Scaling)

```python
StableDog.preprocessing.scaling(data, start_range, end_range)
```

- **data**: 待处理数据
- **start_range**: 缩放下界
- **end_range**: 缩放上界

**return**: 缩放后的数据

示例代码：

```python
>>> data_scaled = preprocessing.scaling(data, -1, 1) # 缩放数据到[-1,1]
>>> print("\n缩放后数据:\n", data_scaled)
```

输出：

```
缩放后数据:
 [[ 1.         -1.          1.         -1.        ]
 [-1.          1.         -0.17948718  1.        ]
 [-0.33333333  0.74545455 -1.         -0.70666667]]
```

## 3. 归一化数据(Normalization)

```
StableDog.preprocessing.normalization(data, norm='l1', axis=1)
```

- **data**: 待处理数据
- **norm**: 'l1', 'l2', or 'max'.(默认'l1')
- **axis**: 取1, 独立地归一化每一个样本; 取0, 归一化每一维特征.(默认1)

**return**: 归一化后的数据

示例代码：

```python
>>> data_normalized = preprocessing.normalization(data, 'l1') # 独立归一化每一个样本
>>> print("\nL1归一化后数据(独立归一化每一个样本):\n", data_normalized) 
>>> data_normalized = preprocessing.normalization(data, 'l1', 0) # 归一化每一维特征
>>> print("\nL1归一化后数据(归一化每一维特征):\n", data_normalized)
```

输出：

```
L1归一化后数据(独立归一化每一个样本):
 [[ 0.25210084 -0.12605042  0.16806723 -0.45378151]
 [ 0.          0.625      -0.046875    0.328125  ]
 [ 0.0952381   0.31428571 -0.18095238 -0.40952381]]
L1归一化后数据(归一化每一维特征):
 [[ 0.75       -0.17045455  0.47619048 -0.45762712]
 [ 0.          0.45454545 -0.07142857  0.1779661 ]
 [ 0.25        0.375      -0.45238095 -0.36440678]]
```

## 4. 二值化数据(Binarization)

```
StableDog.preprocessing.binarization(data, threshold=0.0)
```

- **data**: 待处理数据
- **threshold**: 划分界限, 小于等于界限->0, 大于界限->1.(默认0.0)

**return**: 二值化后的数据

示例代码：

```python
>>> data_binarized = preprocessing.binarization(data, 0.0) # 以0为界限划分
>>> print("\n二值化后数据:\n", data_binarized)
```

输出：

```
二值化后数据:
 [[ 1.  0.  1.  0.]
 [ 0.  1.  0.  1.]
 [ 1.  1.  0.  0.]]
```

## 5. 独热编码

### 5.1 独热编码器

```
StableDog.preprocessing.OneHotEncoder(train_data)
```

- **train_data**: 训练数据

**return**: 独热编码器

### 5.2 独热编码

```
StableDog.preprocessing.oneHotEncoding(data, one_hot_encoder)
```

- **data**: 待编码数据
- **one_hot_encoder**: 独热编码器

**return**: 独热编码后的数据

示例代码：

```python
>>> train_data = np.array([[0, 2, 1, 12], 
                           [1, 3, 5, 3], 
                           [2, 3, 2, 12], 
                           [1, 2, 4, 3]])
>>> one_hot_encoder = preprocessing.OneHotEncoder(train_data) # 构建编码器
>>> data = [[2, 3, 4, 3]]
>>> encoded_vector = preprocessing.oneHotEncoding(data, one_hot_encoder)
>>> print("\n数据:\n", data)
>>> print("\n独热编码后数据:\n", encoded_vector)
```

输出：

```
数据:
 [[2, 3, 4, 3]]
独热编码后数据:
 [[ 0.  0.  1.  0.  1.  0.  0.  1.  0.  1.  0.]]
```

## 6. 标签编码

### 6.1 标签编码器

```
StableDog.preprocessing.LabelEncoder(label_classes, log=True)
```

- **label_classes**: 标签列表
- **log**: 打印信息(默认打印)

**return**: 标签编码器

### 6.2 标签编码

```
StableDog.preprocessing.labelEncoding(labels, label_encoder)
```

- **labels**: 标签列表
- **label_encoder**: 标签编码器

**return**: 编码后的标签列表

### 6.3 标签解码

```
StableDog.preprocessing.labelDecoding(encoded_labels, label_encoder)
```

- **encoded_labels**: 编码后的标签列表
- **label_encoder**: 标签编码器

**return**: 原始标签列表

示例代码：

```python
>>> label_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
>>> label_encoder = preprocessing.LabelEncoder(label_classes)

>>> labels = ['toyota', 'ford', 'audi', 'ford']
>>> encoded_labels = preprocessing.labelEncoding(labels, label_encoder)
>>> print("\n原始标签 =", labels)
>>> print("编码后的标签 =", encoded_labels)

>>> encoded_labels = [2, 1, 0, 3, 1]
>>> decoded_labels = preprocessing.labelDecoding(encoded_labels, label_encoder)
>>> print("\n编码后的标签 =", encoded_labels)
>>> print("解码后的标签 =", decoded_labels)
```

输出：

```
Class mapping:
audi --> 0
bmw --> 1
ford --> 2
toyota --> 3

原始标签 = ['toyota', 'ford', 'audi', 'ford']
编码后的标签 = [3, 2, 0, 2]

编码后的标签 = [2, 1, 0, 3, 1]
解码后的标签 = ['ford', 'bmw', 'audi', 'toyota', 'bmw']
```

## Demo

- [preprocessingDemo.py](https://github.com/jsksxs360/StableDog/blob/master/Demo/preprocessingDemo.py)