# StableDog
一个简易的机器学习框架，实现了对 scikit-learn 等已有机器学习库的进一步封装。

注：该项目处于起步阶段，还在建设中。

## 下载

暂无

## 使用

暂无

## 说明

### 1. 数据预处理

StableDog 通过 [preprocessing 模块](https://github.com/jsksxs360/StableDog/blob/master/Document/preprocessing.markdown)提供了常用的一些数据预处理技术，包括 **均值移除(Mean removal)**、**范围缩放(Scaling)**、**归一化(Normalization)**、**二值化(Binarization)** 等。

通过 [data 模块](https://github.com/jsksxs360/StableDog/blob/master/Document/data.markdown)提供了从数据文件(文本、CSV格式)读取数据并构建数据集合的函数，并且包含了随机打乱数据、划分训练集测试集等功能，简化了数据处理的过程。

### 2. 回归器

回归是估计输入数据与连续值输出数据之间关系的过程。数据通常是实数形式的，我们的目标是估计满足输入到输出映射关系的基本函数。

StableDog 通过 [regressor 模块](https://github.com/jsksxs360/StableDog/blob/master/Document/regressor.markdown)提供了从简单的 **线性回归器(LinearRegressor)**、**多项式回归器(PolynomialRegressor)**、**决策树回归器(DecisionTreeRegressor)** 到更为实用的 **AdaBoost决策树回归器(AdaBoostRegressor)**、**随机森林回归器(RandomForestRegressor)** 等各种回归器。

