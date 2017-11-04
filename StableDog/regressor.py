#coding:utf-8

from sklearn import linear_model, tree, ensemble
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
import pickle

class Regressor:
    """
    回归器
    """
    def __init__(self):
        raise NotImplementedError

    def train(self, X_train, y_train):
        """
        训练模型
        - X_train: 训练特征集
        - y_train: 训练目标集
        """
        self.regressor.fit(X_train, y_train)

    def test(self, X_test, y_test):
        """
        测试模型
        - X_test: 测试特征集
        - y_test: 测试目标集
        return: 预测目标
        """
        y_test_pred = self.regressor.predict(X_test)
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
        return y_test_pred

    def predict(self, X_test):
        """
        预测
        - X_test: 测试特征集
        return: 预测目标
        """
        y_test_pred = self.regressor.predict(X_test)
        return y_test_pred

    def saveModel(self, output_model_file):
        """
        存储模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'wb') as f:
            pickle.dump(self.regressor, f)

    def loadModel(self, output_model_file):
        """
        加载模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'rb') as f:
            self.regressor = pickle.load(f)

class LinearRegressor(Regressor):
    """
    线性回归器
    """
    def __init__(self):
        """
        初始化模型
        """
        self.regressor = linear_model.LinearRegression()

class RidgeRegressor(Regressor):
    """
    岭回归器
    """
    def __init__(self, alpha=1.0, max_iter_num=10000):
        """
        初始化模型
        - alpha: 正则化强度
        - max_iter_num: 最大迭代次数
        """
        self.regressor = linear_model.Ridge(alpha, fit_intercept=True, max_iter=max_iter_num)

class PolynomialRegressor(Regressor):
    """
    多项式回归器
    """
    def __init__(self, degree=3):
        """
        初始化模型
        - degree: 多项式的次数
        """
        self.polynomial = PolynomialFeatures(degree)
        self.regressor = linear_model.LinearRegression()

    def train(self, X_train, y_train):
        """
        训练模型
        - X_train: 训练特征集
        - y_train: 训练目标集
        """
        X_train_transformed = self.polynomial.fit_transform(X_train)
        self.regressor.fit(X_train_transformed, y_train)
    
    def test(self, X_test, y_test):
        """
        测试模型
        - X_test: 测试特征集
        - y_test: 测试目标集
        return: 预测目标
        """
        X_test_transformed = self.polynomial.fit_transform(X_test)
        y_test_pred = self.regressor.predict(X_test_transformed)
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
        return y_test_pred

    def predict(self, X_test):
        """
        预测
        - X_test: 测试特征集
        return: 预测目标
        """
        X_test_transformed = self.polynomial.fit_transform(X_test)
        y_test_pred = self.regressor.predict(X_test_transformed)
        return y_test_pred

class SGDRegressor(Regressor):
    """
    SGD回归器
    """
    def __init__(self, loss='squared_loss', epoch=50):
        """
        初始化模型
        - loss: 损失函数,'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
        - epoch: 迭代次数
        """
        self.regressor = linear_model.SGDRegressor(loss, n_iter=epoch)

class DecisionTreeRegressor(Regressor):
    """
    决策树回归器
    """
    def __init__(self, depth=4):
        """
        初始化模型
        - depth: 树最大深度
        """
        self.regressor = tree.DecisionTreeRegressor(max_depth=depth)

    def getFeatureImportances(self):
        """
        获得特征的相对重要性
        return: 特征的相对重要性
        """
        return self.regressor.feature_importances_

class AdaBoostRegressor(Regressor):
    """
    AdaBoost决策树回归器
    """
    def __init__(self, depth=4, max_estimators=400, random_seed=None):
        """
        初始化模型
        - depth: 树最大深度
        - max_estimators: 最大容忍误差, 一旦某个基学习器误差超过该值, 学习过程提前终止
        - random_seed: 随机数种子
        """
        self.regressor = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=depth), n_estimators=max_estimators, random_state=random_seed)

    def getFeatureImportances(self):
        """
        获得特征的相对重要性
        return: 特征的相对重要性
        """
        return self.regressor.feature_importances_

class RandomForestRegressor(Regressor):
    """
    随机森林回归器
    """
    def __init__(self, tree_num=1000, depth=10, min_split=2):
        """
        初始化模型
        - tree_num: 森林中树的数量
        - depth: 树的最大深度
        - min_split: 分割内部节点所需的最小样本数量
        """
        self.regressor = ensemble.RandomForestRegressor(n_estimators=tree_num, max_depth=depth, min_samples_split=min_split)

    def getFeatureImportances(self):
        """
        获得特征的相对重要性
        return: 特征的相对重要性
        """
        return self.regressor.feature_importances_
