#coding:utf-8

from sklearn import linear_model, tree, ensemble
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
import pickle

class LinearRegressor:
    """
    线性回归器
    """
    def train(self, X_train, y_train):
        """
        训练模型
        - X_train 训练特征集
        - y_train 训练目标集
        """
        self.__linear_regressor = linear_model.LinearRegression()
        self.__linear_regressor.fit(X_train, y_train)
    
    def test(self, X_test, y_test):
        """
        测试模型
        - X_test 测试特征集
        - y_test 测试目标集
        return: 预测目标
        """
        y_test_pred = self.__linear_regressor.predict(X_test)
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
        return y_test_pred

    def predict(self, X_test):
        """
        预测
        - X_test 测试特征集
        return: 预测目标
        """
        y_test_pred = self.__linear_regressor.predict(X_test)
        return y_test_pred

    def saveModel(self, output_model_file):
        """
        存储模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'wb') as f:
            pickle.dump(self.__linear_regressor, f)

    def loadModel(self, output_model_file):
        """
        加载模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'rb') as f:
            self.__linear_regressor = pickle.load(f)

class RidgeRegressor:
    """
    岭回归器
    """
    def __init__(self, alpha=1.0, max_iter_num=10000):
        """
        初始化模型
        - alpha: 正则化强度(默认0.01)
        - max_iter_num: 最大迭代次数
        """
        self.__ridge_regressor = linear_model.Ridge(alpha, fit_intercept=True, max_iter=max_iter_num)

    def train(self, X_train, y_train):
        """
        训练模型
        - X_train 训练特征集
        - y_train 训练目标集
        """
        self.__ridge_regressor.fit(X_train, y_train)
    
    def test(self, X_test, y_test):
        """
        测试模型
        - X_test 测试特征集
        - y_test 测试目标集
        return: 预测目标
        """
        y_test_pred = self.__ridge_regressor.predict(X_test)
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
        return y_test_pred

    def predict(self, X_test):
        """
        预测
        - X_test 测试特征集
        return: 预测目标
        """
        y_test_pred = self.__ridge_regressor.predict(X_test)
        return y_test_pred

    def saveModel(self, output_model_file):
        """
        存储模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'wb') as f:
            pickle.dump(self.__ridge_regressor, f)

    def loadModel(self, output_model_file):
        """
        加载模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'rb') as f:
            self.__ridge_regressor = pickle.load(f)

class PolynomialRegressor:
    """
    多项式回归器
    """
    def __init__(self, degree=3):
        """
        初始化模型
        - degree: 多项式的次数
        """
        self.__polynomial = PolynomialFeatures(degree)
        self.__poly_linear_model = linear_model.LinearRegression()

    def train(self, X_train, y_train):
        """
        训练模型
        - X_train 训练特征集
        - y_train 训练目标集
        """
        X_train_transformed = self.__polynomial.fit_transform(X_train)
        self.__poly_linear_model.fit(X_train_transformed, y_train)
    
    def test(self, X_test, y_test):
        """
        测试模型
        - X_test 测试特征集
        - y_test 测试目标集
        return: 预测目标
        """
        X_test_transformed = self.__polynomial.fit_transform(X_test)
        y_test_pred = self.__poly_linear_model.predict(X_test_transformed)
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
        return y_test_pred

    def predict(self, X_test):
        """
        预测
        - X_test 测试特征集
        return: 预测目标
        """
        X_test_transformed = self.__polynomial.fit_transform(X_test)
        y_test_pred = self.__poly_linear_model.predict(X_test_transformed)
        return y_test_pred

    def saveModel(self, output_model_file):
        """
        存储模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'wb') as f:
            pickle.dump(self.__poly_linear_model, f)

    def loadModel(self, degree, output_model_file):
        """
        加载模型
        - output_model_file: 模型名称
        """
        self.__polynomial = PolynomialFeatures(degree)
        with open(output_model_file, 'rb') as f:
            self.__poly_linear_model = pickle.load(f)

class SGDRegressor:
    """
    SGD回归器
    """
    def __init__(self, loss='squared_loss', epoch=50):
        """
        初始化模型
        - loss: 损失函数,'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
        - epoch: 迭代次数
        """
        self.__sgd_regressor = linear_model.SGDRegressor(loss, n_iter=epoch)

    def train(self, X_train, y_train):
        """
        训练模型
        - X_train 训练特征集
        - y_train 训练目标集
        """
        self.__sgd_regressor.fit(X_train, y_train)
    
    def test(self, X_test, y_test):
        """
        测试模型
        - X_test 测试特征集
        - y_test 测试目标集
        return: 预测目标
        """
        y_test_pred = self.__sgd_regressor.predict(X_test)
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
        return y_test_pred

    def predict(self, X_test):
        """
        预测
        - X_test 测试特征集
        return: 预测目标
        """
        y_test_pred = self.__sgd_regressor.predict(X_test)
        return y_test_pred

    def saveModel(self, output_model_file):
        """
        存储模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'wb') as f:
            pickle.dump(self.__sgd_regressor, f)

    def loadModel(self, output_model_file):
        """
        加载模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'rb') as f:
            self.__sgd_regressor = pickle.load(f)

class DecisionTreeRegressor:
    """
    决策树回归器
    """
    def __init__(self, depth=4):
        """
        初始化模型
        - depth: 树最大深度
        """
        self.__dt_regressor = tree.DecisionTreeRegressor(max_depth=depth)

    def train(self, X_train, y_train):
        """
        训练模型
        - X_train 训练特征集
        - y_train 训练目标集
        """
        self.__dt_regressor.fit(X_train, y_train)
    
    def test(self, X_test, y_test):
        """
        测试模型
        - X_test 测试特征集
        - y_test 测试目标集
        return: 预测目标
        """
        y_test_pred = self.__dt_regressor.predict(X_test)
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
        return y_test_pred

    def predict(self, X_test):
        """
        预测
        - X_test 测试特征集
        return: 预测目标
        """
        y_test_pred = self.__dt_regressor.predict(X_test)
        return y_test_pred

    def getFeatureImportances(self):
        """
        获得特征的相对重要性
        return: 特征的相对重要性
        """
        return self.__dt_regressor.feature_importances_

    def saveModel(self, output_model_file):
        """
        存储模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'wb') as f:
            pickle.dump(self.__dt_regressor, f)

    def loadModel(self, output_model_file):
        """
        加载模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'rb') as f:
            self.__dt_regressor = pickle.load(f)

class AdaBoostRegressor:
    """
    AdaBoost决策树回归器
    """
    def __init__(self, depth=4, max_estimators=400, random_seed=None):
        """
        初始化模型
        - depth: 树最大深度
        """
        self.__ab_regressor = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=depth), n_estimators=max_estimators, random_state=random_seed)

    def train(self, X_train, y_train):
        """
        训练模型
        - X_train 训练特征集
        - y_train 训练目标集
        """
        self.__ab_regressor.fit(X_train, y_train)
    
    def test(self, X_test, y_test):
        """
        测试模型
        - X_test 测试特征集
        - y_test 测试目标集
        return: 预测目标
        """
        y_test_pred = self.__ab_regressor.predict(X_test)
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
        return y_test_pred

    def predict(self, X_test):
        """
        预测
        - X_test 测试特征集
        return: 预测目标
        """
        y_test_pred = self.__ab_regressor.predict(X_test)
        return y_test_pred

    def getFeatureImportances(self):
        """
        获得特征的相对重要性
        return: 特征的相对重要性
        """
        return self.__ab_regressor.feature_importances_

    def saveModel(self, output_model_file):
        """
        存储模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'wb') as f:
            pickle.dump(self.__ab_regressor, f)

    def loadModel(self, output_model_file):
        """
        加载模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'rb') as f:
            self.__ab_regressor = pickle.load(f)

class RandomForestRegressor:
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
        self.__rf_regressor = ensemble.RandomForestRegressor(n_estimators=tree_num, max_depth=depth, min_samples_split=min_split)

    def train(self, X_train, y_train):
        """
        训练模型
        - X_train 训练特征集
        - y_train 训练目标集
        """
        self.__rf_regressor.fit(X_train, y_train)
    
    def test(self, X_test, y_test):
        """
        测试模型
        - X_test 测试特征集
        - y_test 测试目标集
        return: 预测目标
        """
        y_test_pred = self.__rf_regressor.predict(X_test)
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
        print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
        print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
        print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
        print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
        return y_test_pred

    def predict(self, X_test):
        """
        预测
        - X_test 测试特征集
        return: 预测目标
        """
        y_test_pred = self.__rf_regressor.predict(X_test)
        return y_test_pred

    def getFeatureImportances(self):
        """
        获得特征的相对重要性
        return: 特征的相对重要性
        """
        return self.__rf_regressor.feature_importances_

    def saveModel(self, output_model_file):
        """
        存储模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'wb') as f:
            pickle.dump(self.__rf_regressor, f)

    def loadModel(self, output_model_file):
        """
        加载模型
        - output_model_file: 模型名称
        """
        with open(output_model_file, 'rb') as f:
            self.__rf_regressor = pickle.load(f)