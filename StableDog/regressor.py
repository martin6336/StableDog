#coding:utf-8

from sklearn import linear_model
import sklearn.metrics as sm
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
        return: 预测目标集
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
        return: 预测目标集
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
