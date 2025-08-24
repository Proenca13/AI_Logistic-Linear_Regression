import numpy as np
import base_model
from utils import Round_Squared_Mean_Error
class LinearRegression(base_model.BaseModel):

    def __init__(self, learning_rate, iterations = 1000,lambda_ = 0.01):
        super().__init__(learning_rate, iterations,lambda_)

    def fit(self, X_train, y_train):
        self.weights = np.random.rand(X_train.shape[1])
        self.bias = 0
        for _ in range(self.iterations):
            self.gradient_descent(X_train, y_train)

    def predict(self, X_test):
        prediction = np.dot(X_test, self.weights) + self.bias
        return prediction

    def evaluate(self, X_test, y_test):
        y_pred = np.dot(X_test, self.weights) + self.bias
        return Round_Squared_Mean_Error(y_pred, y_test)

    def gradient_descent(self, X_train, y_train):
        n_samples = X_train.shape[0]
        y_pred = np.dot(X_train, self.weights) + self.bias
        dw = (1/n_samples) * np.dot(X_train.T, (y_pred - y_train))
        db = (1/n_samples) * np.sum(y_pred - y_train)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
