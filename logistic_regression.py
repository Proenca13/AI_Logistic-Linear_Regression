import numpy as np
import base_model
from utils import *
class LogisticRegression(base_model.BaseModel):

    def __init__(self, learning_rate = 0.01, iterations = 1000,lambda_ = 0.01):
        super().__init__(learning_rate, iterations,lambda_)

    def fit(self, X_train, y_train):
        self.weights = np.random.rand(X_train.shape[1])
        self.bias = 0
        return

    def predict(self, X_test):
        prediction = np.dot(X_test, self.weights) + self.bias
        return sigmoid(prediction)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test)

    def gradient_descent(self, X_train, y_train):
        return