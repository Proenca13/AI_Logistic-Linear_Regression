import numpy as np
import base_model
class LogisticRegression(base_model.BaseModel):

    def __init__(self, learning_rate = 0.01, iterations = 1000,lambda_ = 0.01):
        super().__init__(learning_rate, iterations,lambda_)

    def fit(self, X_train, y_train):
        self.weights = np.random.rand(X_train.shape[1])
        self.bias = 0
        return
    def predict(self, X_test):
        return
    def evaluate(self, X_test, y_test):
        return
    def gradient_descent(self, X_train, y_train):
        return