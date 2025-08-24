from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, learning_rate, iterations,lambda_):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_ = lambda_

    @abstractmethod
    def fit(self,X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass
