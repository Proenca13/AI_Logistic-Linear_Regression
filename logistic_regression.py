import numpy as np
import base_model
from utils import *
class LogisticRegression(base_model.BaseModel):

    def __init__(self, learning_rate = 0.01, iterations = 1000,lambda_ = 0.01):
        super().__init__(learning_rate, iterations,lambda_)

    def fit(self, X_train, y_train):
        """
        Train the Logistic Regression model using gradient descent.

        Args:
            X_train (ndarray): Training features, shape (n_samples, n_features).
            y_train (ndarray): Training targets, shape (n_samples,).
        """
        self.weights = np.random.rand(X_train.shape[1])
        self.bias = 0
        self.cost_history = np.zeros(self.iterations, dtype=np.float64)
        tolerance = 1e-4

        for i in range(self.iterations):
            # Update weights and bias
            self.gradient_descent(X_train, y_train)

            # Compute cost
            y_pred = self.predict(X_train)
            self.cost_history[i] = Binary_Cross_Entropy(y_train, y_pred, lambda_=self.lambda_, weights=self.weights)

            # Early stopping check
            if i > 0 and abs(self.cost_history[i - 1] - self.cost_history[i]) < tolerance:
                self.cost_history = self.cost_history[:i + 1]  # trim unused entries
                break

    def predict(self, X_test):
        """
        Computes the predicted probabilities for the input data using the logistic (sigmoid) function.

        Args:
            X_test (ndarray): Input features of shape (n_samples, n_features).

        Returns:
            ndarray: Predicted probabilities of shape (n_samples,), values in range (0,1).
        """
        prediction = np.dot(X_test, self.weights) + self.bias
        return sigmoid(prediction)

    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluates the model's accuracy on the test dataset using a classification threshold.

        Args:
            X_test (ndarray): Input features of shape (n_samples, n_features).
            y_test (ndarray): True binary labels of shape (n_samples,).
            threshold (float): Decision threshold for classifying probabilities (default=0.5).

        Returns:
            float: Classification accuracy (correct predictions / total samples).
        """
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred > threshold).astype(int)
        return np.sum(y_pred_binary == y_test) / len(y_test)

    def gradient_descent(self, X_train, y_train):
        return