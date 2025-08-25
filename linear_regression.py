import numpy as np
import base_model
from utils import Mean_Squared_Error,Root_Mean_Squared_Error
class LinearRegression(base_model.BaseModel):

    def __init__(self, learning_rate, iterations = 1000,lambda_ = 0.01):
        """
                Initialize LinearRegression model.

                Args:
                    learning_rate (float): Step size for gradient descent.
                    iterations (int): Number of gradient descent iterations. Default = 1000.
                    lambda_ (float): Regularization strength (L2). Default = 0.01.
        """
        super().__init__(learning_rate, iterations,lambda_)

    def fit(self, X_train, y_train):
        """
        Train the Linear Regression model using gradient descent.

        Args:
            X_train (ndarray): Training features, shape (n_samples, n_features).
            y_train (ndarray): Training targets, shape (n_samples,).
        """
        self.weights = np.random.rand(X_train.shape[1])  # shape (n_features,)
        self.bias = 0  # scalar

        self.cost_history = np.zeros(self.iterations, dtype=np.float64)
        tolerance = 1e-4

        for i in range(self.iterations):
            # Update weights and bias
            self.gradient_descent(X_train, y_train)

            # Compute cost
            y_pred = self.predict(X_train)
            self.cost_history[i] = Mean_Squared_Error(y_train, y_pred)

            # Early stopping check (skip i=0)
            if i > 0 and abs(self.cost_history[i - 1] - self.cost_history[i]) < tolerance:
                self.cost_history = self.cost_history[:i + 1]  # trim unused entries
                break

    def predict(self, X_test):
        """
                Predict target values for given input data.

                Args:
                    X_test (ndarray): Test features, shape (n_samples, n_features).

                Returns:
                    ndarray: Predicted values, shape (n_samples,).
        """
        prediction = np.dot(X_test, self.weights) + self.bias
        return prediction

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data and return MSE and RMSE.

        Args:
            X_test (ndarray): Test features, shape (n_samples, n_features).
            y_test (ndarray): True target values, shape (n_samples,).

        Returns:
            tuple: (MSE, RMSE)
        """
        y_pred = np.dot(X_test, self.weights) + self.bias
        mse = float(Mean_Squared_Error(y_test, y_pred))
        rmse = float(Root_Mean_Squared_Error(y_test, y_pred))
        return mse, rmse

    def gradient_descent(self, X_train, y_train):
            """
                    Perform one step of gradient descent and update weights and bias.

                    Args:
                        X_train (ndarray): Training features, shape (n_samples, n_features).
                        y_train (ndarray): Training targets, shape (n_samples,).
            """
            n_samples = X_train.shape[0]

            y_pred = np.dot(X_train, self.weights) + self.bias

            ## Calculate the derivates of w and b
            dw = (1/n_samples) * np.dot(X_train.T, (y_pred - y_train))
            db = (1/n_samples) * np.sum(y_pred - y_train)

            ## Update w and b
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
