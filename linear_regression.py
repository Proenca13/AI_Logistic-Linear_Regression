import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base_model
class LinearRegression(base_model.BaseModel):

    def __init__(self, learning_rate, iterations,lambda_):
        super().__init__(learning_rate, iterations,lambda_)

    def fit(self, X_train, y_train):
        return

        return
    def predict(self, X_test):
        return
    def evaluate(self, X_test, y_test):
        return