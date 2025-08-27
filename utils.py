import numpy as np
def Mean_Squared_Error(y_test, y_pred,lambda_,weights):
    """
                    Calculate the Root Mean Squared Error (MSE).

                    Args:
                        y_test (ndarray): True target values, shape (n_samples,).
                        y_pred (ndarray): Predicted values, shape (n_samples,).
                        lambda_ (float, optional): Regularization strength (default is 0.0).
                        weights (ndarray, optional): Model weights for regularization, shape (n_features,).

                    Returns:
                        float: Evaluation metric (MSE).
            """
    m = y_pred.shape[0]
    loss = (y_test - y_pred)**2
    cost = np.sum(loss) / (2*m)
    regularization = (lambda_/(2*m)) * np.sum(weights**2)
    return cost + regularization

def Root_Mean_Squared_Error(y_test, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Args:
        y_test (ndarray): True target values, shape (n_samples,).
        y_pred (ndarray): Predicted values, shape (n_samples,).

    Returns:
        float: RMSE.
    """
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    return rmse

def Binary_Cross_Entropy(y_test, y_pred , lambda_,weights):
    return

def sigmoid(z):
    return 1/(1+np.exp(-z))