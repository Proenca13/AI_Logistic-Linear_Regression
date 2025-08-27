import numpy as np
def Mean_Squared_Error(y_true, y_pred, lambda_=0.0, weights=None):
    """
    Calculate the Mean Squared Error (MSE) with optional L2 regularization.

    Args:
        y_true (ndarray): True target values, shape (n_samples,).
        y_pred (ndarray): Predicted values, shape (n_samples,).
        lambda_ (float, optional): Regularization strength (default is 0.0).
        weights (ndarray, optional): Model weights for regularization, shape (n_features,).

    Returns:
        float: Mean Squared Error (MSE) including regularization term if weights are provided.
    """
    m = y_pred.shape[0]
    loss = (y_true - y_pred) ** 2
    cost = np.sum(loss) / (2 * m)

    regularization = 0.0
    if weights is not None:
        regularization = (lambda_ / (2 * m)) * np.sum(weights ** 2)

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

def Binary_Cross_Entropy(y_true, y_pred, lambda_=0.0, weights=None):
    """
    Calculate Binary Cross-Entropy loss with optional L2 regularization.

    Args:
        y_true (ndarray): True binary labels, shape (n_samples,).
        y_pred (ndarray): Predicted probabilities, shape (n_samples,).
        lambda_ (float, optional): Regularization strength (default is 0.0).
        weights (ndarray, optional): Model weights for regularization, shape (n_features,).

    Returns:
        float: Binary cross-entropy loss (with regularization if weights provided).
    """

    # Avoid log(0) by clipping predictions
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Compute Binary Cross-Entropy
    loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    cost = -np.sum(loss) / y_true.shape[0]

    # Add L2 regularization (if weights provided)
    regularization = 0.0
    if weights is not None:
        regularization = (lambda_ / (2 * y_true.shape[0])) * np.sum(weights ** 2)

    return cost + regularization

def sigmoid(z):
    """
    Compute the sigmoid activation function.

    Args:
        z (float, int, or ndarray): Input value or array of values.

    Returns:
        float or ndarray: Sigmoid of the input, mapping values to the range (0, 1).
    """
    return 1 / (1 + np.exp(-z))
