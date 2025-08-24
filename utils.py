import numpy as np
def Round_Squared_Mean_Error(y_true, y_pred):
    """
                    Calculate the Root Mean Squared Error (RMSE).

                    Args:
                        y_true (ndarray): True target values, shape (n_samples,).
                        y_pred (ndarray): Predicted values, shape (n_samples,).

                    Returns:
                        float: Evaluation metric (RMSE).
            """
    m = y_pred.shape[0]
    loss = (y_true - y_pred)**2
    cost = np.sum(loss) / (2*m)
    return cost

def Binary_Cross_Entropy(y_true, y_pred):
    return