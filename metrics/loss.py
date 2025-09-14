# metrics/loss.py - Loss functions
"""
Loss function implementations for linear regression.

Functions to implement:
- mse(y_true, y_pred): Mean Squared Error
  * J = (1/(2*m)) * sum((y_pred - y_true)^2)
  * Return scalar loss value
- rmse(y_true, y_pred): Root Mean Squared Error
  * RMSE = sqrt(MSE)
  * Return scalar loss value in same units as target
"""


import numpy as np

def mse(y_true, y_pred):
    """
    Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mse(y_true, y_pred))

