# model/gradients.py - Gradient computation
"""
Gradient computation for linear regression optimization.

Functions to implement:
- compute_gradients(X, y, weights, bias): Calculate partial derivatives
  * Compute gradients for weights: dJ/dw = (1/m) * X.T @ (y_pred - y)
  * Compute gradient for bias: dJ/db = (1/m) * sum(y_pred - y)
  * Handle both single and multiple features
  * Return gradients as dictionary or tuple
"""
import numpy as np

def compute_gradients(X, y, weights, bias):
    """
    Compute gradients for linear regression
    dW = -(2/n) * X.T @ (y - y_pred)
    db = -(2/n) * sum(y - y_pred)
    """
    n = len(y)
    y_pred = X @ weights + bias
    error = y - y_pred
    
    dW = -(2/n) * X.T @ error
    db = -(2/n) * np.sum(error)
    return dW, db
