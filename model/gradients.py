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
