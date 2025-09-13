# model/predict.py - Prediction (Hypothesis Function)
"""
Prediction module implementing the hypothesis function for linear regression.

Functions to implement:
- predict(X, weights, bias): Make predictions using linear model
  * Compute y_pred = X @ weights + bias (matrix multiplication)
  * Handle both single sample and batch predictions
  * Support for single and multiple features
  * Return predictions as numpy array or list
"""

# model/predict.py
import numpy as np

def predict(X, weights, bias):
    """Predict target values given features and parameters."""
    return np.dot(X, weights) + bias

