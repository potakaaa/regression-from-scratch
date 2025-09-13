# model/parameters.py - Model parameter initialization
"""
Model parameter initialization for linear regression.

Functions to implement:
- initialize_weights(n_features): Initialize weights and bias
  * Random initialization with small values
  * Zero initialization option
  * Xavier/Glorot initialization for better convergence
  * Return weights array and bias scalar
"""

# model/parameters.py
import numpy as np

def initialize_weights(n_features):
    """Initialize weights and bias to zeros."""
    weights = np.zeros(n_features)
    bias = 0.0
    return weights, bias
