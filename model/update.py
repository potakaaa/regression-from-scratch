# model/update.py - Parameter updates
"""
Parameter update module for gradient descent optimization.

Functions to implement:
- update_weights(weights, bias, gradients, lr): Update model parameters
  * weights = weights - lr * dJ/dw
  * bias = bias - lr * dJ/db
  * Support for different learning rates
  * Optional momentum or other optimization techniques
  * Return updated weights and bias
"""

def update_weights(weights, bias, dW, db, lr):
    """
    Gradient descent parameter update
    """
    weights -= lr * dW
    bias -= lr * db
    return weights, bias
