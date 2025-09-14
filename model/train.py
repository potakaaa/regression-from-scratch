# model/train.py - Main training loop
"""
Main training module implementing the gradient descent algorithm.

Functions to implement:
- train(X, y, lr, epochs): Main training loop
  * Initialize weights and bias
  * For each epoch:
    * Make predictions
    * Compute loss
    * Calculate gradients
    * Update parameters
    * Track loss history
  * Early stopping support
  * Return trained weights, bias, and loss history
"""
import numpy as np
from model.predict import predict
from model.gradients import compute_gradients
from model.update import update_weights
from metrics.loss import mse

def train(X, y, lr=0.01, epochs=1000):
    """
    Training loop for linear regression
    """
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    history = []

    for epoch in range(epochs):
        y_pred = predict(X, weights, bias)
        dW, db = compute_gradients(X, y, weights, bias)
        weights, bias = update_weights(weights, bias, dW, db, lr)

        # Track loss
        loss = mse(y, y_pred)
        history.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights, bias, history
