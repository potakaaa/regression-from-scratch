# main.py - Run the pipeline
"""
Main entry point for the linear regression from scratch implementation.

This script orchestrates the entire pipeline:
1. Load data
2. Split into train/test sets
3. Initialize model parameters
4. Train the model using gradient descent
5. Evaluate performance
6. Visualize results (optional)

Functions to implement:
- main(): Orchestrate the entire pipeline
- run_pipeline(): Execute the complete workflow
"""

# main.py
import numpy as np
from data.loader import load_data
from utils.data_split import train_test_split
from model.parameters import initialize_weights
from model.predict import predict
from visualization.plot import plot_regression_line

# ----- 1. Data Preparation -----
print("=== Data Preparation ===")
# Fake dataset (instead of CSV for demo)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship y = 2x

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print("Train X:", X_train)
print("Test X:", X_test)

# ----- 2. Model Initialization -----
print("\n=== Model Initialization ===")
weights, bias = initialize_weights(n_features=X_train.shape[1])
print("Weights:", weights)
print("Bias:", bias)

# ----- 3. Prediction -----
print("\n=== Prediction ===")
# Set dummy weights for testing
weights = np.array([2.0])  # True slope
bias = 0.0
y_pred = predict(X_test, weights, bias)
print("Predictions:", y_pred)
print("Actual:", y_test)

# ----- 4. Visualization -----
print("\n=== Visualization ===")
plot_regression_line(X, y, predict(X, weights, bias))
