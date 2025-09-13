# visualization/plot.py - Data and model visualization
"""
Visualization module for plotting training results and model performance.

Functions to implement:
- plot_loss(history): Visualize training loss over epochs
  * Plot loss curve showing convergence
  * Support for training and validation loss
  * Add proper labels and title
- plot_regression_line(X, y, y_pred): Plot data points and regression line
  * For simple linear regression (single feature)
  * Scatter plot of actual vs predicted values
  * Overlay regression line
  * Add proper labels and title
"""

# visualization/plot.py
import matplotlib.pyplot as plt

def plot_regression_line(X, y, y_pred):
    """Plot scatter of actual values and regression line (only for 1 feature)."""
    if X.shape[1] != 1:
        raise ValueError("plot_regression_line works only for simple linear regression (1 feature).")
    
    plt.scatter(X, y, color="blue", label="Actual")
    plt.plot(X, y_pred, color="red", label="Prediction")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression Fit")
    plt.show()

def plot_loss(history):
    """Plot loss curve over epochs."""
    plt.plot(history, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()
