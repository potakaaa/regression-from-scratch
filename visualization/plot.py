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

def plot_predictions_vs_actual(y_true, y_pred):
    """Scatter plot of predictions vs actual values (works for any number of features)."""
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.show()

def plot_residuals(y_true, y_pred):
    """Histogram of residuals (y_true - y_pred)."""
    residuals = y_true - y_pred
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residuals Histogram")
    plt.show()

def plot_residuals_vs_predicted(y_true, y_pred):
    """Scatter of residuals vs predicted values to check homoscedasticity."""
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0.0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.show()
