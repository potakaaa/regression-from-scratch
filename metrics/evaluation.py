# metrics/evaluation.py - Model evaluation metrics
"""
Model evaluation metrics for linear regression performance assessment.

Functions to implement:
- r2_score(y_true, y_pred): R² (coefficient of determination)
  * R² = 1 - (SS_res / SS_tot)
  * Measures proportion of variance explained by model
  * Range: (-∞, 1], where 1 is perfect prediction
- nrmse(y_true, y_pred): Normalized Root Mean Squared Error
  * NRMSE = RMSE / (max(y) - min(y))
  * Scale-independent metric
  * Return normalized error value
"""
import numpy as np
from metrics.loss import rmse

def r2_score(y_true, y_pred):
    """
    R-squared: 1 - (SS_res / SS_tot)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def nrmse(y_true, y_pred):
    """
    Normalized RMSE (divided by range of y)
    """
    return rmse(y_true, y_pred) / (y_true.max() - y_true.min())
