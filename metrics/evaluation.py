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
