# utils/data_split.py - Data splitting functionality
"""
Data splitting utilities for train/test/validation sets.

Functions to implement:
- train_test_split(X, y, test_size=0.2, seed=42): Split data into training and testing sets
  * Random shuffling with seed for reproducibility
  * Support for different test sizes
  * Return X_train, X_test, y_train, y_test
  * Optional validation split support
"""

# utils/data_split.py
import numpy as np

def train_test_split(X, y, test_size=0.2, seed=42):
    """Splits dataset into train and test sets."""
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_count = int(test_size * X.shape[0])
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
