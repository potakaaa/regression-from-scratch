# Linear Regression From Scratch

A complete implementation of linear regression (single and multiple variables) built entirely from scratch without using any machine learning libraries like scikit-learn.

## 📂 Project Structure

```
├── main.py              # Main entry point and pipeline orchestration
├── data/
│   ├── __init__.py
│   └── loader.py        # Data loading functionality
├── utils/
│   ├── __init__.py
│   ├── data_split.py    # Train/test splitting utilities
│   └── preprocessing.py # Feature normalization/scaling
├── model/
│   ├── __init__.py
│   ├── parameters.py    # Weight initialization
│   ├── predict.py       # Hypothesis function
│   ├── gradients.py     # Gradient computation
│   ├── update.py        # Parameter updates
│   └── train.py         # Main training loop
├── metrics/
│   ├── __init__.py
│   ├── loss.py          # MSE, RMSE calculations
│   └── evaluation.py    # R², NRMSE metrics
└── visualization/
    ├── __init__.py
    └── plot.py          # Training curves and regression plots
```

## 🚀 Implementation Pipeline

1. **Data Preparation**

   - Load data from CSV/text files
   - Split into training and testing sets
   - Optional feature normalization

2. **Model Setup**

   - Initialize weights and bias parameters
   - Define hypothesis function

3. **Training**

   - Implement gradient descent algorithm
   - Compute cost function (MSE)
   - Update parameters iteratively

4. **Evaluation**

   - Calculate performance metrics (R², RMSE)
   - Validate on test set

5. **Visualization** (Optional)
   - Plot training loss curves
   - Visualize regression line vs actual data

## 📋 Function Signatures

### Data Module

- `load_data(filepath)` → Load and return X, y
- `train_test_split(X, y, test_size=0.2, seed=42)` → Split data

### Utils Module

- `normalize(X)` → Scale features to [0,1] or standardize

### Model Module

- `initialize_weights(n_features)` → Return weights, bias
- `predict(X, weights, bias)` → Return predictions
- `compute_gradients(X, y, weights, bias)` → Return gradients
- `update_weights(weights, bias, gradients, lr)` → Update parameters
- `train(X, y, lr, epochs)` → Train model and return parameters

### Metrics Module

- `mse(y_true, y_pred)` → Mean Squared Error
- `rmse(y_true, y_pred)` → Root Mean Squared Error
- `r2_score(y_true, y_pred)` → R² coefficient
- `nrmse(y_true, y_pred)` → Normalized RMSE

### Visualization Module

- `plot_loss(history)` → Plot training loss curve
- `plot_regression_line(X, y, y_pred)` → Scatter plot with regression line

## 🎯 Key Features

- **No external ML libraries**: Pure Python/NumPy implementation
- **Modular design**: Each component in separate files
- **Educational focus**: Step-by-step implementation for learning
- **Multiple metrics**: Comprehensive evaluation suite
- **Visualization support**: Training progress and results plotting

## 🏃 Getting Started

1. Implement functions in each module (follow the comments in each file)
2. Run `python main.py` to execute the complete pipeline
3. Modify hyperparameters and observe results
4. Use visualization functions to understand model behavior

## 📊 Mathematical Foundation

The implementation follows the standard linear regression approach:

**Hypothesis**: `h(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`

**Cost Function**: `J(w,b) = (1/(2m)) Σ(h(xⁱ) - yⁱ)²`

**Gradient Descent**:

- `w := w - α * ∂J/∂w`
- `b := b - α * ∂J/∂b`

Where:

- `m` = number of training examples
- `α` = learning rate
- `w` = weight parameters
- `b` = bias term
