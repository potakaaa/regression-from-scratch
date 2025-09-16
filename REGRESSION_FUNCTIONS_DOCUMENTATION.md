# Linear Regression from Scratch - Comprehensive Function Documentation

This document provides a detailed explanation of each function implemented in the linear regression from scratch project. The implementation follows a modular approach with clear separation of concerns across different areas of the machine learning pipeline.

## Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Model Initialization](#2-model-initialization)
3. [Prediction (Hypothesis Function)](#3-prediction-hypothesis-function)
4. [Cost Function (Loss)](#4-cost-function-loss)
5. [Optimization (Training)](#5-optimization-training)
6. [Model Evaluation](#6-model-evaluation)
7. [Visualization](#7-visualization)

---

## 1. Data Preparation

### 📂 `data/loader.py`

#### `load_data(filepath, *, target, features=None, delimiter=",", has_header=True)`

**Purpose**: Loads CSV data and returns feature matrix (X) and target vector (y) as numpy arrays.

**Mathematical Background**:

- Converts raw CSV data into numerical format suitable for machine learning
- Handles both numerical and categorical features through one-hot encoding
- Ensures data consistency and type safety

**Parameters**:

- `filepath` (str): Path to the CSV file
- `target` (Union[str, int]): Target column name or index
- `features` (Optional[Iterable[Union[str, int]]]): Feature columns to select (None = all except target)
- `delimiter` (str): CSV delimiter character
- `has_header` (bool): Whether the CSV has a header row

**Returns**: `Tuple[np.ndarray, np.ndarray]` - (X, y) where X is feature matrix and y is target vector

**Implementation Details**:

- Uses two-pass algorithm: first pass collects categorical categories, second pass builds feature vectors
- Automatically detects categorical vs numerical columns
- Performs one-hot encoding for categorical features
- Skips rows with invalid data (non-numeric target values)
- Ensures 2D feature matrix format

**Usage Example**:

```python
X, y = load_data(
    "dataset.csv",
    target="Sleep Duration (hours)",
    features=["Daily Steps", "Age", "Occupation"],
    delimiter=",",
    has_header=True
)
```

---

### 📂 `utils/data_split.py`

#### `train_test_split(X, y, test_size=0.2, seed=42)`

**Purpose**: Splits dataset into training and testing sets with random shuffling.

**Mathematical Background**:

- Ensures unbiased evaluation by randomly splitting data
- Maintains data distribution across splits
- Provides reproducible results through seed control

**Parameters**:

- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Target vector
- `test_size` (float): Proportion of data for testing (0.0 to 1.0)
- `seed` (int): Random seed for reproducibility

**Returns**: `Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` - (X_train, X_test, y_train, y_test)

**Implementation Details**:

- Creates random permutation of indices
- Splits based on test_size proportion
- Maintains correspondence between X and y

**Usage Example**:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=42)
```

---

### 📂 `utils/preprocessing.py`

#### `normalize(X)` - _Not Yet Implemented_

**Purpose**: Normalize/scale features using standardization or min-max scaling.

**Mathematical Background**:

- **Standardization**: z = (x - μ) / σ (zero mean, unit variance)
- **Min-Max Scaling**: x' = (x - min) / (max - min) (range [0,1])

**Note**: This function is planned but not yet implemented in the current codebase.

---

## 2. Model Initialization

### 📂 `model/parameters.py`

#### `initialize_weights(n_features)`

**Purpose**: Initialize model parameters (weights and bias) for linear regression.

**Mathematical Background**:

- Linear regression hypothesis: h(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
- Where w₀ is bias (intercept) and w₁...wₙ are feature weights

**Parameters**:

- `n_features` (int): Number of input features

**Returns**: `Tuple[np.ndarray, float]` - (weights, bias)

**Implementation Details**:

- Initializes weights as zero vector: w = [0, 0, ..., 0]
- Initializes bias as zero: b = 0
- Simple initialization suitable for linear regression

**Usage Example**:

```python
weights, bias = initialize_weights(n_features=3)
# weights = [0.0, 0.0, 0.0], bias = 0.0
```

---

## 3. Prediction (Hypothesis Function)

### 📂 `model/predict.py`

#### `predict(X, weights, bias)`

**Purpose**: Makes predictions using the linear regression model.

**Mathematical Background**:

- **Hypothesis Function**: h(x) = Xw + b
- Where X is feature matrix, w is weight vector, b is bias scalar
- Matrix multiplication: Xw computes weighted sum of features

**Parameters**:

- `X` (np.ndarray): Feature matrix (m × n) where m = samples, n = features
- `weights` (np.ndarray): Weight vector (n × 1)
- `bias` (float): Bias term (scalar)

**Returns**: `np.ndarray` - Predicted values (m × 1)

**Implementation Details**:

- Uses numpy dot product for efficient matrix multiplication
- Broadcasting handles bias addition across all samples
- Supports both single samples and batch predictions

**Mathematical Formula**:

```
y_pred = X @ weights + bias
```

**Usage Example**:

```python
y_pred = predict(X_test, weights, bias)
```

---

## 4. Cost Function (Loss)

### 📂 `metrics/loss.py`

#### `mse(y_true, y_pred)`

**Purpose**: Computes Mean Squared Error loss function.

**Mathematical Background**:

- **MSE Formula**: MSE = (1/m) × Σ(y_true - y_pred)²
- Measures average squared difference between actual and predicted values
- Penalizes larger errors more heavily (quadratic penalty)

**Parameters**:

- `y_true` (np.ndarray): Actual target values
- `y_pred` (np.ndarray): Predicted values

**Returns**: `float` - Mean squared error

**Mathematical Formula**:

```
MSE = (1/m) × Σᵢ₌₁ᵐ (y_trueᵢ - y_predᵢ)²
```

**Usage Example**:

```python
loss = mse(y_train, y_pred_train)
```

---

#### `rmse(y_true, y_pred)`

**Purpose**: Computes Root Mean Squared Error.

**Mathematical Background**:

- **RMSE Formula**: RMSE = √MSE
- Same units as target variable (unlike MSE)
- More interpretable than MSE for reporting

**Parameters**:

- `y_true` (np.ndarray): Actual target values
- `y_pred` (np.ndarray): Predicted values

**Returns**: `float` - Root mean squared error

**Mathematical Formula**:

```
RMSE = √[(1/m) × Σᵢ₌₁ᵐ (y_trueᵢ - y_predᵢ)²]
```

**Usage Example**:

```python
error = rmse(y_test, y_pred_test)
```

---

## 5. Optimization (Training)

### 📂 `model/gradients.py`

#### `compute_gradients(X, y, weights, bias)`

**Purpose**: Computes gradients of the cost function with respect to parameters.

**Mathematical Background**:

- **Gradient Descent**: Updates parameters in direction of steepest descent
- **Weight Gradient**: ∂J/∂w = -(2/m) × Xᵀ(y - y_pred)
- **Bias Gradient**: ∂J/∂b = -(2/m) × Σ(y - y_pred)

**Parameters**:

- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Actual target values
- `weights` (np.ndarray): Current weight vector
- `bias` (float): Current bias value

**Returns**: `Tuple[np.ndarray, float]` - (dW, db) gradients

**Mathematical Formulas**:

```
dW = -(2/m) × Xᵀ(y - y_pred)
db = -(2/m) × Σ(y - y_pred)
```

**Implementation Details**:

- Computes prediction first: y_pred = X @ weights + bias
- Calculates error: error = y - y_pred
- Uses matrix transpose for efficient gradient computation

**Usage Example**:

```python
dW, db = compute_gradients(X_train, y_train, weights, bias)
```

---

### 📂 `model/update.py`

#### `update_weights(weights, bias, dW, db, lr)`

**Purpose**: Updates model parameters using gradient descent.

**Mathematical Background**:

- **Gradient Descent Update Rule**:
  - w = w - α × ∂J/∂w
  - b = b - α × ∂J/∂b
- Where α (alpha) is the learning rate

**Parameters**:

- `weights` (np.ndarray): Current weight vector
- `bias` (float): Current bias value
- `dW` (np.ndarray): Weight gradients
- `db` (float): Bias gradient
- `lr` (float): Learning rate

**Returns**: `Tuple[np.ndarray, float]` - Updated (weights, bias)

**Mathematical Formulas**:

```
w_new = w_old - lr × dW
b_new = b_old - lr × db
```

**Usage Example**:

```python
weights, bias = update_weights(weights, bias, dW, db, lr=0.01)
```

---

### 📂 `model/train.py`

#### `train(X, y, lr=0.01, epochs=1000)`

**Purpose**: Main training loop implementing gradient descent algorithm.

**Mathematical Background**:

- **Gradient Descent Algorithm**:
  1. Initialize parameters
  2. For each epoch:
     - Compute predictions
     - Calculate loss
     - Compute gradients
     - Update parameters
  3. Return trained parameters

**Parameters**:

- `X` (np.ndarray): Training feature matrix
- `y` (np.ndarray): Training target values
- `lr` (float): Learning rate (default: 0.01)
- `epochs` (int): Number of training iterations (default: 1000)

**Returns**: `Tuple[np.ndarray, float, List[float]]` - (trained_weights, trained_bias, loss_history)

**Implementation Details**:

- Initializes weights and bias to zeros
- Tracks loss history for monitoring convergence
- Prints progress every 100 epochs
- Returns final parameters and training history

**Usage Example**:

```python
weights, bias, history = train(X_train, y_train, lr=0.01, epochs=500)
```

---

## 6. Model Evaluation

### 📂 `metrics/evaluation.py`

#### `r2_score(y_true, y_pred)`

**Purpose**: Computes R² (coefficient of determination) metric.

**Mathematical Background**:

- **R² Formula**: R² = 1 - (SS_res / SS_tot)
- **SS_res** (Residual Sum of Squares): Σ(y_true - y_pred)²
- **SS_tot** (Total Sum of Squares): Σ(y_true - ȳ)²
- Measures proportion of variance explained by the model

**Parameters**:

- `y_true` (np.ndarray): Actual target values
- `y_pred` (np.ndarray): Predicted values

**Returns**: `float` - R² score (range: -∞ to 1, where 1 is perfect)

**Mathematical Formula**:

```
R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - ȳ)²)
```

**Interpretation**:

- R² = 1: Perfect predictions
- R² = 0: Model performs as well as predicting the mean
- R² < 0: Model performs worse than predicting the mean

**Usage Example**:

```python
score = r2_score(y_test, y_pred_test)
```

---

#### `nrmse(y_true, y_pred)`

**Purpose**: Computes Normalized Root Mean Squared Error.

**Mathematical Background**:

- **NRMSE Formula**: NRMSE = RMSE / (y_max - y_min)
- Scale-independent metric for comparing models across different datasets
- Normalizes by the range of target values

**Parameters**:

- `y_true` (np.ndarray): Actual target values
- `y_pred` (np.ndarray): Predicted values

**Returns**: `float` - Normalized RMSE

**Mathematical Formula**:

```
NRMSE = RMSE / (max(y_true) - min(y_true))
```

**Usage Example**:

```python
nrmse_score = nrmse(y_test, y_pred_test)
```

---

## 7. Visualization

### 📂 `visualization/plot.py`

#### `plot_loss(history)`

**Purpose**: Visualizes training loss curve over epochs.

**Mathematical Background**:

- Plots loss values to monitor training convergence
- Helps identify overfitting, underfitting, or convergence issues
- Essential for hyperparameter tuning

**Parameters**:

- `history` (List[float]): List of loss values from training

**Returns**: `None` (displays plot)

**Usage Example**:

```python
plot_loss(history)
```

---

#### `plot_regression_line(X, y, y_pred)`

**Purpose**: Plots data points and regression line for simple linear regression.

**Mathematical Background**:

- Visualizes the linear relationship between single feature and target
- Shows how well the model fits the data
- Only works for single-feature regression

**Parameters**:

- `X` (np.ndarray): Feature matrix (must be single feature)
- `y` (np.ndarray): Actual target values
- `y_pred` (np.ndarray): Predicted values

**Returns**: `None` (displays plot)

**Usage Example**:

```python
plot_regression_line(X, y, y_pred)
```

---

#### `plot_predictions_vs_actual(y_true, y_pred)`

**Purpose**: Scatter plot comparing predicted vs actual values.

**Mathematical Background**:

- Perfect predictions would lie on the diagonal line y = x
- Helps assess model performance across different value ranges
- Works for any number of features

**Parameters**:

- `y_true` (np.ndarray): Actual target values
- `y_pred` (np.ndarray): Predicted values

**Returns**: `None` (displays plot)

**Usage Example**:

```python
plot_predictions_vs_actual(y_test, y_pred_test)
```

---

#### `plot_residuals(y_true, y_pred)`

**Purpose**: Histogram of prediction residuals.

**Mathematical Background**:

- **Residuals**: e = y_true - y_pred
- Ideal residuals should be normally distributed around zero
- Helps check model assumptions

**Parameters**:

- `y_true` (np.ndarray): Actual target values
- `y_pred` (np.ndarray): Predicted values

**Returns**: `None` (displays plot)

**Usage Example**:

```python
plot_residuals(y_test, y_pred_test)
```

---

#### `plot_residuals_vs_predicted(y_true, y_pred)`

**Purpose**: Scatter plot of residuals vs predicted values.

**Mathematical Background**:

- Checks for homoscedasticity (constant variance of residuals)
- Random scatter around zero indicates good model fit
- Patterns suggest model inadequacy

**Parameters**:

- `y_true` (np.ndarray): Actual target values
- `y_pred` (np.ndarray): Predicted values

**Returns**: `None` (displays plot)

**Usage Example**:

```python
plot_residuals_vs_predicted(y_test, y_pred_test)
```

---

## Complete Pipeline Example

Here's how all functions work together in the complete pipeline:

```python
# 1. Data Preparation
X, y = load_data("dataset.csv", target="target_column", features=["feature1", "feature2"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Model Initialization
weights, bias = initialize_weights(n_features=X_train.shape[1])

# 3. Training
trained_weights, trained_bias, history = train(X_train, y_train, lr=0.01, epochs=1000)

# 4. Prediction
y_pred = predict(X_test, trained_weights, trained_bias)

# 5. Evaluation
mse_score = mse(y_test, y_pred)
rmse_score = rmse(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
nrmse_score = nrmse(y_test, y_pred)

# 6. Visualization
plot_loss(history)
plot_predictions_vs_actual(y_test, y_pred)
```

---

## Mathematical Summary

The linear regression model implements the following mathematical concepts:

1. **Hypothesis**: h(x) = Xw + b
2. **Cost Function**: J(w,b) = (1/2m) × Σ(y_pred - y_true)²
3. **Gradients**:
   - ∂J/∂w = (1/m) × Xᵀ(y_pred - y_true)
   - ∂J/∂b = (1/m) × Σ(y_pred - y_true)
4. **Update Rule**:
   - w = w - α × ∂J/∂w
   - b = b - α × ∂J/∂b

This implementation provides a complete, educational framework for understanding linear regression from first principles.
