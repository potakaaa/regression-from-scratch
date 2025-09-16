# Linear Regression Pipeline - Comprehensive Documentation

This document provides a detailed explanation of the complete linear regression pipeline as implemented in `main.py`. The pipeline demonstrates a full machine learning workflow from raw data to trained model with evaluation and visualization.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Step-by-Step Pipeline Breakdown](#step-by-step-pipeline-breakdown)
3. [Data Flow Diagram](#data-flow-diagram)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Code Implementation Details](#code-implementation-details)
6. [Pipeline Execution Flow](#pipeline-execution-flow)
7. [Outputs and Results](#outputs-and-results)

---

## Pipeline Overview

The linear regression pipeline implements a complete machine learning workflow with 8 distinct phases:

```
Raw CSV Data → Data Preparation → Model Training → Evaluation → Visualization
```

### **Pipeline Objectives:**

- **Predict**: Sleep Duration (hours) from lifestyle features
- **Learn**: Linear relationships between features and target
- **Evaluate**: Model performance using multiple metrics
- **Visualize**: Training progress and prediction quality

### **Key Features:**

- **End-to-end workflow**: From raw data to trained model
- **Modular design**: Each step uses dedicated functions
- **Comprehensive evaluation**: Multiple performance metrics
- **Rich visualization**: Training curves and prediction analysis
- **Error handling**: Graceful failure for visualization components

---

## Step-by-Step Pipeline Breakdown

### **Step 1: Data Preparation** (Lines 43-66)

#### **Purpose:**

Transform raw CSV data into machine learning-ready format with proper preprocessing.

#### **What Happens:**

1. **Load CSV Data** (Lines 48-54)
2. **Split into Train/Test Sets** (Line 57)
3. **Feature Standardization** (Lines 62-66)

#### **Detailed Code Analysis:**

```python
# 1.1 Load CSV Data
csv_path = "sleep_health_lifestyle_dataset.csv"
X, y = load_data(
    csv_path,
    target="Sleep Duration (hours)",
    features=["Daily Steps", "Age", "Occupation", "Physical Activity Level (minutes/day)"],
    delimiter=",",
    has_header=True,
)
```

**Explanation:**

- **`csv_path`**: Points to the sleep health dataset
- **`target`**: Column we want to predict (Sleep Duration)
- **`features`**: Input variables for prediction
- **`delimiter`**: CSV separator character
- **`has_header`**: First row contains column names

**Data Transformation:**

- **Raw CSV** → **Feature Matrix (X)** + **Target Vector (y)**
- **Categorical encoding**: Occupation becomes one-hot vectors
- **Type conversion**: All data becomes numerical

```python
# 1.2 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
```

**Explanation:**

- **`test_size=0.4`**: 40% of data for testing, 60% for training
- **Random shuffling**: Ensures unbiased data distribution
- **Maintains correspondence**: X_train[i] corresponds to y_train[i]

```python
# 1.3 Feature Standardization
x_mean = X_train.mean(axis=0)  # Calculate mean for each feature
x_std = X_train.std(axis=0)    # Calculate standard deviation for each feature
x_std[x_std == 0] = 1.0        # Prevent division by zero
X_train_s = (X_train - x_mean) / x_std  # Standardize training data
X_test_s = (X_test - x_mean) / x_std    # Standardize test data using training stats
```

**Mathematical Formula:**

```
z = (x - μ) / σ
```

Where:

- **z**: Standardized value
- **x**: Original value
- **μ**: Mean of the feature
- **σ**: Standard deviation of the feature

**Why Standardization:**

- **Equal feature importance**: Prevents features with larger scales from dominating
- **Faster convergence**: Gradient descent works better with normalized features
- **Numerical stability**: Prevents overflow/underflow issues

#### **Outputs:**

- **X_train_s**: Standardized training features
- **X_test_s**: Standardized test features
- **y_train**: Training targets
- **y_test**: Test targets
- **x_mean, x_std**: Scaling parameters (for later use)

---

### **Step 2: Model Initialization** (Lines 68-72)

#### **Purpose:**

Initialize the linear regression model parameters (weights and bias).

#### **What Happens:**

```python
weights, bias = initialize_weights(n_features=X_train_s.shape[1])
```

**Explanation:**

- **`X_train_s.shape[1]`**: Number of features after one-hot encoding
- **`weights`**: Vector of feature weights (one per feature)
- **`bias`**: Intercept term (scalar)

**Initialization Strategy:**

- **Zero initialization**: All weights start at 0
- **Simple approach**: Works well for linear regression
- **Deterministic**: Same starting point every time

**Mathematical Model:**

```
h(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

Where:

- **h(x)**: Predicted sleep duration
- **w₀**: Bias term (default when values are 0)
- **w₁...wₙ**: Feature weights
- **x₁...xₙ**: Feature values

#### **Outputs:**

- **weights**: Initial weight vector [0, 0, ..., 0]
- **bias**: Initial bias value 0.0

---

### **Step 3: Prediction Example** (Lines 74-81)

#### **Purpose:**

Demonstrate how the prediction function works with untrained parameters.

#### **What Happens:**

```python
demo_weights = np.zeros(X_test.shape[1])
demo_bias = 0.0
y_pred_demo = predict(X_test, demo_weights, demo_bias)
```

**Explanation:**

- **Demo parameters**: All zeros (untrained model)
- **Prediction**: Shows what untrained model predicts
- **Expected result**: All predictions should be 0.0 (since bias=0, weights=0)

**Mathematical Calculation:**

```
y_pred = X @ weights + bias
y_pred = X @ [0,0,0,...] + 0
y_pred = [0, 0, 0, ...]  # All predictions are zero
```

**Purpose of Demo:**

- **Verification**: Ensures prediction function works correctly
- **Baseline**: Shows untrained model performance
- **Debugging**: Helps identify issues early

#### **Outputs:**

- **y_pred_demo**: Vector of zero predictions
- **Console output**: Shows demo predictions vs actual values

---

### **Step 4: Gradient Computation Example** (Lines 83-89)

#### **Purpose:**

Demonstrate gradient calculation and parameter update mechanics.

#### **What Happens:**

```python
dW, db = compute_gradients(X_train_s, y_train, weights, bias)
upd_weights, upd_bias = update_weights(weights.copy(), float(bias), dW, db, lr=0.01)
```

**Mathematical Process:**

**4.1 Gradient Calculation:**

```python
# Inside compute_gradients()
y_pred = X @ weights + bias
error = y - y_pred
dW = -(2/n) * X.T @ error
db = -(2/n) * np.sum(error)
```

**Mathematical Formulas:**

```
∂J/∂w = -(2/m) × Xᵀ(y - y_pred)
∂J/∂b = -(2/m) × Σ(y - y_pred)
```

**4.2 Parameter Update:**

```python
# Inside update_weights()
weights = weights - lr * dW
bias = bias - lr * db
```

**Mathematical Formula:**

```
w_new = w_old - α × ∂J/∂w
b_new = b_old - α × ∂J/∂b
```

**Explanation:**

- **Gradients**: Point in direction of steepest increase in cost
- **Negative gradients**: Point toward cost reduction
- **Learning rate (α=0.01)**: Controls step size
- **One step**: Shows how parameters change in one iteration

#### **Outputs:**

- **dW, db**: Gradient values
- **upd_weights, upd_bias**: Parameters after one update step

---

### **Step 5: Model Training** (Lines 91-93)

#### **Purpose:**

Train the linear regression model using gradient descent optimization.

#### **What Happens:**

```python
trained_weights, trained_bias, history = train(X_train_s, y_train, lr=0.01, epochs=500)
```

**Training Process:**

1. **Initialize**: Start with zero weights and bias
2. **Iterate**: For 500 epochs:
   - Make predictions
   - Calculate loss
   - Compute gradients
   - Update parameters
   - Record loss history
3. **Return**: Trained parameters and loss history

**Mathematical Algorithm:**

```
For epoch in range(500):
    y_pred = X @ weights + bias
    loss = MSE(y, y_pred)
    dW, db = compute_gradients(X, y, weights, bias)
    weights = weights - lr * dW
    bias = bias - lr * db
    history.append(loss)
```

**Hyperparameters:**

- **Learning rate (lr=0.01)**: Step size for parameter updates
- **Epochs (500)**: Number of training iterations
- **Batch size**: Full dataset (batch gradient descent)

**Convergence Monitoring:**

- **Loss tracking**: Monitors training progress
- **Progress printing**: Shows loss every 100 epochs
- **Early stopping**: Could be added for efficiency

#### **Outputs:**

- **trained_weights**: Optimized weight vector
- **trained_bias**: Optimized bias value
- **history**: List of loss values for each epoch

---

### **Step 6: Training Loss Evaluation** (Lines 95-100)

#### **Purpose:**

Evaluate the trained model's performance on training data.

#### **What Happens:**

```python
y_pred_train = predict(X_train_s, trained_weights, trained_bias)
train_mse = mse(y_train, y_pred_train)
train_rmse = rmse(y_train, y_pred_train)
```

**Mathematical Calculations:**

**6.1 Predictions:**

```
y_pred_train = X_train_s @ trained_weights + trained_bias
```

**6.2 Mean Squared Error:**

```
MSE = (1/m) × Σ(y_train - y_pred_train)²
```

**6.3 Root Mean Squared Error:**

```
RMSE = √MSE
```

**Purpose:**

- **Training performance**: How well model fits training data
- **Baseline comparison**: Compare with test performance
- **Overfitting detection**: High training vs test performance gap

#### **Outputs:**

- **train_mse**: Training mean squared error
- **train_rmse**: Training root mean squared error
- **Console output**: Formatted performance metrics

---

### **Step 7: Model Evaluation** (Lines 102-107)

#### **Purpose:**

Evaluate the trained model's performance on unseen test data.

#### **What Happens:**

```python
y_pred_test = predict(X_test_s, trained_weights, trained_bias)
r2 = r2_score(y_test, y_pred_test)
test_nrmse = nrmse(y_test, y_pred_test)
```

**Mathematical Calculations:**

**7.1 Test Predictions:**

```
y_pred_test = X_test_s @ trained_weights + trained_bias
```

**7.2 R² Score:**

```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(y_test - y_pred_test)²
SS_tot = Σ(y_test - ȳ_test)²
```

**7.3 Normalized RMSE:**

```
NRMSE = RMSE / (max(y_test) - min(y_test))
```

**Evaluation Metrics Explained:**

**R² Score:**

- **Range**: -∞ to 1
- **R² = 1**: Perfect predictions
- **R² = 0**: Model performs as well as predicting the mean
- **R² < 0**: Model performs worse than predicting the mean

**NRMSE:**

- **Scale-independent**: Allows comparison across different datasets
- **Interpretation**: Percentage of target range that represents error
- **Lower is better**: Closer to 0 indicates better performance

#### **Outputs:**

- **r2**: Coefficient of determination
- **test_nrmse**: Normalized root mean squared error
- **Console output**: Formatted evaluation metrics

---

### **Step 8: Visualization** (Lines 109-128)

#### **Purpose:**

Create visual representations of training progress and model performance using matplotlib for comprehensive model analysis and debugging.

#### **Implementation Overview:**

The visualization step uses the **`visualization/plot.py`** module which contains 5 specialized plotting functions:

```python
from visualization.plot import (
    plot_regression_line,
    plot_loss,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_residuals_vs_predicted,
)
```

#### **Detailed Implementation:**

**8.1 Loss Curve Visualization:**

```python
try:
    plot_loss(history)
except Exception as e:
    print("plot_loss failed:", e)
```

**Module Used:** `matplotlib.pyplot`
**Function:** `plot_loss(history)`

**Implementation Details:**

```python
def plot_loss(history):
    """Plot loss curve over epochs."""
    plt.plot(history, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()
```

**What This Does:**

- **Input**: `history` - List of loss values from training (500 values for 500 epochs)
- **Creates**: Line plot showing loss decreasing over time
- **Purpose**:
  - **Training monitoring**: Verify loss is decreasing
  - **Convergence analysis**: Check if training has converged
  - **Hyperparameter tuning**: Identify optimal learning rate and epochs
  - **Overfitting detection**: Spot if loss starts increasing

**8.2 Feature-Dependent Visualization Logic:**

```python
X_s = (X - x_mean) / x_std
if X.shape[1] == 1:
    # Simple linear regression visualization
    plot_regression_line(X, y, predict(X_s, trained_weights, trained_bias))
else:
    # Multi-feature regression visualizations
    plot_predictions_vs_actual(y_test, y_pred_test)
    plot_residuals(y_test, y_pred_test)
    plot_residuals_vs_predicted(y_test, y_pred_test)
```

**Conditional Logic Explanation:**

- **`X.shape[1] == 1`**: Single feature regression (simple linear regression)
- **`X.shape[1] > 1`**: Multiple features regression (multiple linear regression)

**8.3 Single Feature Visualization:**

```python
plot_regression_line(X, y, predict(X_s, trained_weights, trained_bias))
```

**Module Used:** `matplotlib.pyplot`
**Function:** `plot_regression_line(X, y, y_pred)`

**Implementation Details:**

```python
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
```

**What This Creates:**

- **Blue dots**: Actual data points (X vs y)
- **Red line**: Predicted regression line
- **Purpose**: Visualize how well the linear model fits the data

**8.4 Multi-Feature Visualizations:**

**8.4.1 Predictions vs Actual Plot:**

```python
plot_predictions_vs_actual(y_test, y_pred_test)
```

**Module Used:** `matplotlib.pyplot`
**Function:** `plot_predictions_vs_actual(y_true, y_pred)`

**Implementation Details:**

```python
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
```

**What This Creates:**

- **Scatter plot**: Each point is (actual_value, predicted_value)
- **Red dashed line**: Perfect prediction line (y = x)
- **Purpose**:
  - **Model accuracy**: Points close to red line = good predictions
  - **Bias detection**: Systematic over/under-prediction
  - **Outlier identification**: Points far from the line

**8.4.2 Residuals Histogram:**

```python
plot_residuals(y_test, y_pred_test)
```

**Module Used:** `matplotlib.pyplot`
**Function:** `plot_residuals(y_true, y_pred)`

**Implementation Details:**

```python
def plot_residuals(y_true, y_pred):
    """Histogram of residuals (y_true - y_pred)."""
    residuals = y_true - y_pred
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residuals Histogram")
    plt.show()
```

**What This Creates:**

- **Histogram**: Distribution of prediction errors
- **X-axis**: Residual values (actual - predicted)
- **Y-axis**: Frequency of each residual value
- **Purpose**:
  - **Normality check**: Ideal residuals are normally distributed around 0
  - **Model assumptions**: Verify linear regression assumptions
  - **Error distribution**: Understand prediction error patterns

**8.4.3 Residuals vs Predicted Plot:**

```python
plot_residuals_vs_predicted(y_test, y_pred_test)
```

**Module Used:** `matplotlib.pyplot`
**Function:** `plot_residuals_vs_predicted(y_true, y_pred)`

**Implementation Details:**

```python
def plot_residuals_vs_predicted(y_true, y_pred):
    """Scatter of residuals vs predicted values to check homoscedasticity."""
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0.0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.show()
```

**What This Creates:**

- **Scatter plot**: Residuals vs predicted values
- **Red horizontal line**: Zero residual line
- **Purpose**:
  - **Homoscedasticity check**: Constant variance of residuals
  - **Model adequacy**: Random scatter = good model
  - **Pattern detection**: Curves or trends indicate model problems

#### **Error Handling Implementation:**

**Robust Error Handling:**

```python
try:
    plot_loss(history)
except Exception as e:
    print("plot_loss failed:", e)
```

**Why This Matters:**

- **Graceful degradation**: Pipeline continues even if visualization fails
- **Debug information**: Error messages help identify issues
- **Production readiness**: Prevents crashes in different environments
- **Cross-platform compatibility**: Handles different matplotlib backends

**Common Failure Scenarios:**

- **Display issues**: No GUI available (headless servers)
- **Backend problems**: Matplotlib backend not configured
- **Memory issues**: Large datasets causing memory problems
- **Permission issues**: Cannot create display windows

#### **Data Flow in Visualization:**

```
history (from training) → plot_loss() → Loss curve plot
y_test, y_pred_test → plot_predictions_vs_actual() → Scatter plot
y_test, y_pred_test → plot_residuals() → Histogram
y_test, y_pred_test → plot_residuals_vs_predicted() → Residuals plot
X, y, predictions → plot_regression_line() → Regression line (single feature only)
```

#### **Mathematical Concepts Visualized:**

**1. Loss Convergence:**

- **Mathematical**: J(θ) decreasing over epochs
- **Visual**: Downward trending line
- **Interpretation**: Model is learning

**2. Model Fit Quality:**

- **Mathematical**: R² = 1 - (SS_res / SS_tot)
- **Visual**: Points close to y=x line
- **Interpretation**: High R² = good fit

**3. Residual Analysis:**

- **Mathematical**: e = y - ŷ
- **Visual**: Normally distributed around zero
- **Interpretation**: Good model assumptions

**4. Homoscedasticity:**

- **Mathematical**: Var(e) = constant
- **Visual**: Random scatter around zero line
- **Interpretation**: Constant error variance

#### **Outputs:**

**Visual Outputs:**

- **Loss curve plot**: Training convergence visualization
- **Regression line plot**: Data and fitted line (single feature)
- **Prediction scatter plot**: Actual vs predicted values
- **Residuals histogram**: Error distribution
- **Residuals vs predicted plot**: Homoscedasticity check

**Console Outputs:**

- **Error messages**: Visualization failure notifications
- **Debug information**: Troubleshooting details

#### **Performance Considerations:**

**Memory Efficiency:**

- **Lazy loading**: Plots created only when needed
- **Efficient data structures**: NumPy arrays for calculations
- **Minimal data copying**: Direct array operations

**Display Optimization:**

- **Appropriate bin sizes**: 30 bins for histograms
- **Alpha transparency**: 0.6-0.7 for scatter plots
- **Clear labels**: Descriptive titles and axis labels

---

## Data Flow Diagram

```
Raw CSV Data
    ↓
[load_data()] → Feature Matrix (X) + Target Vector (y)
    ↓
[train_test_split()] → X_train, X_test, y_train, y_test
    ↓
[Standardization] → X_train_s, X_test_s (scaled features)
    ↓
[initialize_weights()] → weights, bias (initial parameters)
    ↓
[train()] → trained_weights, trained_bias, history
    ↓
[predict()] → y_pred_train, y_pred_test
    ↓
[mse(), rmse(), r2_score(), nrmse()] → Performance Metrics
    ↓
[plot_*()] → Visualizations
```

---

## Mathematical Foundations

### **Linear Regression Model:**

```
h(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

### **Cost Function (MSE):**

```
J(w,b) = (1/2m) × Σ(h(x) - y)²
```

### **Gradient Descent:**

```
w = w - α × ∂J/∂w
b = b - α × ∂J/∂b
```

### **Gradients:**

```
∂J/∂w = (1/m) × Xᵀ(h(x) - y)
∂J/∂b = (1/m) × Σ(h(x) - y)
```

---

## Multiple Linear Regression Formula Application

### **Mathematical Model Overview:**

The multiple linear regression model extends simple linear regression to handle multiple input features:

```
h(x) = w₀ + w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ
```

**Where:**

- **h(x)**: Predicted value (hypothesis)
- **w₀**: Bias term (intercept)
- **w₁, w₂, ..., wₙ**: Feature weights (coefficients)
- **x₁, x₂, ..., xₙ**: Input features
- **n**: Number of features

### **Matrix Form Representation:**

In matrix notation, the model becomes:

```
h(X) = Xw + b
```

**Where:**

- **X**: Feature matrix (m × n) - m samples, n features
- **w**: Weight vector (n × 1)
- **b**: Bias scalar
- **h(X)**: Prediction vector (m × 1)

### **Real-World Application in Our Sleep Health Pipeline:**

Based on our sleep health dataset, the model becomes:

```
Sleep_Duration = w₀ + w₁×Daily_Steps + w₂×Age + w₃×Occupation_Engineer + w₄×Occupation_Teacher + w₅×Occupation_Doctor + w₆×Physical_Activity
```

**Feature Breakdown:**

- **w₀**: Base sleep duration (intercept)
- **w₁**: Effect of daily steps on sleep
- **w₂**: Effect of age on sleep
- **w₃, w₄, w₅**: Effects of different occupations (one-hot encoded)
- **w₆**: Effect of physical activity on sleep

### **Mathematical Example with Our Data:**

**Sample Data:**

```
Person 1: [8000 steps, 25 age, Engineer, 30 min activity] → 7.5 hours sleep
Person 2: [12000 steps, 30 age, Teacher, 45 min activity] → 8.0 hours sleep
```

**Feature Matrix (after one-hot encoding):**

```
X = [[8000, 25, 1, 0, 0, 30],    # Engineer
     [12000, 30, 0, 1, 0, 45]]   # Teacher
```

**Target Vector:**

```
y = [7.5, 8.0]
```

**Prediction Calculation:**

```
h(x⁽¹⁾) = w₀ + w₁×8000 + w₂×25 + w₃×1 + w₄×0 + w₅×0 + w₆×30
h(x⁽²⁾) = w₀ + w₁×12000 + w₂×30 + w₃×0 + w₄×1 + w₅×0 + w₆×45
```

**Cost Calculation:**

```
J = (1/4) × [(h(x⁽¹⁾) - 7.5)² + (h(x⁽²⁾) - 8.0)²]
```

**Gradient Calculation:**

```
∂J/∂w₁ = (1/2) × [(h(x⁽¹⁾) - 7.5)×8000 + (h(x⁽²⁾) - 8.0)×12000]
∂J/∂w₂ = (1/2) × [(h(x⁽¹⁾) - 7.5)×25 + (h(x⁽²⁾) - 8.0)×30]
...
```

### **Feature Scaling Impact:**

**Before Standardization:**

```
X = [[8000, 25, 1, 0, 0, 30],
     [12000, 30, 0, 1, 0, 45]]
```

**After Standardization:**

```
X_std = [[0.0, -0.5, 1.0, 0.0, 0.0, -0.3],
         [1.0, 0.5, 0.0, 1.0, 0.0, 0.3]]
```

**Why This Matters:**

- **Equal feature importance**: All features contribute equally to gradient updates
- **Faster convergence**: Gradient descent converges more quickly
- **Numerical stability**: Prevents overflow/underflow issues

### **Model Interpretation:**

**Trained Model Example:**

```
Sleep_Duration = 7.2 + 0.0001×Daily_Steps + 0.05×Age + 0.3×Engineer + 0.1×Teacher + 0.2×Doctor + 0.02×Physical_Activity
```

**Interpretation:**

- **Base sleep**: 7.2 hours (intercept)
- **Daily steps**: +0.0001 hours per step (minimal effect)
- **Age**: +0.05 hours per year of age
- **Occupation effects**: Engineers sleep 0.3 hours more, Teachers 0.1 hours more, Doctors 0.2 hours more
- **Physical activity**: +0.02 hours per minute of activity

### **Complete Gradient Descent Algorithm:**

```
1. Initialize w = [0, 0, ..., 0], b = 0
2. For each epoch:
   a. Compute predictions: h(X) = Xw + b
   b. Compute cost: J = (1/2m) × ||h(X) - y||²
   c. Compute gradients:
      - ∂J/∂w = (1/m) × Xᵀ(h(X) - y)
      - ∂J/∂b = (1/m) × Σ(h(X) - y)
   d. Update parameters:
      - w = w - α × ∂J/∂w
      - b = b - α × ∂J/∂b
3. Return optimized w and b
```

### **Evaluation Metrics:**

**R² Score:**

```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(y - ŷ)²
SS_tot = Σ(y - ȳ)²
```

**Root Mean Squared Error:**

```
RMSE = √[(1/m) × Σ(y - ŷ)²]
```

**Normalized RMSE:**

```
NRMSE = RMSE / (y_max - y_min)
```

---

## Code Implementation Details

### **Key Design Patterns:**

**1. Modular Architecture:**

- Each function has a single responsibility
- Clear separation of concerns
- Easy to test and maintain

**2. Error Handling:**

- Try-catch blocks for visualization
- Graceful failure handling
- Informative error messages

**3. Data Consistency:**

- Proper train/test split
- Consistent scaling parameters
- Maintained data correspondence

**4. Reproducibility:**

- Fixed random seeds
- Deterministic initialization
- Consistent preprocessing

### **Performance Considerations:**

**1. Memory Efficiency:**

- In-place operations where possible
- Efficient numpy operations
- Minimal data copying

**2. Numerical Stability:**

- Feature standardization
- Division by zero protection
- Appropriate data types

**3. Scalability:**

- Vectorized operations
- Batch processing
- Efficient matrix operations

---

## Pipeline Execution Flow

### **Execution Order:**

1. **Data Loading** → Raw data to structured format
2. **Data Splitting** → Train/test separation
3. **Preprocessing** → Feature standardization
4. **Model Setup** → Parameter initialization
5. **Training** → Gradient descent optimization
6. **Evaluation** → Performance assessment
7. **Visualization** → Results presentation

### **Dependencies:**

- **Step 2** depends on **Step 1** (data preparation)
- **Step 3** depends on **Step 2** (model initialization)
- **Step 4** depends on **Step 2** (model parameters)
- **Step 5** depends on **Step 1** (training data)
- **Step 6** depends on **Step 5** (trained model)
- **Step 7** depends on **Step 5** (trained model)
- **Step 8** depends on **Step 5** (training history)

---

## Outputs and Results

### **Return Dictionary:**

```python
return {
    "X_train": X_train,           # Original training features
    "X_test": X_test,             # Original test features
    "y_train": y_train,           # Training targets
    "y_test": y_test,             # Test targets
    "weights": trained_weights,   # Final model weights
    "bias": trained_bias,         # Final model bias
    "history": history,           # Training loss history
    "y_pred_test": y_pred_test,   # Test predictions
}
```

### **Console Outputs:**

- **Data shapes**: Training and test set sizes
- **Initial parameters**: Starting weights and bias
- **Demo predictions**: Untrained model output
- **Gradient values**: First iteration gradients
- **Training progress**: Loss every 100 epochs
- **Performance metrics**: MSE, RMSE, R², NRMSE
- **Error messages**: Visualization failures

### **Visual Outputs:**

- **Loss curve**: Training convergence plot
- **Regression line**: Data and fitted line (single feature)
- **Prediction scatter**: Actual vs predicted values
- **Residual plots**: Error analysis visualizations

---

## Summary

This pipeline demonstrates a complete machine learning workflow from raw data to trained model. It showcases:

- **Data preprocessing** with proper train/test splitting and standardization
- **Model training** using gradient descent optimization
- **Comprehensive evaluation** with multiple performance metrics
- **Rich visualization** for understanding model behavior
- **Robust error handling** for production readiness

The pipeline serves as both a practical implementation and an educational tool for understanding linear regression from first principles.
