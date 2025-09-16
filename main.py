"""
Main entry point for the linear regression from scratch implementation.

This script orchestrates the entire pipeline and provides example calls for
all atomic functions across the project.

Pipeline:
1. Load data (example via CSV API; uses synthetic data for demo)
2. Split into train/test sets
3. Initialize model parameters
4. Make predictions
5. Compute gradients and update parameters (examples)
6. Train the model using gradient descent
7. Evaluate performance (loss, RMSE, R2, NRMSE)
8. Visualize results (loss curve, regression line)

Exposed functions:
- main(): Orchestrates the entire pipeline with examples
- run_pipeline(): Executes the workflow and returns artifacts
"""

import numpy as np
from data.loader import load_data
from utils.data_split import train_test_split
from model.parameters import initialize_weights
from model.predict import predict
from model.gradients import compute_gradients
from model.update import update_weights
from model.train import train
from metrics.loss import mse, rmse
from metrics.evaluation import r2_score, nrmse
from visualization.plot import (
    plot_regression_line,
    plot_loss,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_residuals_vs_predicted,
)


def run_pipeline():
    """Run the full workflow and return results for downstream usage."""
    # ----- 1. Data Preparation -----
    print("=== Data Preparation ===")
    # Use provided CSV dataset: predict Sleep Duration (hours) from Daily Steps
    # You can swap features/target by column name or index.
    csv_path = "sleep_health_lifestyle_dataset.csv"
    X, y = load_data(
        csv_path,
        target="Sleep Duration (hours)",
        features=["Daily Steps", "Age", "Occupation", "Physical Activity Level (minutes/day)"],
        delimiter=",",
        has_header=True,
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print("Train X:", X_train)
    print("Test X:", X_test)

    # Scale feature(s) to stabilize training (standardization)
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    X_train_s = (X_train - x_mean) / x_std
    X_test_s = (X_test - x_mean) / x_std

    # ----- 2. Model Initialization -----
    print("\n=== Model Initialization ===")
    weights, bias = initialize_weights(n_features=X_train_s.shape[1])
    print("Initial Weights:", weights)
    print("Initial Bias:", bias)

    # ----- 3. Prediction (example) -----
    print("\n=== Prediction (Example) ===")
    # Align demo weights with current feature dimension (may include one-hot columns)
    demo_weights = np.zeros(X_test.shape[1])
    demo_bias = 0.0
    y_pred_demo = predict(X_test, demo_weights, demo_bias)
    print("Demo Predictions:", y_pred_demo)
    print("Actual:", y_test)

    # ----- 4. Gradients and Update (examples) -----
    print("\n=== Gradients and Update (Examples) ===")
    dW, db = compute_gradients(X_train_s, y_train, weights, bias)
    print("Gradients dW:", dW, "db:", db)
    upd_weights, upd_bias = update_weights(weights.copy(), float(bias), dW, db, lr=0.01)
    print("Updated Weights (one step):", upd_weights)
    print("Updated Bias (one step):", upd_bias)

    # ----- 5. Training -----
    print("\n=== Training ===")
    trained_weights, trained_bias, history = train(X_train_s, y_train, lr=0.01, epochs=500)

    # ----- 6. Loss (examples) -----
    print("\n=== Loss (Examples) ===")
    y_pred_train = predict(X_train_s, trained_weights, trained_bias)
    train_mse = mse(y_train, y_pred_train)
    train_rmse = rmse(y_train, y_pred_train)
    print(f"Train MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")

    # ----- 7. Evaluation -----
    print("\n=== Evaluation ===")
    y_pred_test = predict(X_test_s, trained_weights, trained_bias)
    r2 = r2_score(y_test, y_pred_test)
    test_nrmse = nrmse(y_test, y_pred_test)
    print(f"R\u00b2 Score: {r2:.4f}, NRMSE: {test_nrmse:.4f}")

    # ----- 8. Visualization -----
    print("\n=== Visualization ===")
    try:
        plot_loss(history)
    except Exception as e:
        print("plot_loss failed:", e)
    # Plot appropriate visuals depending on feature dimensionality
    X_s = (X - x_mean) / x_std
    if X.shape[1] == 1:
        try:
            plot_regression_line(X, y, predict(X_s, trained_weights, trained_bias))
        except Exception as e:
            print("plot_regression_line failed:", e)
    else:
        try:
            plot_predictions_vs_actual(y_test, y_pred_test)
            plot_residuals(y_test, y_pred_test)
            plot_residuals_vs_predicted(y_test, y_pred_test)
        except Exception as e:
            print("multi-feature plots failed:", e)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "weights": trained_weights,
        "bias": trained_bias,
        "history": history,
        "y_pred_test": y_pred_test,
    }


def main():
    """Orchestrate the pipeline and show atomic function examples."""
    _ = run_pipeline()


if __name__ == "__main__":
    main()
