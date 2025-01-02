import numpy as np
import pandas as pd

from utils.dataset import generate_regression_test_dataset


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":
    df = generate_regression_test_dataset()
    y_true = df['True_Values'].values
    y_pred = df['Predicted_Values'].values

    print("Mean Absolute Error (MAE):", mean_absolute_error(y_true, y_pred))
    print("Mean Squared Error (MSE):", mean_squared_error(y_true, y_pred))
    print("Root Mean Squared Error (RMSE):", root_mean_squared_error(y_true, y_pred))
    print("R-squared (R²):", r_squared(y_true, y_pred))
    print("Mean Absolute Percentage Error (MAPE):", mean_absolute_percentage_error(y_true, y_pred))
