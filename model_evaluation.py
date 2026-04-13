import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error (MSE)"""
    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE)"""
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error (MAE)"""
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true, y_pred):
    """Calculate R² score"""
    return r2_score(y_true, y_pred)


def display_metrics(y_true, y_pred):
    """Display all metrics"""
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")
