"""
SpendSmart AI - Regression Models
Implements:
- Practical 3: Multiple Linear Regression with chronological split
- Practical 4: Polynomial Regression (Degrees 1, 2, 3) illustrating Bias-Variance Trade-off
- Practical 5: Ridge and Lasso Regularized Regression with coefficient shrinkage & sparsity
- Practical 10 (Regression): Baseline vs Random Forest vs HistGradientBoosting
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def train_multiple_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Trains Multiple Linear Regression using standard scaling inside a pipeline.
    Prevents leakage by fitting scaler on X_train only.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_mse = float(mean_squared_error(y_train, y_train_pred))
    test_mse = float(mean_squared_error(y_test, y_test_pred))
    train_rmse = float(np.sqrt(train_mse))
    test_rmse = float(np.sqrt(test_mse))
    train_r2 = float(r2_score(y_train, y_train_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))

    model_step = pipeline.named_steps["model"]
    coefficients = dict(zip(X_train.columns, model_step.coef_.tolist()))

    residuals_test = (y_test.values - y_test_pred).tolist()

    return {
        "model": pipeline,
        "feature_names": list(X_train.columns),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_time_sec": train_time,
        "intercept": float(model_step.intercept_),
        "coefficients": coefficients,
        "y_test_actual": y_test.values.tolist(),
        "y_test_pred": y_test_pred.tolist(),
        "residuals": residuals_test,
    }


def compare_polynomial_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    degrees: List[int] = [1, 2, 3],
) -> Dict[str, Any]:
    """
    Compares Polynomial Regression models of degrees 1, 2, and 3.
    Demonstrates bias-variance trade-off, underfitting, and overfitting.
    """
    results = {}
    curves = {}

    # Sort test inputs for smooth curve plotting
    sort_idx = np.argsort(X_test.squeeze())
    X_test_sorted = X_test[sort_idx]
    y_test_sorted = y_test[sort_idx]

    # Fine grid for smooth visualization
    x_min, x_max = min(X_train.min(), X_test.min()), max(X_train.max(), X_test.max())
    x_grid = np.linspace(x_min, x_max, 200).reshape(-1, 1)

    for deg in degrees:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

        pipe.fit(X_train, y_train)

        y_tr_pred = pipe.predict(X_train)
        y_te_pred = pipe.predict(X_test)
        y_grid_pred = pipe.predict(x_grid)

        tr_mse = float(mean_squared_error(y_train, y_tr_pred))
        te_mse = float(mean_squared_error(y_test, y_te_pred))
        tr_rmse = float(np.sqrt(tr_mse))
        te_rmse = float(np.sqrt(te_mse))
        tr_r2 = float(r2_score(y_train, y_tr_pred))
        te_r2 = float(r2_score(y_test, y_te_pred))

        results[f"Degree {deg}"] = {
            "degree": deg,
            "train_mse": tr_mse,
            "test_mse": te_mse,
            "train_rmse": tr_rmse,
            "test_rmse": te_rmse,
            "train_r2": tr_r2,
            "test_r2": te_r2,
            "n_poly_features": pipe.named_steps["poly"].n_output_features_,
        }

        curves[f"Degree {deg}"] = {
            "grid_x": x_grid.squeeze().tolist(),
            "grid_y": y_grid_pred.tolist(),
        }

    return {
        "metrics": results,
        "curves": curves,
        "X_train": X_train.squeeze().tolist(),
        "y_train": y_train.tolist(),
        "X_test_sorted": X_test_sorted.squeeze().tolist(),
        "y_test_sorted": y_test_sorted.tolist(),
    }


def compare_ridge_lasso_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    Trains and compares Linear, Ridge (L2), and Lasso (L1) regression models.
    Demonstrates coefficient shrinkage, zeroing out features (sparsity), and multicollinearity.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        f"Ridge (alpha={alpha})": Ridge(alpha=alpha, random_state=42),
        f"Lasso (alpha={alpha})": Lasso(alpha=alpha, random_state=42, max_iter=10000),
    }

    comparison = {}
    coef_df_dict = {"Feature": list(X_train.columns)}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_tr_pred = model.predict(X_train_scaled)
        y_te_pred = model.predict(X_test_scaled)

        tr_mse = float(mean_squared_error(y_train, y_tr_pred))
        te_mse = float(mean_squared_error(y_test, y_te_pred))
        tr_r2 = float(r2_score(y_train, y_tr_pred))
        te_r2 = float(r2_score(y_test, y_te_pred))

        coefs = model.coef_
        zero_coefs = int(np.sum(np.isclose(coefs, 0.0, atol=1e-5)))

        comparison[name] = {
            "train_mse": tr_mse,
            "test_mse": te_mse,
            "train_rmse": float(np.sqrt(tr_mse)),
            "test_rmse": float(np.sqrt(te_mse)),
            "train_r2": tr_r2,
            "test_r2": te_r2,
            "zeroed_features_count": zero_coefs,
        }
        coef_df_dict[name] = coefs.tolist()

    # Multicollinearity check (correlation matrix)
    corr_matrix = pd.DataFrame(X_train).corr().round(3).to_dict()

    return {
        "comparison": comparison,
        "coefficients": coef_df_dict,
        "feature_correlations": corr_matrix,
        "alpha": alpha,
    }


def compare_ensemble_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Practical 10: Ensemble Learning Regression.
    Compares Baseline Linear Regression vs Random Forest vs HistGradientBoosting.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(random_state=42),
    }

    results = {}
    feature_importances = {}

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr_s, y_train)
        train_duration = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_te_s)
        pred_duration = time.perf_counter() - t1

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        results[name] = {
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "Train_Time_Sec": round(train_duration, 4),
            "Predict_Time_Sec": round(pred_duration, 4),
        }

        if hasattr(model, "feature_importances_"):
            feature_importances[name] = dict(zip(X_train.columns, model.feature_importances_.tolist()))
        elif name == "Linear Regression (Baseline)":
            feature_importances[name] = dict(zip(X_train.columns, np.abs(model.coef_).tolist()))

    return {
        "metrics_table": results,
        "feature_importances": feature_importances,
    }
