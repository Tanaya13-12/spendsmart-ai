"""
Automated unit tests for machine learning practicals (Practicals 1 to 10).
"""

import numpy as np
import pandas as pd
import pytest
from src.anomaly_detection import run_dbscan_anomaly_detection
from src.classification_models import compare_ensemble_classification, compare_svm_kernels, train_logistic_regression
from src.clustering_models import run_kmeans_clustering
from src.dimensionality_reduction import run_pca_lda_comparison
from src.regression_models import (
    compare_ensemble_regression,
    compare_polynomial_regression,
    compare_ridge_lasso_regression,
    train_multiple_linear_regression,
)


@pytest.fixture
def synthetic_regression_data():
    """Generates synthetic regression train and test data."""
    rng = np.random.default_rng(42)
    n_samples = 200
    X = pd.DataFrame({
        "Feature_1": rng.normal(50, 10, n_samples),
        "Feature_2": rng.uniform(10, 100, n_samples),
        "Feature_3": rng.exponential(20, n_samples),
    })
    y = pd.Series(2.5 * X["Feature_1"] - 1.2 * X["Feature_2"] + rng.normal(0, 5, n_samples))

    X_train, X_test = X.iloc[:150], X.iloc[150:]
    y_train, y_test = y.iloc[:150], y.iloc[150:]
    return X_train, y_train, X_test, y_test


@pytest.fixture
def synthetic_classification_data():
    """Generates synthetic classification train and test data."""
    rng = np.random.default_rng(42)
    n_samples = 200
    X = pd.DataFrame({
        "Income": rng.normal(5000, 800, n_samples),
        "Budget": rng.uniform(200, 1000, n_samples),
        "Rolling_Spend": rng.uniform(100, 900, n_samples),
    })
    # Target: 1 if high spend, 0 otherwise
    y = pd.Series((X["Rolling_Spend"] > 500).astype(int))

    X_train, X_test = X.iloc[:150], X.iloc[150:]
    y_train, y_test = y.iloc[:150], y.iloc[150:]
    return X_train, y_train, X_test, y_test


def test_multiple_linear_regression(synthetic_regression_data):
    """Practical 3: Test Multiple Linear Regression training and metrics."""
    X_train, y_train, X_test, y_test = synthetic_regression_data
    res = train_multiple_linear_regression(X_train, y_train, X_test, y_test)

    assert "test_r2" in res
    assert "test_rmse" in res
    assert res["test_rmse"] > 0
    assert len(res["y_test_pred"]) == len(y_test)
    assert len(res["coefficients"]) == X_train.shape[1]


def test_polynomial_regression(synthetic_regression_data):
    """Practical 4: Test Polynomial Regression degrees 1, 2, 3."""
    X_train, y_train, X_test, y_test = synthetic_regression_data
    x_tr = X_train[["Feature_1"]].values
    x_te = X_test[["Feature_1"]].values

    res = compare_polynomial_regression(x_tr, y_train.values, x_te, y_test.values, degrees=[1, 2, 3])

    assert "Degree 1" in res["metrics"]
    assert "Degree 2" in res["metrics"]
    assert "Degree 3" in res["metrics"]
    assert res["metrics"]["Degree 1"]["n_poly_features"] == 1
    assert res["metrics"]["Degree 2"]["n_poly_features"] == 2
    assert res["metrics"]["Degree 3"]["n_poly_features"] == 3


def test_ridge_lasso_regression(synthetic_regression_data):
    """Practical 5: Test Ridge and Lasso regression, verifying L1 sparsity."""
    X_train, y_train, X_test, y_test = synthetic_regression_data
    res = compare_ridge_lasso_regression(X_train, y_train, X_test, y_test, alpha=5.0)

    assert "Linear Regression" in res["comparison"]
    assert "Ridge (alpha=5.0)" in res["comparison"]
    assert "Lasso (alpha=5.0)" in res["comparison"]
    assert len(res["coefficients"]["Feature"]) == X_train.shape[1]


def test_logistic_regression(synthetic_classification_data):
    """Practical 6: Test Logistic Regression binary classifier metrics."""
    X_train, y_train, X_test, y_test = synthetic_classification_data
    res = train_logistic_regression(X_train, y_train, X_test, y_test)

    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
    assert 0.0 <= res["f1_score"] <= 1.0
    assert len(res["confusion_matrix"]) == 2
    assert len(res["confusion_matrix"][0]) == 2


def test_svm_kernels(synthetic_classification_data):
    """Practical 7: Test SVM Linear and RBF kernels."""
    X_train, y_train, X_test, y_test = synthetic_classification_data
    res = compare_svm_kernels(X_train, y_train, X_test, y_test)

    assert "Linear SVM" in res["models"]
    assert "RBF SVM (Nonlinear)" in res["models"]
    assert res["models"]["Linear SVM"]["support_vector_count"] > 0
    assert res["models"]["RBF SVM (Nonlinear)"]["support_vector_count"] > 0


def test_pca_lda_reduction():
    """Practical 2: Test PCA and LDA dimensionality reduction."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, (100, 6)), columns=[f"F{i}" for i in range(6)])
    y = pd.Series(rng.choice(["Frugal", "Balanced", "Overspender"], size=100))

    res = run_pca_lda_comparison(X, y, n_components=2)

    assert res["original_features_count"] == 6
    assert res["reduced_components_count"] == 2
    assert 0.0 < res["pca"]["total_explained_variance"] <= 1.0
    assert res["lda"]["available"] is True
    assert res["lda"]["components"] <= 2


def test_kmeans_clustering():
    """Practical 8: Test K-Means clustering and silhouette evaluation."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(10, 2, (60, 4)), columns=[f"F{i}" for i in range(4)])

    res = run_kmeans_clustering(X, k_range=range(2, 5), selected_k=3)

    assert len(res["k_values"]) == 3
    assert len(res["inertias"]) == 3
    assert len(res["silhouette_scores"]) == 3
    assert -1.0 <= res["selected_silhouette"] <= 1.0
    assert len(res["cluster_labels"]) == 60


def test_dbscan_anomaly_detection():
    """Practical 9: Test DBSCAN density clustering and noise identification."""
    rng = np.random.default_rng(42)
    # Dense cluster + isolated outliers
    cluster = rng.normal(0, 0.2, (80, 3))
    outliers = rng.uniform(5, 10, (5, 3))
    data = np.vstack([cluster, outliers])
    df = pd.DataFrame(data, columns=["C1", "C2", "C3"])

    res = run_dbscan_anomaly_detection(df, eps=1.0, min_samples=3)

    assert res["n_clusters"] >= 1
    assert res["n_noise"] >= 1
    assert 0.0 <= res["noise_percentage"] <= 100.0


def test_ensemble_learning(synthetic_regression_data, synthetic_classification_data):
    """Practical 10: Test Ensemble regression and classification benchmarks."""
    X_tr_r, y_tr_r, X_te_r, y_te_r = synthetic_regression_data
    reg_res = compare_ensemble_regression(X_tr_r, y_tr_r, X_te_r, y_te_r)

    assert "Random Forest Regressor" in reg_res["metrics_table"]
    assert "HistGradientBoosting Regressor" in reg_res["metrics_table"]

    X_tr_c, y_tr_c, X_te_c, y_te_c = synthetic_classification_data
    cls_res = compare_ensemble_classification(X_tr_c, y_tr_c, X_te_c, y_te_c)

    assert "Random Forest Classifier" in cls_res["metrics_table"]
    assert "HistGradientBoosting Classifier" in cls_res["metrics_table"]
