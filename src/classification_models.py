"""
SpendSmart AI - Classification Models
Implements:
- Practical 6: Logistic Regression binary classifier with transparent target definition
- Practical 7: Support Vector Machines (Linear vs RBF Kernels)
- Practical 10 (Classification): Baseline vs Random Forest vs HistGradientBoosting
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Practical 6: Logistic Regression.
    Trains scaled binary logistic regression and reports complete classification metrics.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, random_state=42, max_iter=1000)),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_duration = time.perf_counter() - t0

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    roc_auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.0

    model_step = pipeline.named_steps["model"]
    coef_dict = dict(zip(X_train.columns, model_step.coef_[0].tolist()))

    class_distribution = {
        "train_class_0": int((y_train == 0).sum()),
        "train_class_1": int((y_train == 1).sum()),
        "test_class_0": int((y_test == 0).sum()),
        "test_class_1": int((y_test == 1).sum()),
    }

    return {
        "pipeline": pipeline,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "train_time_sec": train_duration,
        "coefficients": coef_dict,
        "class_distribution": class_distribution,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist() if y_prob is not None else [],
    }


def compare_svm_kernels(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Practical 7: Support Vector Machines (Linear Kernel vs RBF Kernel).
    Compares metrics, decision boundaries, and execution time using identical scaling and splits.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    kernels = {
        "Linear SVM": SVC(kernel="linear", C=C, probability=True, random_state=42),
        "RBF SVM (Nonlinear)": SVC(kernel="rbf", C=C, probability=True, random_state=42),
    }

    results = {}

    for name, model in kernels.items():
        t0 = time.perf_counter()
        model.fit(X_tr_s, y_train)
        train_duration = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_te_s)
        pred_duration = time.perf_counter() - t1

        y_prob = model.predict_proba(X_te_s)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        roc_auc = float(roc_auc_score(y_test, y_prob))

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "train_time_sec": train_duration,
            "predict_time_sec": pred_duration,
            "support_vector_count": int(model.support_vectors_.shape[0]),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist(),
        }

    return {
        "models": results,
        "y_test": y_test.tolist(),
    }


def compare_ensemble_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Practical 10: Ensemble Learning Classification.
    Compares Baseline Logistic Regression vs Random Forest vs HistGradientBoosting.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        "Logistic Regression (Baseline)": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "HistGradientBoosting Classifier": HistGradientBoostingClassifier(random_state=42),
    }

    results = {}
    feature_importances = {}

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr_s, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_te_s)
        pred_time = time.perf_counter() - t1

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "Train_Time_Sec": round(train_time, 4),
            "Predict_Time_Sec": round(pred_time, 4),
            "confusion_matrix": cm,
        }

        if hasattr(model, "feature_importances_"):
            feature_importances[name] = dict(zip(X_train.columns, model.feature_importances_.tolist()))
        elif name.startswith("Logistic"):
            feature_importances[name] = dict(zip(X_train.columns, np.abs(model.coef_[0]).tolist()))

    return {
        "metrics_table": results,
        "feature_importances": feature_importances,
        "y_test": y_test.tolist(),
    }
