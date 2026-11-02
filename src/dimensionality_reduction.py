"""
SpendSmart AI - Dimensionality Reduction (PCA & LDA)
Implements:
- Practical 2: Principal Component Analysis (Unsupervised) and Linear Discriminant Analysis (Supervised)
- Compares variance, class separability, 2D projections, classification accuracy before & after, and execution times.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_pca_lda_comparison(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    n_components: int = 2,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Practical 2: PCA vs LDA.
    Fits PCA and LDA on training data only to avoid leakage.
    Evaluates downstream classifier performance before and after dimensionality reduction.
    """
    # Check eligibility for LDA
    n_classes = len(labels.unique())
    n_features = feature_matrix.shape[1]
    max_lda_components = min(n_features, n_classes - 1)

    lda_possible = True
    lda_error_msg = ""
    if max_lda_components < 1:
        lda_possible = False
        lda_error_msg = (
            f"LDA requires at least 2 distinct classes to produce components. Found {n_classes} classes."
        )

    # Check minimum class frequency for stratified split
    min_class_count = labels.value_counts().min()
    use_stratify = labels if min_class_count >= 2 else None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, labels, test_size=test_size, stratify=use_stratify, random_state=random_state
    )

    # 1. Standard Scaler fitted on X_train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Downstream classifier baseline (All Features)
    clf_orig = LogisticRegression(max_iter=1000, random_state=random_state)
    t0 = time.perf_counter()
    clf_orig.fit(X_train_scaled, y_train)
    time_orig = time.perf_counter() - t0
    y_pred_orig = clf_orig.predict(X_test_scaled)
    acc_orig = float(accuracy_score(y_test, y_pred_orig))
    f1_orig = float(f1_score(y_test, y_pred_orig, average="weighted", zero_division=0))

    # 2. PCA (Unsupervised)
    t0 = time.perf_counter()
    pca = PCA(n_components=min(n_components, n_features), random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    time_pca = time.perf_counter() - t0

    clf_pca = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_pca.fit(X_train_pca, y_train)
    y_pred_pca = clf_pca.predict(X_test_pca)
    acc_pca = float(accuracy_score(y_test, y_pred_pca))
    f1_pca = float(f1_score(y_test, y_pred_pca, average="weighted", zero_division=0))

    explained_var_pca = pca.explained_variance_ratio_.tolist()
    total_var_pca = float(np.sum(pca.explained_variance_ratio_))

    # PCA 2D coordinates for entire dataset (for visualization)
    X_all_scaled = scaler.transform(feature_matrix)
    pca_all_coords = pca.transform(X_all_scaled)
    pca_plot_df = pd.DataFrame({
        "PC1": pca_all_coords[:, 0],
        "PC2": pca_all_coords[:, 1] if pca_all_coords.shape[1] > 1 else np.zeros(len(pca_all_coords)),
        "Label": labels.values,
    })

    # PCA Component loadings (feature influence)
    loadings = pd.DataFrame(
        pca.components_[:2, :].T,
        columns=["PC1_Loading", "PC2_Loading"] if pca.components_.shape[0] > 1 else ["PC1_Loading"],
        index=feature_matrix.columns,
    ).round(4).to_dict()

    # 3. LDA (Supervised)
    lda_results = {}
    lda_plot_df = pd.DataFrame()

    if lda_possible:
        lda_comps = min(n_components, max_lda_components)
        t0 = time.perf_counter()
        lda = LinearDiscriminantAnalysis(n_components=lda_comps)
        try:
            X_train_lda = lda.fit_transform(X_train_scaled, y_train)
            X_test_lda = lda.transform(X_test_scaled)
            time_lda = time.perf_counter() - t0

            clf_lda = LogisticRegression(max_iter=1000, random_state=random_state)
            clf_lda.fit(X_train_lda, y_train)
            y_pred_lda = clf_lda.predict(X_test_lda)
            acc_lda = float(accuracy_score(y_test, y_pred_lda))
            f1_lda = float(f1_score(y_test, y_pred_lda, average="weighted", zero_division=0))

            explained_var_lda = (
                lda.explained_variance_ratio_.tolist()
                if hasattr(lda, "explained_variance_ratio_")
                else []
            )

            # LDA 2D coordinates for entire dataset
            lda_all_coords = lda.transform(X_all_scaled)
            lda_plot_df = pd.DataFrame({
                "LD1": lda_all_coords[:, 0],
                "LD2": (
                    lda_all_coords[:, 1]
                    if lda_all_coords.shape[1] > 1
                    else np.zeros(len(lda_all_coords))
                ),
                "Label": labels.values,
            })

            lda_results = {
                "available": True,
                "components": lda_comps,
                "fit_time_sec": time_lda,
                "accuracy": acc_lda,
                "f1_score": f1_lda,
                "explained_variance": explained_var_lda,
            }
        except Exception as e:
            lda_results = {"available": False, "error": str(e)}
    else:
        lda_results = {"available": False, "error": lda_error_msg}

    return {
        "original_features_count": n_features,
        "reduced_components_count": n_components,
        "classifier_baseline": {
            "accuracy": acc_orig,
            "f1_score": f1_orig,
            "fit_time_sec": time_orig,
        },
        "pca": {
            "components": pca.n_components_,
            "fit_time_sec": time_pca,
            "accuracy": acc_pca,
            "f1_score": f1_pca,
            "explained_variance_ratios": explained_var_pca,
            "total_explained_variance": total_var_pca,
            "plot_data": pca_plot_df.to_dict(orient="records"),
            "loadings": loadings,
        },
        "lda": lda_results,
        "lda_plot_data": lda_plot_df.to_dict(orient="records") if not lda_plot_df.empty else [],
    }
