"""
SpendSmart AI - Anomaly Detection (DBSCAN)
Implements:
- Practical 9: Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
- Detects non-linear spending clusters and flags outlier/noise transactions
- Provides noise metrics, 2D density visualizations, and anomaly explanations
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_dbscan_anomaly_detection(
    feature_df: pd.DataFrame,
    eps: float = 1.2,
    min_samples: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Practical 9: DBSCAN.
    Identifies high-density spending clusters and flags points in sparse regions as noise (-1).
    Does NOT require a pre-specified cluster count k.
    """
    # 1. Feature scaling is essential for distance-based DBSCAN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    # 2. DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # 3. Calculate metrics
    unique_labels = set(labels)
    n_noise = int(np.sum(labels == -1))
    total_points = len(labels)
    noise_percentage = float((n_noise / total_points) * 100) if total_points > 0 else 0.0

    # Cluster count excluding noise (-1)
    n_clusters = len(unique_labels - {-1})

    # 4. Compute 2D PCA projection for visualization
    pca_2d = PCA(n_components=2, random_state=random_state)
    coords_2d = pca_2d.fit_transform(X_scaled)

    # Distinguish noise vs clusters
    status_tags = ["Anomaly (Noise)" if lbl == -1 else f"Cluster {lbl}" for lbl in labels]

    plot_df = pd.DataFrame({
        "PCA_1": coords_2d[:, 0],
        "PCA_2": coords_2d[:, 1],
        "Cluster_Label": labels,
        "Status": status_tags,
        "Is_Noise": (labels == -1).astype(int),
    })

    # Summary of cluster distributions
    cluster_counts = pd.Series(labels).value_counts().to_dict()

    return {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_percentage": round(noise_percentage, 2),
        "total_points": total_points,
        "cluster_distribution": cluster_counts,
        "labels": labels.tolist(),
        "pca_variance_ratio": pca_2d.explained_variance_ratio_.tolist(),
        "plot_data": plot_df.to_dict(orient="records"),
    }


def explain_anomalies(
    df: pd.DataFrame,
    noise_mask: np.ndarray,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extracts detected noise transactions and provides interpretable diagnostic reasons
    (e.g., category-relative z-score surge, unusually high spend).
    """
    anomalies = df[noise_mask].copy()
    if anomalies.empty:
        return pd.DataFrame()

    # Calculate category spending mean and standard deviation
    cat_stats = df.groupby("Category")["Expense_Amount"].agg(["mean", "std"]).reset_index()
    anomalies = pd.merge(anomalies, cat_stats, on="Category", how="left")

    # Compute category Z-score
    anomalies["Z_Score"] = (
        (anomalies["Expense_Amount"] - anomalies["mean"]) / anomalies["std"].replace(0, 1)
    ).round(2)

    # Diagnostic reason generator
    def generate_reason(row):
        reasons = []
        if row["Z_Score"] > 3.0:
            reasons.append(f"Expense is {row['Z_Score']} std deviations above {row['Category']} average")
        elif row["Expense_Amount"] > 500:
            reasons.append(f"Unusually large one-off transaction (${row['Expense_Amount']:.2f})")
        else:
            reasons.append("Sparse multi-dimensional feature space / isolated transaction behavior")
        return "; ".join(reasons)

    anomalies["Anomaly_Reason"] = anomalies.apply(generate_reason, axis=1)

    display_cols = [
        col for col in [
            "Transaction_ID", "Date", "Category", "Subcategory",
            "Expense_Amount", "Payment_Method", "Z_Score", "Anomaly_Reason"
        ] if col in anomalies.columns
    ]

    return anomalies.sort_values("Expense_Amount", ascending=False).head(top_n)[display_cols]
