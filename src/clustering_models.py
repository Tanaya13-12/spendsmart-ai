"""
SpendSmart AI - Clustering Models (K-Means)
Implements:
- Practical 8: K-Means Clustering on financial spending profiles
- Evaluates Elbow Curve (Inertia) & Silhouette Analysis
- 2D Cluster Projections, Cluster Profiling, and Downloadable Assignments
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_kmeans_clustering(
    feature_df: pd.DataFrame,
    k_range: range = range(2, 8),
    selected_k: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Practical 8: K-Means Clustering.
    Fits StandardScaler, computes Elbow curve (inertias) & Silhouette scores across k_range,
    then fits the model with selected_k and derives interpretable cluster personas.
    """
    # 1. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    # 2. Elbow & Silhouette Evaluation
    inertias = []
    silhouette_scores = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        score = float(silhouette_score(X_scaled, labels))
        silhouette_scores.append(score)

    # 3. Fit final selected K
    final_km = KMeans(n_clusters=selected_k, random_state=random_state, n_init=15)
    cluster_labels = final_km.fit_predict(X_scaled)

    # Compute 2D PCA projection for visualization
    pca_2d = PCA(n_components=2, random_state=random_state)
    coords_2d = pca_2d.fit_transform(X_scaled)

    # Cluster profiles (mean values per feature in original scale)
    clustered_df = feature_df.copy()
    clustered_df["Cluster"] = cluster_labels
    cluster_means = clustered_df.groupby("Cluster").mean().round(2)

    # Automatically derive helpful persona tags based on dominant spending traits
    personas = {}
    for c_id in range(selected_k):
        c_row = cluster_means.loc[c_id] if c_id in cluster_means.index else None
        if c_row is not None:
            # Check highest spending categories or ratios
            exp_cols = [c for c in c_row.index if c.startswith("Exp_")]
            top_cat = c_row[exp_cols].idxmax().replace("Exp_", "") if exp_cols else "General"
            personas[c_id] = f"Cluster {c_id}: Heavy {top_cat} Spenders"
        else:
            personas[c_id] = f"Cluster {c_id}"

    # Prepare exportable / plot dataframe
    plot_df = pd.DataFrame({
        "PCA_1": coords_2d[:, 0],
        "PCA_2": coords_2d[:, 1],
        "Cluster": [f"Cluster {c}" for c in cluster_labels],
        "Cluster_ID": cluster_labels,
    })

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "selected_k": selected_k,
        "selected_silhouette": float(silhouette_score(X_scaled, cluster_labels)),
        "cluster_means": cluster_means.to_dict(),
        "personas": personas,
        "cluster_labels": cluster_labels.tolist(),
        "pca_variance_ratio": pca_2d.explained_variance_ratio_.tolist(),
        "plot_data": plot_df.to_dict(orient="records"),
    }
