"""
SpendSmart AI - Page 3: Spending Analysis
Implements Unsupervised Clustering (K-Means & DBSCAN Anomaly Detection)
and Dimensionality Reduction (PCA & LDA).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.anomaly_detection import explain_anomalies, run_dbscan_anomaly_detection
from src.clustering_models import run_kmeans_clustering
from src.dimensionality_reduction import run_pca_lda_comparison
from src.evaluation import apply_dark_layout, plot_2d_projection, THEME_CORAL, THEME_EMERALD, THEME_TEAL
from src.ui_helpers import (
    get_active_dataset,
    inject_custom_styles,
    render_disclaimer,
    render_metric_card,
    render_sidebar_data_source,
)

st.set_page_config(page_title="Spending Analysis | SpendSmart AI", page_icon="🔍", layout="wide")
inject_custom_styles()

st.title("🔍 Spending Analysis & Pattern Discovery")
st.markdown(
    "Discover user spending profiles, cluster personas, unusual anomaly transactions, "
    "and structural dimensionality reduction using **K-Means, DBSCAN, PCA, and LDA**."
)

render_disclaimer()

# Render centralized data source selector in sidebar
render_sidebar_data_source()

# Load active dataset
try:
    feat_df, monthly_profiles, quality_report = get_active_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Prepare feature matrix for clustering / dimensionality reduction
cluster_features = [
    c for c in monthly_profiles.columns
    if c.startswith("Exp_") or c in [
        "Total_Expenses", "Avg_Transaction", "Monthly_Income",
        "Savings_Rate", "Budget_Utilization", "Discretionary_Ratio"
    ]
]
X_monthly = monthly_profiles[cluster_features].copy()
y_profiles = monthly_profiles["Spending_Profile"]

tab_kmeans, tab_dbscan, tab_dimred = st.tabs([
    "🎯 K-Means Persona Clustering (Prac 8)",
    "🚨 DBSCAN Anomaly Detection (Prac 9)",
    "📐 PCA vs. LDA Reduction (Prac 2)",
])

# =========================================================================
# TAB 1: K-MEANS PERSONA CLUSTERING
# =========================================================================
with tab_kmeans:
    st.subheader("K-Means Spending Profile Segmentation")
    st.markdown(
        "Segments financial months into distinct behavioral patterns using normalized category spending, "
        "savings rates, and budget utilization."
    )

    col_k_ctrl, col_k_info = st.columns([1, 2])
    with col_k_ctrl:
        k_clusters = st.slider("Select Number of Clusters (k):", min_value=2, max_value=6, value=3, step=1)

    kmeans_res = run_kmeans_clustering(X_monthly, k_range=range(2, 7), selected_k=k_clusters)

    km1, km2, km3 = st.columns(3)
    with km1:
        render_metric_card("Selected Clusters (k)", f"{k_clusters}", "User Persona Groups")
    with km2:
        render_metric_card("Silhouette Score", f"{kmeans_res['selected_silhouette']:.3f}", "Cluster Cohesion & Separation")
    with km3:
        best_k = kmeans_res["k_values"][int(np.argmax(kmeans_res["silhouette_scores"]))]
        render_metric_card("Optimal k (by Silhouette)", f"{best_k}", "Highest Separation Score", sub_color="#3b82f6")

    # Evaluation curves (Elbow & Silhouette)
    col_elbow, col_sil = st.columns(2)
    with col_elbow:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=kmeans_res["k_values"],
            y=kmeans_res["inertias"],
            mode="lines+markers",
            marker=dict(color=THEME_TEAL, size=8),
            line=dict(color=THEME_TEAL, width=2),
        ))
        fig_elbow.update_xaxes(title_text="Number of Clusters (k)")
        fig_elbow.update_yaxes(title_text="Inertia (Within-Cluster Sum of Squares)")
        st.plotly_chart(apply_dark_layout(fig_elbow, title="Elbow Method Curve"), use_container_width=True)

    with col_sil:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=kmeans_res["k_values"],
            y=kmeans_res["silhouette_scores"],
            marker_color=[THEME_EMERALD if k == best_k else "#334155" for k in kmeans_res["k_values"]],
        ))
        fig_sil.update_xaxes(title_text="Number of Clusters (k)")
        fig_sil.update_yaxes(title_text="Silhouette Coefficient")
        st.plotly_chart(apply_dark_layout(fig_sil, title="Silhouette Coefficient by k"), use_container_width=True)

    # 2D Cluster Visualizations
    fig_km_2d = plot_2d_projection(
        plot_data=kmeans_res["plot_data"],
        x_col="PCA_1",
        y_col="PCA_2",
        color_col="Cluster",
        title=f"K-Means (k={k_clusters}) 2D PCA Spending Cluster Projection",
    )
    st.plotly_chart(fig_km_2d, use_container_width=True)

    # Personas Table
    st.markdown("#### 👥 Cluster Spending Personas (Mean Values)")
    st.dataframe(pd.DataFrame(kmeans_res["cluster_means"]), use_container_width=True)

# =========================================================================
# TAB 2: DBSCAN ANOMALY DETECTION
# =========================================================================
with tab_dbscan:
    st.subheader("DBSCAN Density-Based Outlier & Anomaly Detection")
    st.markdown(
        "Identifies high-density clusters of routine transactions and flags points in sparse multi-dimensional regions as **Noise (-1)**. "
        "Unlike K-Means, DBSCAN does not assume spherical clusters or require a fixed cluster count."
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        db_eps = st.slider("Neighborhood Radius (eps):", min_value=0.5, max_value=5.0, value=3.2, step=0.1)
    with col_d2:
        db_min_samples = st.slider("Minimum Samples (min_samples):", min_value=2, max_value=10, value=2, step=1)

    # Run DBSCAN on monthly spending profiles
    dbscan_res = run_dbscan_anomaly_detection(X_monthly, eps=db_eps, min_samples=db_min_samples)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        render_metric_card("Dense Clusters", f"{dbscan_res['n_clusters']}", "Core Dense Regions")
    with d2:
        render_metric_card("Anomalies (Noise)", f"{dbscan_res['n_noise']}", f"Points labeled as -1", sub_color="#f43f5e")
    with d3:
        render_metric_card("Noise Proportion", f"{dbscan_res['noise_percentage']}%", "Outlier Frequency", sub_color="#f59e0b")
    with d4:
        render_metric_card("Total Analyzed", f"{dbscan_res['total_points']}", "Monthly Profiles")

    # 2D Scatter Plot
    fig_db_2d = plot_2d_projection(
        plot_data=dbscan_res["plot_data"],
        x_col="PCA_1",
        y_col="PCA_2",
        color_col="Status",
        title=f"DBSCAN Density Clusters & Outliers (eps={db_eps}, min_samples={db_min_samples})",
    )
    st.plotly_chart(fig_db_2d, use_container_width=True)

    st.markdown("#### 🚨 Transaction-Level Anomalies & Diagnostic Reasoning")
    # Also run DBSCAN on a sample of raw transaction features for transaction-level fraud/anomaly inspection
    raw_expense_df = feat_df[feat_df["Transaction_Type"] != "Income"].head(1000).copy()
    raw_num_features = raw_expense_df[["Expense_Amount", "Budget", "Rolling_7d_Expense", "Rolling_30d_Expense"]].fillna(0)
    db_txn = run_dbscan_anomaly_detection(raw_num_features, eps=1.8, min_samples=4)

    noise_mask = np.array(db_txn["labels"]) == -1
    anomaly_table = explain_anomalies(raw_expense_df, noise_mask, top_n=15)

    if not anomaly_table.empty:
        st.dataframe(anomaly_table, use_container_width=True, hide_index=True)
    else:
        st.info("No extreme noise points detected with current density threshold.")

# =========================================================================
# TAB 3: PCA VS LDA DIMENSIONALITY REDUCTION
# =========================================================================
with tab_dimred:
    st.subheader("Dimensionality Reduction: Unsupervised PCA vs. Supervised LDA")
    st.markdown(
        "Dimensionality reduction condenses wide financial feature spaces into lower-dimensional projections. "
        "**PCA** preserves maximum variance without labels, while **LDA** maximizes class separability between spending personas."
    )

    pca_lda_res = run_pca_lda_comparison(X_monthly, labels=y_profiles, n_components=2)

    p1, p2, p3 = st.columns(3)
    with p1:
        render_metric_card("Original Features", f"{pca_lda_res['original_features_count']}", "Input Dimensions")
    with p2:
        render_metric_card("PCA Retained Variance", f"{pca_lda_res['pca']['total_explained_variance']*100:.1f}%", "2 Principal Components")
    with p3:
        lda_acc = f"{pca_lda_res['lda']['accuracy']*100:.1f}%" if pca_lda_res["lda"].get("available") else "N/A"
        render_metric_card("LDA Classifier Accuracy", lda_acc, "Downstream Linear Boundary", sub_color="#3b82f6")

    col_pca_plot, col_lda_plot = st.columns(2)
    with col_pca_plot:
        fig_pca = plot_2d_projection(
            plot_data=pca_lda_res["pca"]["plot_data"],
            x_col="PC1",
            y_col="PC2",
            color_col="Label",
            title=f"PCA 2D Projection (Variance: {pca_lda_res['pca']['total_explained_variance']*100:.1f}%)",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_lda_plot:
        if pca_lda_res["lda"].get("available") and pca_lda_res["lda_plot_data"]:
            fig_lda = plot_2d_projection(
                plot_data=pca_lda_res["lda_plot_data"],
                x_col="LD1",
                y_col="LD2",
                color_col="Label",
                title="LDA 2D Projection (Maximized Class Separation)",
            )
            st.plotly_chart(fig_lda, use_container_width=True)
        else:
            st.warning(f"LDA could not be rendered: {pca_lda_res['lda'].get('error', 'Insufficient classes')}")

    # Theoretical Distinction Expander
    with st.expander("📚 Theoretical Comparison: PCA vs. LDA", expanded=True):
        st.markdown(
            """
            - **Principal Component Analysis (PCA):** Unsupervised algorithm that identifies orthogonal axes (eigenvectors of covariance matrix) 
              that maximize feature variance. It ignores class labels completely.
            - **Linear Discriminant Analysis (LDA):** Supervised algorithm that finds linear combinations of features that maximize between-class 
              scatter relative to within-class scatter:
              $$\\max \\frac{w^T S_B w}{w^T S_W w}$$
              *Constraint:* LDA can generate at most $\\min(p, C-1)$ components, where $C$ is the number of distinct target classes.
            """
        )
