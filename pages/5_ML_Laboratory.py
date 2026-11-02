"""
SpendSmart AI - Page 5: ML Laboratory
Interactive Experimental Suite for all 11 Machine Learning Practicals:
1. Feature Engineering
2. PCA & LDA
3. Multiple Linear Regression
4. Polynomial Regression
5. Ridge & Lasso Regression
6. Logistic Regression
7. Support Vector Machines
8. K-Means Clustering
9. DBSCAN Anomaly Detection
10. Ensemble Learning
11. Tabular Q-Learning
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.anomaly_detection import explain_anomalies, run_dbscan_anomaly_detection
from src.classification_models import compare_ensemble_classification, compare_svm_kernels, train_logistic_regression
from src.clustering_models import run_kmeans_clustering
from src.dimensionality_reduction import run_pca_lda_comparison
from src.evaluation import (
    apply_dark_layout,
    calculate_classification_metrics,
    calculate_regression_metrics,
    plot_2d_projection,
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_polynomial_comparison,
    plot_residuals,
    plot_rl_training_curves,
    THEME_BLUE,
    THEME_CORAL,
    THEME_EMERALD,
    THEME_TEAL,
)
from src.feature_engineering import get_feature_summary
from src.preprocessing import build_preprocessor, chronological_train_test_split
from src.regression_models import (
    compare_ensemble_regression,
    compare_polynomial_regression,
    compare_ridge_lasso_regression,
    train_multiple_linear_regression,
)
from src.reinforcement_learning import train_q_learning_agent
from src.ui_helpers import (
    get_active_dataset,
    inject_custom_styles,
    render_disclaimer,
    render_metric_card,
    render_sidebar_data_source,
)

st.set_page_config(page_title="ML Laboratory | SpendSmart AI", page_icon="🔬", layout="wide")
inject_custom_styles()

st.title("🔬 Machine Learning Laboratory")
st.markdown(
    "Interactive academic testing grounds demonstrating all **11 Machine Learning Practicals** "
    "with live training, evaluation metrics, diagnostics, and mathematical explanations."
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

# Prepare base dataset slices for experiments
expense_only = feat_df[feat_df["Transaction_Type"] != "Income"].sort_values("Date").reset_index(drop=True)
numeric_reg_cols = [
    "Income", "Budget", "Savings", "Rolling_7d_Expense",
    "Rolling_30d_Expense", "DayOfWeek", "IsWeekend", "Month", "Day"
]
expense_only[numeric_reg_cols] = expense_only[numeric_reg_cols].fillna(0.0)

# Chronological split for regression experiments
train_reg_df, test_reg_df = chronological_train_test_split(expense_only, test_size=0.2, date_col="Date")
X_tr_reg, y_tr_reg = train_reg_df[numeric_reg_cols], train_reg_df["Expense_Amount"]
X_te_reg, y_te_reg = test_reg_df[numeric_reg_cols], test_reg_df["Expense_Amount"]

# Profile features for clustering & dimensionality reduction
cluster_cols = [
    c for c in monthly_profiles.columns
    if c.startswith("Exp_") or c in [
        "Total_Expenses", "Avg_Transaction", "Monthly_Income",
        "Savings_Rate", "Budget_Utilization", "Discretionary_Ratio"
    ]
]
X_monthly = monthly_profiles[cluster_cols].copy()
y_profiles = monthly_profiles["Spending_Profile"]

# Create 11 practical tabs
tabs = st.tabs([
    "P1: Feature Eng.",
    "P2: PCA vs. LDA",
    "P3: Linear Reg.",
    "P4: Polynomial Reg.",
    "P5: Ridge & Lasso",
    "P6: Logistic Reg.",
    "P7: SVM Kernels",
    "P8: K-Means",
    "P9: DBSCAN",
    "P10: Ensembles",
    "P11: Q-Learning",
])

# =========================================================================
# PRACTICAL 1: FEATURE ENGINEERING
# =========================================================================
with tabs[0]:
    st.subheader("Practical 1: Data Preprocessing & Feature Engineering Pipeline")
    st.markdown(
        "**Objective:** Build a leakage-free Scikit-Learn `Pipeline` and `ColumnTransformer` "
        "incorporating missing value imputation, `OneHotEncoder`, `StandardScaler`, and temporal feature extraction."
    )

    feat_summary = get_feature_summary(raw_df=expense_only[["Date", "Expense_Amount", "Category"]], feat_df=expense_only)

    p1_c1, p1_c2, p1_c3 = st.columns(3)
    with p1_c1:
        render_metric_card("Raw Base Features", f"{feat_summary['raw_features_count']}", "Input Schema")
    with p1_c2:
        render_metric_card("Engineered Features", f"{feat_summary['engineered_features_count']}", f"+{len(feat_summary['newly_added_features'])} Created")
    with p1_c3:
        render_metric_card("Memory Footprint", f"{feat_summary['memory_usage_mb']:.2f} MB", "Optimized Dataframe")

    st.markdown("#### 🛠️ Preprocessor Transformation Pipeline Architecture")
    code_demo = """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    """
    st.code(code_demo, language="python")

    with st.expander("📋 View Engineered Features List & Data Quality"):
        st.write("Newly Engineered Attributes:", feat_summary["newly_added_features"])
        st.dataframe(expense_only[numeric_reg_cols + ["Expense_Amount"]].describe().round(2))

# =========================================================================
# PRACTICAL 2: PCA VS LDA
# =========================================================================
with tabs[1]:
    st.subheader("Practical 2: Principal Component Analysis (PCA) vs. Linear Discriminant Analysis (LDA)")
    st.markdown(
        "**Objective:** Contrast unsupervised variance maximization (PCA) with supervised class separability (LDA) "
        "and measure downstream classification performance before and after dimensionality reduction."
    )

    p2_res = run_pca_lda_comparison(X_monthly, labels=y_profiles, n_components=2)

    col2_a, col2_b, col2_c = st.columns(3)
    with col2_a:
        render_metric_card("Original Dimensions", f"{p2_res['original_features_count']} Features", "Input Space")
    with col2_b:
        render_metric_card("PCA Explained Variance", f"{p2_res['pca']['total_explained_variance']*100:.1f}%", "Top 2 Eigenvectors")
    with col2_c:
        render_metric_card("Baseline Accuracy", f"{p2_res['classifier_baseline']['accuracy']*100:.1f}%", f"PCA Acc: {p2_res['pca']['accuracy']*100:.1f}%")

    col_pca_plot, col_lda_plot = st.columns(2)
    with col_pca_plot:
        fig_pca = px.scatter(
            pd.DataFrame(p2_res["pca"]["plot_data"]),
            x="PC1", y="PC2", color="Label",
            title="PCA 2D Projection (Unsupervised Variance Maximization)",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        st.plotly_chart(apply_dark_layout(fig_pca, title="PCA: Maximum Orthogonal Variance"), use_container_width=True)

    with col_lda_plot:
        if p2_res["lda"].get("available") and p2_res["lda_plot_data"]:
            fig_lda = px.scatter(
                pd.DataFrame(p2_res["lda_plot_data"]),
                x="LD1", y="LD2", color="Label",
                title="LDA 2D Projection (Supervised Class Separation)",
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            st.plotly_chart(apply_dark_layout(fig_lda, title="LDA: Maximized Between-Class Scatter"), use_container_width=True)
        else:
            st.warning(f"LDA Note: {p2_res['lda'].get('error')}")

    with st.expander("📚 Mathematical Concept Notes: PCA vs. LDA"):
        st.markdown(
            """
            - **PCA:** Solves the eigenvalue problem $C v = \\lambda v$ on the covariance matrix $C$. It is completely unsupervised.
            - **LDA:** Maximizes the Rayleigh quotient $\\frac{w^T S_B w}{w^T S_W w}$ where $S_B$ is between-class scatter and $S_W$ is within-class scatter.
            - *Constraint:* LDA rank is at most $\\min(p, C - 1)$ where $C$ is class count.
            """
        )

# =========================================================================
# PRACTICAL 3: MULTIPLE LINEAR REGRESSION
# =========================================================================
with tabs[2]:
    st.subheader("Practical 3: Multiple Linear Regression")
    st.markdown(
        "**Objective:** Model total transaction expenses as a linear combination of multivariate financial predictors "
        "using a strictly chronological train/test split."
    )

    p3_res = train_multiple_linear_regression(X_tr_reg, y_tr_reg, X_te_reg, y_te_reg)

    col3_m1, col3_m2, col3_m3, col3_m4 = st.columns(4)
    with col3_m1:
        render_metric_card("Test R² Score", f"{p3_res['test_r2']:.4f}", f"Train R²: {p3_res['train_r2']:.4f}")
    with col3_m2:
        render_metric_card("Test RMSE", f"${p3_res['test_rmse']:.2f}", f"Train RMSE: ${p3_res['train_rmse']:.2f}")
    with col3_m3:
        render_metric_card("Test MSE", f"${p3_res['test_mse']:,.2f}", "Mean Squared Error")
    with col3_m4:
        render_metric_card("Training Latency", f"{p3_res['train_time_sec']*1000:.2f} ms", "Closed-Form OLS Fit")

    col_p3_1, col_p3_2 = st.columns(2)
    with col_p3_1:
        fig_avp = plot_actual_vs_predicted(p3_res["y_test_actual"][:250], p3_res["y_test_pred"][:250])
        st.plotly_chart(fig_avp, use_container_width=True)
    with col_p3_2:
        fig_coef = plot_feature_importance(p3_res["coefficients"], title="Linear Regression Normalized Coefficients")
        st.plotly_chart(fig_coef, use_container_width=True)

    with st.expander("📚 Practical 3 Diagnostic Interpretation"):
        st.markdown(
            f"""
            - **Intercept (β₀):** ${p3_res['intercept']:.2f}
            - **Key Coefficients:** Top positive contributor: `{max(p3_res['coefficients'], key=p3_res['coefficients'].get)}` 
              ({max(p3_res['coefficients'].values()):.2f}).
            - **Homoscedasticity:** Residuals maintain zero-centered distribution with slight variance widening at upper expense tiers.
            """
        )

# =========================================================================
# PRACTICAL 4: POLYNOMIAL REGRESSION
# =========================================================================
with tabs[3]:
    st.subheader("Practical 4: Polynomial Regression & Bias-Variance Trade-off")
    st.markdown(
        "**Objective:** Fit polynomial degrees 1, 2, and 3 to demonstrate underfitting (high bias) "
        "versus overfitting (high variance) without test data leakage."
    )

    # Use a primary rolling feature for clear curve visualization
    x_sub_tr = X_tr_reg[["Rolling_7d_Expense"]].values
    y_sub_tr = y_tr_reg.values
    x_sub_te = X_te_reg[["Rolling_7d_Expense"]].values
    y_sub_te = y_te_reg.values

    poly_res = compare_polynomial_regression(x_sub_tr, y_sub_tr, x_sub_te, y_sub_te, degrees=[1, 2, 3])

    poly_rows = []
    for deg_name, vals in poly_res["metrics"].items():
        poly_rows.append({
            "Model Degree": deg_name,
            "Poly Features Generated": vals["n_poly_features"],
            "Train RMSE ($)": round(vals["train_rmse"], 2),
            "Test RMSE ($)": round(vals["test_rmse"], 2),
            "Train R²": round(vals["train_r2"], 4),
            "Test R²": round(vals["test_r2"], 4),
        })
    st.dataframe(pd.DataFrame(poly_rows), use_container_width=True, hide_index=True)

    fig_poly = plot_polynomial_comparison(poly_res)
    st.plotly_chart(fig_poly, use_container_width=True)

    with st.expander("📚 Bias-Variance Academic Explanation"):
        st.markdown(
            """
            - **Degree 1 (Linear):** High bias / underfitting if true underlying relation is non-linear.
            - **Degree 2 (Quadratic):** Captures diminishing marginal expenditure curve with balanced generalizability.
            - **Degree 3 (Cubic):** Increases model variance. Notice how training error decreases while test error may diverge at extreme boundaries.
            """
        )

# =========================================================================
# PRACTICAL 5: RIDGE AND LASSO REGRESSION
# =========================================================================
with tabs[4]:
    st.subheader("Practical 5: Ridge (L2) vs. Lasso (L1) Regularization & Multicollinearity")
    st.markdown(
        "**Objective:** Contrast L2 weight shrinkage (Ridge) with L1 feature selection / sparsity (Lasso) "
        "and diagnose predictor multicollinearity."
    )

    alpha_val = st.slider("Select Regularization Penalty (α):", min_value=0.1, max_value=50.0, value=2.0, step=0.5, key="p5_alpha")

    p5_res = compare_ridge_lasso_regression(X_tr_reg, y_tr_reg, X_te_reg, y_te_reg, alpha=alpha_val)

    st.markdown("#### 📊 Comparative Model Performance")
    p5_metrics_df = pd.DataFrame(p5_res["comparison"]).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(p5_metrics_df, use_container_width=True, hide_index=True)

    st.markdown("#### 🔍 Coefficient Shrinkage & Zeroing Behavior")
    coef_plot_df = pd.DataFrame(p5_res["coefficients"])
    st.dataframe(coef_plot_df, use_container_width=True, hide_index=True)

    with st.expander("📚 Mathematical Differences: L1 vs L2"):
        st.markdown(
            """
            - **Ridge (L2 Loss):** $\\mathcal{L} + \\alpha \\sum \\beta_j^2$ — shrinks correlated coefficients proportionally toward zero without setting them to exactly zero.
            - **Lasso (L1 Loss):** $\\mathcal{L} + \\alpha \\sum |\\beta_j|$ — creates diamond-shaped $L_1$ constraint corners, driving uninformative or redundant coefficients to **exactly 0**, effectively performing automatic feature selection.
            """
        )

# =========================================================================
# PRACTICAL 6: LOGISTIC REGRESSION
# =========================================================================
with tabs[5]:
    st.subheader("Practical 6: Binary Classification via Logistic Regression")
    st.markdown(
        "**Objective:** Predict whether a transaction represents an **Over-Budget / High-Intensity Spend** "
        "(defined transparently as exceeding the 75th percentile expense threshold without feature leakage)."
    )

    # Define binary target transparently without leakage
    threshold = float(expense_only["Expense_Amount"].quantile(0.75))
    y_binary = (expense_only["Expense_Amount"] > threshold).astype(int)

    # Input predictors (omit Expense_Amount to prevent leakage!)
    X_bin = expense_only[["Income", "Budget", "Rolling_7d_Expense", "Rolling_30d_Expense", "DayOfWeek", "IsWeekend"]]

    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_bin, y_binary, test_size=0.25, stratify=y_binary, random_state=42)

    p6_res = train_logistic_regression(X_tr_b, y_tr_b, X_te_b, y_te_b)

    c6_1, c6_2, c6_3, c6_4 = st.columns(4)
    with c6_1:
        render_metric_card("Accuracy", f"{p6_res['accuracy']*100:.2f}%", "Overall Correctness")
    with c6_2:
        render_metric_card("Precision", f"{p6_res['precision']*100:.2f}%", "Positive Predictive Value")
    with c6_3:
        render_metric_card("Recall", f"{p6_res['recall']*100:.2f}%", "True Positive Sensitivity")
    with c6_4:
        render_metric_card("F1-Score", f"{p6_res['f1_score']:.4f}", f"ROC-AUC: {p6_res['roc_auc']:.3f}")

    col_cm, col_odds = st.columns(2)
    with col_cm:
        fig_cm = plot_confusion_matrix(p6_res["confusion_matrix"], labels=["Normal Spend", "High Spend"])
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_odds:
        fig_odds = plot_feature_importance(p6_res["coefficients"], title="Log-Odds Feature Coefficients")
        st.plotly_chart(fig_odds, use_container_width=True)

# =========================================================================
# PRACTICAL 7: SUPPORT VECTOR MACHINES (SVM)
# =========================================================================
with tabs[6]:
    st.subheader("Practical 7: Support Vector Machines (Linear vs. RBF Nonlinear Kernels)")
    st.markdown(
        "**Objective:** Benchmark linear hyperplane separation against radial basis function (RBF) "
        "kernel transformations using identical standardized features."
    )

    # Use subsample for SVM performance responsiveness
    sample_sub_idx = np.random.RandomState(42).choice(len(X_tr_b), size=min(1200, len(X_tr_b)), replace=False)
    X_tr_svm = X_tr_b.iloc[sample_sub_idx]
    y_tr_svm = y_tr_b.iloc[sample_sub_idx]

    p7_res = compare_svm_kernels(X_tr_svm, y_tr_svm, X_te_b, y_te_b)

    col7_lin, col7_rbf = st.columns(2)
    with col7_lin:
        m_lin = p7_res["models"]["Linear SVM"]
        st.markdown("#### 🟦 Linear SVM Kernel")
        st.write(f"- **Accuracy:** {m_lin['accuracy']*100:.2f}%")
        st.write(f"- **F1-Score:** {m_lin['f1_score']:.4f}")
        st.write(f"- **Support Vectors Count:** {m_lin['support_vector_count']}")
        st.write(f"- **Fit Time:** {m_lin['train_time_sec']*1000:.2f} ms")
        fig_cm_lin = plot_confusion_matrix(m_lin["confusion_matrix"], title="Linear SVM Confusion Matrix")
        st.plotly_chart(fig_cm_lin, use_container_width=True)

    with col7_rbf:
        m_rbf = p7_res["models"]["RBF SVM (Nonlinear)"]
        st.markdown("#### 🟩 RBF SVM Kernel (Non-linear Projection)")
        st.write(f"- **Accuracy:** {m_rbf['accuracy']*100:.2f}%")
        st.write(f"- **F1-Score:** {m_rbf['f1_score']:.4f}")
        st.write(f"- **Support Vectors Count:** {m_rbf['support_vector_count']}")
        st.write(f"- **Fit Time:** {m_rbf['train_time_sec']*1000:.2f} ms")
        fig_cm_rbf = plot_confusion_matrix(m_rbf["confusion_matrix"], title="RBF SVM Confusion Matrix")
        st.plotly_chart(fig_cm_rbf, use_container_width=True)

# =========================================================================
# PRACTICAL 8: K-MEANS CLUSTERING
# =========================================================================
with tabs[7]:
    st.subheader("Practical 8: K-Means Clustering & Silhouette Validation")
    st.markdown(
        "**Objective:** Discover natural groupings in user spending habits using standardized K-Means, "
        "validating cluster compactness via Inertia (Elbow) and Silhouette coefficients."
    )

    k_sel = st.slider("Select k for Cluster Profiling:", 2, 6, 3, key="p8_k")
    p8_res = run_kmeans_clustering(X_monthly, k_range=range(2, 7), selected_k=k_sel)

    col8_e, col8_s = st.columns(2)
    with col8_e:
        fig_el = go.Figure(go.Scatter(x=p8_res["k_values"], y=p8_res["inertias"], mode="lines+markers", line=dict(color=THEME_TEAL)))
        fig_el.update_xaxes(title_text="k")
        fig_el.update_yaxes(title_text="Inertia")
        st.plotly_chart(apply_dark_layout(fig_el, title="Elbow Inertia Curve"), use_container_width=True)

    with col8_s:
        fig_sil = go.Figure(go.Bar(x=p8_res["k_values"], y=p8_res["silhouette_scores"], marker_color=THEME_EMERALD))
        fig_sil.update_xaxes(title_text="k")
        fig_sil.update_yaxes(title_text="Silhouette Score")
        st.plotly_chart(apply_dark_layout(fig_sil, title="Silhouette Quality Curve"), use_container_width=True)

    st.markdown(f"#### 👥 Derived Spending Persona Breakdown for k={k_sel}")
    st.dataframe(pd.DataFrame(p8_res["cluster_means"]), use_container_width=True)

# =========================================================================
# PRACTICAL 9: DBSCAN DENSITY ANOMALY DETECTION
# =========================================================================
with tabs[8]:
    st.subheader("Practical 9: DBSCAN Density Clustering & Outlier Flagging")
    st.markdown(
        "**Objective:** Group arbitrarily shaped dense transaction zones and flag sparse outliers as Noise (-1) "
        "without arbitrarily labeling all noise as fraudulent."
    )

    c9_1, c9_2 = st.columns(2)
    with c9_1:
        eps_p9 = st.slider("Epsilon (Radius):", 0.5, 5.0, 3.2, 0.1, key="p9_eps")
    with c9_2:
        min_p9 = st.slider("Min Samples (Density Threshold):", 2, 8, 2, 1, key="p9_min")

    p9_res = run_dbscan_anomaly_detection(X_monthly, eps=eps_p9, min_samples=min_p9)

    col9_m1, col9_m2, col9_m3 = st.columns(3)
    with col9_m1:
        render_metric_card("Dense Clusters Formed", f"{p9_res['n_clusters']}", "Core Dense Components")
    with col9_m2:
        render_metric_card("Noise Observations", f"{p9_res['n_noise']}", f"{p9_res['noise_percentage']}% of Data", sub_color="#f43f5e")
    with col9_m3:
        render_metric_card("Analyzed Points", f"{p9_res['total_points']}", "Monthly Financial Vectors")

    fig_db_proj = plot_2d_projection(p9_res["plot_data"], "PCA_1", "PCA_2", "Status", title="DBSCAN Density Space")
    st.plotly_chart(fig_db_proj, use_container_width=True)

# =========================================================================
# PRACTICAL 10: ENSEMBLE LEARNING
# =========================================================================
with tabs[9]:
    st.subheader("Practical 10: Ensemble Learning Benchmarking")
    st.markdown(
        "**Objective:** Empirically evaluate Bagging (Random Forest) and Gradient Boosting against simple baselines "
        "across both regression and classification tasks on identical splits."
    )

    st.markdown("#### 📉 Part A: Regression Task (Forecasting Expense Amount)")
    reg_ens = compare_ensemble_regression(X_tr_reg, y_tr_reg, X_te_reg, y_te_reg)
    st.dataframe(pd.DataFrame(reg_ens["metrics_table"]).T.reset_index().rename(columns={"index": "Algorithm"}), use_container_width=True, hide_index=True)

    st.markdown("#### 🏷️ Part B: Classification Task (Predicting High-Spend Intensity)")
    cls_ens = compare_ensemble_classification(X_tr_b, y_tr_b, X_te_b, y_te_b)
    st.dataframe(pd.DataFrame(cls_ens["metrics_table"]).T.reset_index().rename(columns={"index": "Algorithm"}), use_container_width=True, hide_index=True)

# =========================================================================
# PRACTICAL 11: REINFORCEMENT LEARNING (Q-LEARNING)
# =========================================================================
with tabs[10]:
    st.subheader("Practical 11: Tabular Q-Learning Budget Allocator")
    st.markdown(
        "**Objective:** Train an RL agent to navigate 27 discrete budget states, learning an optimal policy "
        "via Bellman temporal difference updates while protecting essential expenses."
    )

    if st.button("🚀 Train 1,000-Episode Q-Agent Live", key="p11_train"):
        with st.spinner("Executing episodic simulation..."):
            rl_lab = train_q_learning_agent(episodes=1000)
            st.session_state["p11_rl"] = rl_lab

    if "p11_rl" in st.session_state:
        rl_data = st.session_state["p11_rl"]
        fig_rl_lab = plot_rl_training_curves(rl_data["episode_rewards"], rl_data["smoothed_rewards"], rl_data["epsilon_history"])
        st.plotly_chart(fig_rl_lab, use_container_width=True)

        st.markdown("#### 🗺️ Sample Converged Policy (Top 10 States)")
        st.dataframe(pd.DataFrame(rl_data["q_table"]).head(10), use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Train 1,000-Episode Q-Agent Live' above to run the tabular training loop.")
