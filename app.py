"""
SpendSmart AI - Main Application Entry Point
End-to-End Financial Analytics, Machine Learning Suite & Q-Learning Simulator
"""

import streamlit as st
import pandas as pd
from src.ui_helpers import (
    get_active_dataset,
    inject_custom_styles,
    render_disclaimer,
    render_metric_card,
    render_sidebar_data_source,
)

st.set_page_config(
    page_title="SpendSmart AI | Personal Finance ML Suite",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_styles()

# ----------------- SIDEBAR CONTROLS -----------------
render_sidebar_data_source()

with st.sidebar:
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.4;">
            <b>Academic Project Details:</b><br>
            • BE AI & Data Science (3rd Year)<br>
            • Covers Practicals 1 through 11<br>
            • Scikit-learn + Tabular Q-Learning<br>
            • Leakage-Free Chronological Splitting
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- LOAD DATASET -----------------
try:
    feat_df, monthly_profiles, quality_report = get_active_dataset()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ----------------- HERO HEADER -----------------
st.markdown('<span class="badge-tag">Academic ML Project • 11 Practicals Demonstrated</span>', unsafe_allow_html=True)
st.title("SpendSmart AI: Personal Finance & Spending Analyzer")
st.markdown(
    "An end-to-end intelligent personal finance platform integrating descriptive analytics, "
    "supervised forecasting, unsupervised segmentation, density anomaly detection, and "
    "reinforcement learning budget simulation."
)

render_disclaimer()

# ----------------- DATA QUALITY & OVERVIEW BANNER -----------------
st.subheader("📊 Ingestion & Data Quality Summary")

q1, q2, q3, q4 = st.columns(4)
with q1:
    render_metric_card("Total Transactions", f"{quality_report['cleaned_row_count']:,}", "Verified Records")
with q2:
    render_metric_card("Duplicates Cleaned", f"{quality_report['duplicates_removed']}", "Integrity Enforced", sub_color="#3b82f6")
with q3:
    render_metric_card("Missing Imputed", f"{sum(quality_report['missing_before'].values())}", "Group-median / Mode", sub_color="#f59e0b")
with q4:
    render_metric_card("Active Categories", f"{quality_report['total_categories']}", f"{quality_report['date_range'][0]} to {quality_report['date_range'][1]}")

with st.expander("🔍 Inspect Full Data Quality Audit Report", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Data Cleaning Audit Trail:**")
        if quality_report["issues_found"]:
            for issue in quality_report["issues_found"]:
                st.markdown(f"- ⚠️ {issue}")
        else:
            st.markdown("- ✅ No critical data quality defects detected.")
        st.markdown(f"- **Date Coverage:** `{quality_report['date_range'][0]}` to `{quality_report['date_range'][1]}`")
        st.markdown(f"- **Gross Expense Volume:** `${quality_report['total_expense']:,.2f}`")

    with col_b:
        st.markdown("**Missing Values Handled by Preprocessing Pipeline:**")
        missing_df = pd.DataFrame({
            "Column": list(quality_report["missing_before"].keys()),
            "Missing (Raw)": list(quality_report["missing_before"].values()),
            "Missing (Cleaned)": list(quality_report["missing_after"].values()),
        })
        st.dataframe(missing_df[missing_df["Missing (Raw)"] > 0], use_container_width=True, hide_index=True)

# ----------------- NAVIGATION ROADMAP CARDS -----------------
st.markdown("---")
st.subheader("🧭 Explore Modules & Machine Learning Practicals")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="metric-card">
            <h4>📈 1. Financial Overview</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Descriptive financial health dashboard with monthly cash flows, category spending breakdowns,
                budget variances, and transaction filters.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-card">
            <h4>🤖 4. Smart Budget Planner</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                <b>Practical 11:</b> Interactive tabular Q-Learning simulator that learns optimal budget allocation
                while protecting essential expenses.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="metric-card">
            <h4>🔮 2. Expense Prediction</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                <b>Practicals 3, 4, 5, 10:</b> Multivariate linear regression, polynomial trade-offs, Ridge/Lasso shrinkage,
                and Random Forest / Gradient Boosting ensembles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-card">
            <h4>🔬 5. ML Laboratory</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Comprehensive experimental suite showcasing all <b>11 Practicals</b> with mathematical objectives,
                hyperparameter controls, evaluation metrics, and plots.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="metric-card">
            <h4>🔍 3. Spending Analysis</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                <b>Practicals 2, 8, 9:</b> K-Means spending persona clustering, DBSCAN anomaly detection,
                and PCA vs. LDA dimensionality reduction comparisons.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- TRANSACTION PREVIEW TABLE -----------------
st.markdown("---")
st.subheader("📋 Dataset Preview & Exploration")
col_cat, col_search = st.columns([1, 2])
with col_cat:
    selected_category = st.selectbox(
        "Filter by Category:",
        ["All Categories"] + sorted(feat_df["Category"].unique().tolist()),
    )
with col_search:
    search_keyword = st.text_input("Search Merchant, Subcategory, or Transaction ID:", "")

display_df = feat_df.copy()
if selected_category != "All Categories":
    display_df = display_df[display_df["Category"] == selected_category]

if search_keyword.strip():
    kw = search_keyword.strip().lower()
    mask = (
        display_df["Transaction_ID"].astype(str).str.lower().str.contains(kw)
        | display_df["Merchant_Type"].astype(str).str.lower().str.contains(kw)
        | display_df["Subcategory"].astype(str).str.lower().str.contains(kw)
    )
    display_df = display_df[mask]

st.dataframe(
    display_df.head(100)[[
        "Transaction_ID", "Date", "Expense_Amount", "Category",
        "Subcategory", "Payment_Method", "Transaction_Type", "Budget", "Savings", "Merchant_Type"
    ]],
    use_container_width=True,
    hide_index=True,
)
st.caption(f"Displaying top {min(len(display_df), 100)} of {len(display_df):,} matching transactions. Use sidebar to navigate to specialized pages.")
