"""
SpendSmart AI - Page 1: Financial Overview
Descriptive financial analytics: cash flow, spending trends, category distributions,
budget utilization, and interactive transaction filtering.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.evaluation import apply_dark_layout, THEME_BLUE, THEME_CORAL, THEME_EMERALD, THEME_TEAL
from src.ui_helpers import (
    get_active_dataset,
    inject_custom_styles,
    render_disclaimer,
    render_metric_card,
    render_sidebar_data_source,
)

st.set_page_config(page_title="Financial Overview | SpendSmart AI", page_icon="📈", layout="wide")
inject_custom_styles()

st.title("📈 Financial Overview & Cash Flow")
st.markdown("Track and analyze monthly cash flows, category allocations, budget variances, and spending habits.")

render_disclaimer()

# Render centralized data source selector in sidebar
render_sidebar_data_source()

# Load active dataset
try:
    feat_df, monthly_profiles, quality_report = get_active_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ----------------- FILTERS -----------------
with st.sidebar:
    st.header("🔍 Overview Filters")
    min_date = feat_df["Date"].min().to_pydatetime()
    max_date = feat_df["Date"].max().to_pydatetime()

    date_range = st.date_input(
        "Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    all_categories = sorted(feat_df["Category"].unique().tolist())
    selected_categories = st.multiselect(
        "Select Categories:",
        options=all_categories,
        default=all_categories,
    )

# Filter dataframe
filtered_df = feat_df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df["Date"] >= start_dt) & (filtered_df["Date"] <= end_dt)]

if selected_categories:
    filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]

if filtered_df.empty:
    st.warning("No transactions match the selected filter criteria.")
    st.stop()

# ----------------- FINANCIAL KPIS -----------------
# Income records vs Expense records
expense_df = filtered_df[filtered_df["Transaction_Type"] != "Income"]
income_df = filtered_df[filtered_df["Transaction_Type"] == "Income"]

# If income transactions are separated, use them, otherwise calculate monthly income baseline
total_expense = float(expense_df["Expense_Amount"].sum())
if not income_df.empty:
    total_income = float(income_df["Income"].sum())
else:
    # Monthly income times number of active months
    n_months = max(1, filtered_df["Date"].dt.to_period("M").nunique())
    total_income = float(filtered_df["Income"].mean() * n_months)

net_savings = total_income - total_expense
savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_metric_card("Total Income", f"${total_income:,.2f}", "Cumulative Cash Inflow", sub_color="#3b82f6")
with k2:
    render_metric_card("Total Expenses", f"${total_expense:,.2f}", f"{len(expense_df):,} Transactions", sub_color="#f43f5e")
with k3:
    savings_sub = "Positive Cash Flow" if net_savings >= 0 else "Deficit Cash Flow"
    savings_color = "#10b981" if net_savings >= 0 else "#f43f5e"
    render_metric_card("Net Savings", f"${net_savings:,.2f}", savings_sub, sub_color=savings_color)
with k4:
    rate_sub = "Target: ≥ 20.0%"
    rate_color = "#10b981" if savings_rate >= 20 else "#f59e0b"
    render_metric_card("Savings Rate", f"{savings_rate:.1f}%", rate_sub, sub_color=rate_color)

# ----------------- VISUALIZATIONS SECTION 1 -----------------
st.markdown("---")
col_chart1, col_chart2 = st.columns([3, 2])

with col_chart1:
    st.subheader("Monthly Income vs. Expense Flow")
    # Aggregate monthly cash flows
    monthly_flow = filtered_df.groupby(filtered_df["Date"].dt.to_period("M")).agg(
        Monthly_Expense=("Expense_Amount", "sum"),
        Monthly_Income=("Income", "mean"),
    ).reset_index()
    monthly_flow["YearMonth"] = monthly_flow["Date"].astype(str)

    fig_flow = go.Figure()
    fig_flow.add_trace(go.Bar(
        x=monthly_flow["YearMonth"],
        y=monthly_flow["Monthly_Income"],
        name="Income",
        marker_color=THEME_BLUE,
    ))
    fig_flow.add_trace(go.Bar(
        x=monthly_flow["YearMonth"],
        y=monthly_flow["Monthly_Expense"],
        name="Expenses",
        marker_color=THEME_CORAL,
    ))
    fig_flow.update_layout(barmode="group", xaxis_title="Month", yaxis_title="Amount ($)")
    st.plotly_chart(apply_dark_layout(fig_flow, title="Cash Inflow vs. Outflow by Month"), use_container_width=True)

with col_chart2:
    st.subheader("Expense Distribution by Category")
    cat_summary = expense_df.groupby("Category")["Expense_Amount"].sum().reset_index()
    cat_summary = cat_summary.sort_values("Expense_Amount", ascending=False)

    fig_donut = px.pie(
        cat_summary,
        names="Category",
        values="Expense_Amount",
        hole=0.55,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(apply_dark_layout(fig_donut, title="Category Share of Wallet"), use_container_width=True)

# ----------------- VISUALIZATIONS SECTION 2 -----------------
st.markdown("---")
col_chart3, col_chart4 = st.columns([3, 2])

with col_chart3:
    st.subheader("Budget vs. Actual Spending Variance")
    # Budget vs Actual per category
    b_vs_a = expense_df.groupby("Category").agg(
        Actual_Spend=("Expense_Amount", "sum"),
        Budget_Allocated=("Budget", "sum"),
    ).reset_index()
    b_vs_a["Variance"] = b_vs_a["Budget_Allocated"] - b_vs_a["Actual_Spend"]

    fig_bva = go.Figure()
    fig_bva.add_trace(go.Bar(
        x=b_vs_a["Category"],
        y=b_vs_a["Budget_Allocated"],
        name="Allocated Budget",
        marker_color="#475569",
    ))
    fig_bva.add_trace(go.Bar(
        x=b_vs_a["Category"],
        y=b_vs_a["Actual_Spend"],
        name="Actual Spend",
        marker_color=THEME_TEAL,
    ))
    fig_bva.update_layout(barmode="group", xaxis_title="Category", yaxis_title="Amount ($)")
    st.plotly_chart(apply_dark_layout(fig_bva, title="Category Budget Adherence"), use_container_width=True)

with col_chart4:
    st.subheader("Payment Method Breakdown")
    pay_summary = filtered_df.groupby("Payment_Method")["Expense_Amount"].agg(["sum", "count"]).reset_index()
    fig_pay = px.bar(
        pay_summary,
        x="Payment_Method",
        y="sum",
        color="Payment_Method",
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    fig_pay.update_layout(showlegend=False, xaxis_title="Payment Method", yaxis_title="Total Spend ($)")
    st.plotly_chart(apply_dark_layout(fig_pay, title="Total Spend by Payment Method"), use_container_width=True)

# ----------------- RECENT TRANSACTIONS TABLE -----------------
st.markdown("---")
st.subheader("📑 Filtered Transactions Log")
st.dataframe(
    filtered_df[[
        "Transaction_ID", "Date", "Expense_Amount", "Category",
        "Subcategory", "Payment_Method", "Budget", "Merchant_Type"
    ]].sort_values("Date", ascending=False).head(200),
    use_container_width=True,
    hide_index=True,
)
