"""
SpendSmart AI - Feature Engineering
Extracts temporal features, rolling statistics, financial health ratios,
and constructs monthly spending profiles for clustering, classification, and dimensionality reduction.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def engineer_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers transaction-level features from cleaned transaction data:
    - Date decomposition (Year, Month, Day, DayOfWeek, IsWeekend, Quarter, MonthProgress)
    - Income-to-Expense ratio
    - Budget utilization per transaction
    - Rolling spending totals (7-day, 30-day)
    """
    feat_df = df.copy()

    # Ensure Date is datetime
    if not np.issubdtype(feat_df["Date"].dtype, np.datetime64):
        feat_df["Date"] = pd.to_datetime(feat_df["Date"])

    # Temporal features
    feat_df["Year"] = feat_df["Date"].dt.year
    feat_df["Month"] = feat_df["Date"].dt.month
    feat_df["Day"] = feat_df["Date"].dt.day
    feat_df["DayOfWeek"] = feat_df["Date"].dt.dayofweek  # 0=Mon, 6=Sun
    feat_df["IsWeekend"] = feat_df["DayOfWeek"].isin([5, 6]).astype(int)
    feat_df["Quarter"] = feat_df["Date"].dt.quarter
    feat_df["DayOfYear"] = feat_df["Date"].dt.dayofyear

    # Financial health indicators
    # Handle safe division
    feat_df["Income_to_Expense_Ratio"] = np.where(
        feat_df["Expense_Amount"] > 0,
        (feat_df["Income"] / feat_df["Expense_Amount"]).clip(0, 100),
        100.0,
    )

    feat_df["Budget_Utilization"] = np.where(
        feat_df["Budget"] > 0,
        (feat_df["Expense_Amount"] / feat_df["Budget"]).clip(0, 10),
        0.0,
    )

    feat_df["Savings_Rate"] = np.where(
        feat_df["Income"] > 0,
        (feat_df["Savings"] / feat_df["Income"]).clip(0, 1),
        0.0,
    )

    # Rolling window metrics (chronological)
    feat_df = feat_df.sort_values("Date").reset_index(drop=True)
    # Daily aggregated expenses for rolling computation
    daily_exp = feat_df.groupby("Date")["Expense_Amount"].sum().reset_index()
    daily_exp = daily_exp.set_index("Date")
    daily_exp["Rolling_7d_Expense"] = daily_exp["Expense_Amount"].rolling("7D", min_periods=1).mean()
    daily_exp["Rolling_30d_Expense"] = daily_exp["Expense_Amount"].rolling("30D", min_periods=1).mean()
    daily_exp = daily_exp.reset_index()

    feat_df = pd.merge(
        feat_df,
        daily_exp[["Date", "Rolling_7d_Expense", "Rolling_30d_Expense"]],
        on="Date",
        how="left",
    )

    return feat_df


def create_monthly_spending_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs user-level monthly spending profile matrix.
    Aggregates category-wise monthly expenses, total spending, transaction frequency,
    budget utilization, and assigns spending behavior profile classes for LDA/Classification/Clustering.
    """
    clean_df = df.copy()
    if not np.issubdtype(clean_df["Date"].dtype, np.datetime64):
        clean_df["Date"] = pd.to_datetime(clean_df["Date"])

    clean_df["YearMonth"] = clean_df["Date"].dt.to_period("M").astype(str)

    # Filter out salary deposits for spending profile calculations
    expense_mask = clean_df["Transaction_Type"] != "Income"
    exp_df = clean_df[expense_mask].copy()

    # Pivot category expenses per month
    cat_pivot = exp_df.pivot_table(
        index="YearMonth",
        columns="Category",
        values="Expense_Amount",
        aggfunc="sum",
        fill_value=0.0,
    )
    cat_pivot.columns = [f"Exp_{c}" for c in cat_pivot.columns]

    # Monthly aggregates
    monthly_summary = exp_df.groupby("YearMonth").agg(
        Total_Expenses=("Expense_Amount", "sum"),
        Avg_Transaction=("Expense_Amount", "mean"),
        Transaction_Count=("Transaction_ID", "count"),
        Monthly_Income=("Income", "mean"),
        Monthly_Budget=("Budget", "sum"),
        Monthly_Savings=("Savings", "mean"),
    )

    profile_df = pd.concat([monthly_summary, cat_pivot], axis=1).fillna(0.0)

    # Engineered metrics
    profile_df["Savings_Rate"] = (
        profile_df["Monthly_Savings"] / profile_df["Monthly_Income"].replace(0, 1)
    ).clip(0, 1)

    profile_df["Budget_Utilization"] = (
        profile_df["Total_Expenses"] / profile_df["Monthly_Budget"].replace(0, 1)
    ).clip(0, 5)

    profile_df["Discretionary_Ratio"] = (
        profile_df.get("Exp_Dining", 0)
        + profile_df.get("Exp_Entertainment", 0)
        + profile_df.get("Exp_Shopping", 0)
    ) / profile_df["Total_Expenses"].replace(0, 1)

    # Meaningful ground-truth Spending Profile class definition (without future leakage)
    # 0: "Frugal Savers" (Discretionary ratio < 0.25 and Budget utilization <= 0.85)
    # 1: "Balanced Budgeters" (Budget utilization <= 1.0)
    # 2: "High Discretionary" (Discretionary ratio >= 0.40 and Budget utilization <= 1.15)
    # 3: "Overspenders" (Budget utilization > 1.15)
    def assign_profile(row):
        if row["Budget_Utilization"] > 1.15:
            return "Overspender"
        elif row["Discretionary_Ratio"] >= 0.38:
            return "High Discretionary"
        elif row["Discretionary_Ratio"] < 0.25 and row["Budget_Utilization"] < 0.85:
            return "Frugal"
        else:
            return "Balanced"

    profile_df["Spending_Profile"] = profile_df.apply(assign_profile, axis=1)
    profile_df = profile_df.reset_index()

    return profile_df


def get_feature_summary(raw_df: pd.DataFrame, feat_df: pd.DataFrame) -> Dict[str, any]:
    """
    Returns a before-and-after summary of engineered features.
    """
    return {
        "raw_features_count": raw_df.shape[1],
        "engineered_features_count": feat_df.shape[1],
        "newly_added_features": [c for c in feat_df.columns if c not in raw_df.columns],
        "row_count": len(feat_df),
        "memory_usage_mb": float(feat_df.memory_usage(deep=True).sum() / (1024 * 1024)),
    }
