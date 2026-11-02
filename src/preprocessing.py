"""
SpendSmart AI - Data Validation and Preprocessing
Implements data validation, missing-value imputation, duplicate detection,
data quality reporting, and leakage-free Scikit-learn pipeline transformations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical schema definition
REQUIRED_COLUMNS = [
    "Date",
    "Expense_Amount",
    "Category",
]

RECOMMENDED_COLUMNS = [
    "Transaction_ID",
    "Date",
    "Income",
    "Expense_Amount",
    "Category",
    "Subcategory",
    "Payment_Method",
    "Transaction_Type",
    "Budget",
    "Savings",
    "Merchant_Type",
]


class DataValidationError(Exception):
    """Raised when an uploaded dataset fails schema validation."""
    pass


def validate_and_clean_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates uploaded transactions dataframe against expected schema,
    cleans anomalies, handles duplicates, imputes missing values thoughtfully,
    and returns cleaned dataframe alongside a comprehensive Data Quality Report.
    """
    if df is None or df.empty:
        raise DataValidationError("The provided dataset is empty.")

    raw_row_count = len(df)
    quality_report: Dict[str, Any] = {
        "raw_row_count": raw_row_count,
        "raw_columns": list(df.columns),
        "issues_found": [],
        "missing_before": df.isna().sum().to_dict(),
        "duplicates_removed": 0,
        "invalid_amounts_handled": 0,
    }

    # 1. Schema check
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        raise DataValidationError(
            f"Dataset is missing required columns: {missing_required}. "
            f"Found columns: {list(df.columns)}"
        )

    clean_df = df.copy()

    # 2. Duplicate Detection
    # If Transaction_ID exists, check for duplicate IDs; else check exact duplicate rows
    if "Transaction_ID" in clean_df.columns:
        dup_mask = clean_df.duplicated(subset=["Transaction_ID"], keep="first")
    else:
        dup_mask = clean_df.duplicated(keep="first")

    num_duplicates = int(dup_mask.sum())
    if num_duplicates > 0:
        clean_df = clean_df[~dup_mask].copy()
        quality_report["duplicates_removed"] = num_duplicates
        quality_report["issues_found"].append(f"Removed {num_duplicates} duplicate records.")

    # 3. Date Parsing
    try:
        clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
    except Exception as e:
        raise DataValidationError(f"Failed to parse 'Date' column: {e}")

    invalid_dates = clean_df["Date"].isna().sum()
    if invalid_dates > 0:
        clean_df = clean_df.dropna(subset=["Date"]).copy()
        quality_report["issues_found"].append(f"Dropped {invalid_dates} rows with unparseable dates.")

    clean_df = clean_df.sort_values("Date").reset_index(drop=True)

    # 4. Numeric conversions and invalid amounts handling
    numeric_cols = ["Expense_Amount"]
    for col in ["Income", "Budget", "Savings"]:
        if col in clean_df.columns:
            numeric_cols.append(col)

    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Negative expense amounts handling (refunds vs data entry error)
    neg_expenses = (clean_df["Expense_Amount"] < 0).sum()
    if neg_expenses > 0:
        clean_df["Expense_Amount"] = clean_df["Expense_Amount"].abs()
        quality_report["invalid_amounts_handled"] += int(neg_expenses)
        quality_report["issues_found"].append(
            f"Converted {neg_expenses} negative expense amounts to positive absolute values."
        )

    # 5. Missing value imputation
    # For Expense_Amount: Group-wise median by Category, falling back to global median
    if clean_df["Expense_Amount"].isna().sum() > 0:
        cat_medians = clean_df.groupby("Category")["Expense_Amount"].transform("median")
        global_median = clean_df["Expense_Amount"].median()
        clean_df["Expense_Amount"] = clean_df["Expense_Amount"].fillna(cat_medians).fillna(global_median)

    # Default filling for optional columns if present
    if "Income" in clean_df.columns and clean_df["Income"].isna().sum() > 0:
        clean_df["Income"] = clean_df["Income"].fillna(clean_df["Income"].median())
    elif "Income" not in clean_df.columns:
        clean_df["Income"] = 5000.0  # sensible default baseline

    if "Budget" in clean_df.columns and clean_df["Budget"].isna().sum() > 0:
        clean_df["Budget"] = clean_df["Budget"].fillna(clean_df["Expense_Amount"] * 1.2)
    elif "Budget" not in clean_df.columns:
        clean_df["Budget"] = clean_df["Expense_Amount"] * 1.2

    if "Savings" in clean_df.columns and clean_df["Savings"].isna().sum() > 0:
        clean_df["Savings"] = clean_df["Savings"].fillna(clean_df["Income"] * 0.2)
    elif "Savings" not in clean_df.columns:
        clean_df["Savings"] = clean_df["Income"] * 0.2

    # Categorical columns imputation
    cat_columns_to_fill = {
        "Category": "Uncategorized",
        "Subcategory": "General",
        "Payment_Method": "Debit Card",
        "Transaction_Type": "Expense",
        "Merchant_Type": "Merchant",
    }
    for c, default_val in cat_columns_to_fill.items():
        if c in clean_df.columns:
            clean_df[c] = clean_df[c].fillna(default_val).astype(str).str.strip()
        else:
            clean_df[c] = default_val

    # Re-generate transaction ID if missing
    if "Transaction_ID" not in clean_df.columns:
        clean_df["Transaction_ID"] = [f"TXN-{i+1:05d}" for i in range(len(clean_df))]

    # Final reporting
    quality_report["cleaned_row_count"] = len(clean_df)
    quality_report["missing_after"] = clean_df.isna().sum().to_dict()
    quality_report["date_range"] = (
        clean_df["Date"].min().strftime("%Y-%m-%d"),
        clean_df["Date"].max().strftime("%Y-%m-%d"),
    )
    quality_report["categories"] = sorted(clean_df["Category"].unique().tolist())
    quality_report["total_categories"] = len(quality_report["categories"])
    quality_report["total_expense"] = float(clean_df["Expense_Amount"].sum())

    return clean_df, quality_report


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer pipeline.
    Numerical features -> StandardScaler
    Categorical features -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time-dependent transactions chronologically to prevent data leakage.
    Earliest (1 - test_size) portion -> Train
    Latest test_size portion -> Test
    """
    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1.0 - test_size))
    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()
    return train_df, test_df


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train-test split for classification targets to preserve class proportions.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
