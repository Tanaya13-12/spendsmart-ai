"""
Automated unit tests for data generator and preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.data_generator import generate_transactions
from src.feature_engineering import create_monthly_spending_profiles, engineer_transaction_features
from src.preprocessing import (
    DataValidationError,
    chronological_train_test_split,
    validate_and_clean_data,
)


def test_synthetic_data_generation_reproducibility():
    """Verify data generator produces exact same output with identical seed."""
    df1 = generate_transactions(n_transactions=500, seed=123, introduce_quality_issues=False)
    df2 = generate_transactions(n_transactions=500, seed=123, introduce_quality_issues=False)

    assert len(df1) == len(df2)
    pd.testing.assert_frame_equal(df1, df2)


def test_schema_and_column_presence():
    """Verify generated dataset contains all required and recommended financial columns."""
    df = generate_transactions(n_transactions=200, seed=42)
    expected_cols = [
        "Transaction_ID", "Date", "Income", "Expense_Amount",
        "Category", "Subcategory", "Payment_Method",
        "Transaction_Type", "Budget", "Savings", "Merchant_Type"
    ]
    for col in expected_cols:
        assert col in df.columns


def test_duplicate_removal_and_missing_imputation():
    """Verify validation cleans duplicates and imputes missing values."""
    raw_df = generate_transactions(n_transactions=400, seed=99, introduce_quality_issues=True)
    clean_df, report = validate_and_clean_data(raw_df)

    assert report["duplicates_removed"] >= 0
    assert clean_df["Expense_Amount"].isna().sum() == 0
    assert clean_df["Date"].isna().sum() == 0
    assert len(clean_df) <= len(raw_df)


def test_schema_validation_error_on_missing_column():
    """Verify DataValidationError is raised if required columns are missing."""
    malformed_df = pd.DataFrame({"Wrong_Col": [1, 2, 3]})
    with pytest.raises(DataValidationError):
        validate_and_clean_data(malformed_df)


def test_chronological_train_test_split_no_leakage():
    """Verify chronological split guarantees training dates precede test dates."""
    df = generate_transactions(n_transactions=300, seed=42)
    clean_df, _ = validate_and_clean_data(df)

    train_df, test_df = chronological_train_test_split(clean_df, test_size=0.2, date_col="Date")

    assert len(train_df) + len(test_df) == len(clean_df)
    assert train_df["Date"].max() <= test_df["Date"].min()


def test_feature_engineering_creates_expected_metrics():
    """Verify engineered attributes (ratios, rolling statistics, calendar flags)."""
    df = generate_transactions(n_transactions=300, seed=42, introduce_quality_issues=False)
    clean_df, _ = validate_and_clean_data(df)
    feat_df = engineer_transaction_features(clean_df)

    assert "Rolling_7d_Expense" in feat_df.columns
    assert "Rolling_30d_Expense" in feat_df.columns
    assert "DayOfWeek" in feat_df.columns
    assert "IsWeekend" in feat_df.columns
    assert "Budget_Utilization" in feat_df.columns

    # Verify monthly profile generation
    profiles = create_monthly_spending_profiles(feat_df)
    assert not profiles.empty
    assert "Spending_Profile" in profiles.columns


def test_load_and_preprocess_from_bytes():
    """Verify in-memory CSV byte array loads, cleans, and builds profiles."""
    from src.ui_helpers import load_and_preprocess_from_bytes

    csv_text = "Date,Expense_Amount,Category,Income,Budget\n2024-01-01,50.0,Groceries,5000,600\n2024-01-02,25.0,Dining,5000,300\n"
    csv_bytes = csv_text.encode("utf-8")

    feat_df, profiles, report = load_and_preprocess_from_bytes(csv_bytes, "test_upload.csv")
    assert len(feat_df) == 2
    assert "Rolling_7d_Expense" in feat_df.columns
    assert report["cleaned_row_count"] == 2
    assert "Groceries" in report["categories"]

