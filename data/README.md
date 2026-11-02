# SpendSmart AI Dataset Guide

This directory holds the synthetic and user-uploaded datasets for SpendSmart AI.

## Default Sample Dataset: `sample_transactions.csv`

The generated sample dataset contains over 5,000 realistic personal finance transactions spanning 24 months.

### Schema:
- **Transaction_ID**: Unique identifier (e.g., `TXN-00001`).
- **Date**: Transaction timestamp (`YYYY-MM-DD`).
- **Income**: User's monthly baseline income assigned to the transaction date.
- **Expense_Amount**: Transaction amount in dollars/currency.
- **Category**: Primary spending category (e.g., `Groceries`, `Rent`, `Utilities`, `Dining`, `Transport`, `Healthcare`, `Shopping`, `Entertainment`, `Subscriptions`, `Education`).
- **Subcategory**: Specific item or expense subtype (e.g., `Organic Market`, `Coffee Shop`, `Metro Pass`).
- **Payment_Method**: Transaction mechanism (`Credit Card`, `Debit Card`, `Cash`, `Bank Transfer`, `Digital Wallet`).
- **Transaction_Type**: Classification of transaction flow (`Expense`, `Income`, `Transfer`).
- **Budget**: Designated monthly category budget limit.
- **Savings**: User's cumulative or monthly target savings allocation.
- **Merchant_Type**: Merchant category classification (`Supermarket`, `Restaurant`, `Transit Authority`, `Streaming Service`, `Medical Center`, etc.).

### Preprocessing & Data Quality:
- Contains controlled missing values in non-critical columns (e.g. Subcategory or Payment_Method) and numeric fields to demonstrate real-world cleaning, median/mode imputation, and duplicate detection.
- Categorical features are encoded using `OneHotEncoder`.
- Numerical features are normalized using `StandardScaler`.
- Chronological train/test splitting is strictly enforced for time series forecasting to prevent data leakage.
