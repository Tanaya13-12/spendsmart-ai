"""
SpendSmart AI - Synthetic Financial Data Generator
Produces realistic, reproducible personal transaction datasets covering 12-24 months.
"""

from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd


def generate_transactions(
    n_transactions: int = 5500,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
    introduce_quality_issues: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic personal transactions with realistic spending patterns,
    recurring expenses, weekend dining surges, seasonal peaks, and controlled quality anomalies.
    """
    rng = np.random.default_rng(seed)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    # Base financial profiles
    base_monthly_income = 5800.0
    monthly_savings_target = 1200.0

    categories = [
        "Groceries",
        "Rent",
        "Transport",
        "Utilities",
        "Healthcare",
        "Education",
        "Shopping",
        "Entertainment",
        "Dining",
        "Subscriptions",
    ]

    category_weights = [
        0.24,  # Groceries
        0.04,  # Rent (monthly recurring)
        0.14,  # Transport
        0.06,  # Utilities (monthly recurring)
        0.05,  # Healthcare
        0.04,  # Education
        0.13,  # Shopping
        0.09,  # Entertainment
        0.16,  # Dining
        0.05,  # Subscriptions
    ]

    subcategories_map = {
        "Groceries": ["Supermarket", "Farmers Market", "Bakery", "Wholesale Club"],
        "Rent": ["Apartment Rent", "Parking Space Fee"],
        "Transport": ["Fuel", "Subway Pass", "Rideshare", "Bus Ticket", "Car Maintenance"],
        "Utilities": ["Electricity", "Water & Sewage", "High-Speed Internet", "Cooking Gas"],
        "Healthcare": ["Pharmacy Prescription", "Dental Checkup", "Doctor Consultation", "Lab Tests"],
        "Education": ["Online Course Subscription", "Textbooks", "Workshop Fee", "Certification Exam"],
        "Shopping": ["Apparel", "Electronics", "Home Decor", "Personal Care", "Footwear"],
        "Entertainment": ["Movie Tickets", "Concert Pass", "Gaming Purchase", "Bowling Night"],
        "Dining": ["Coffee Shop", "Fast Food", "Fine Dining", "Casual Bistro", "Food Delivery"],
        "Subscriptions": ["Music Streaming", "Video Streaming", "Cloud Storage", "Gym Membership", "News App"],
    }

    merchants_map = {
        "Groceries": ["FreshMart", "Trader Green", "Target Super", "Costco Wholesale", "Local Bodega"],
        "Rent": ["Metropolis Properties LLC", "CityCenter Leasing"],
        "Transport": ["Shell Station", "Chevron", "Metro Transit Authority", "Uber", "Lyft"],
        "Utilities": ["ConEdison Power", "City Water Board", "Spectrum Internet", "National Grid"],
        "Healthcare": ["CVS Health", "Walgreens", "City Urgent Care", "Smile Dental Clinic"],
        "Education": ["Coursera Inc", "Udemy", "Campus Bookstore", "EdX Global"],
        "Shopping": ["Amazon", "Zara", "Apple Store", "Nike Outlet", "Best Buy", "H&M"],
        "Entertainment": ["AMC Theatres", "LiveNation", "Steam Games", "Bowlero"],
        "Dining": ["Starbucks", "Chipotle", "Olive Garden", "Sweetgreen", "DoorDash", "Blue Bottle"],
        "Subscriptions": ["Netflix", "Spotify", "Apple iCloud", "Equinox Fitness", "NYTimes Digital"],
    }

    payment_methods = ["Credit Card", "Debit Card", "Cash", "Bank Transfer", "Digital Wallet"]
    payment_weights = [0.48, 0.28, 0.08, 0.06, 0.10]

    category_budgets = {
        "Groceries": 650.0,
        "Rent": 1450.0,
        "Transport": 350.0,
        "Utilities": 220.0,
        "Healthcare": 200.0,
        "Education": 150.0,
        "Shopping": 450.0,
        "Entertainment": 250.0,
        "Dining": 400.0,
        "Subscriptions": 80.0,
    }

    records = []

    # 1. Generate core recurring monthly transactions (Rent on 1st, Utilities on 5th, Salary on 1st and 15th)
    current_month_dt = start_dt
    while current_month_dt <= end_dt:
        # Salary Income (bi-weekly)
        for sal_day in [1, 15]:
            sal_date = current_month_dt.replace(day=min(sal_day, 28))
            records.append({
                "Date": sal_date.strftime("%Y-%m-%d"),
                "Income": base_monthly_income,
                "Expense_Amount": 0.0,
                "Category": "Income",
                "Subcategory": "Bi-Weekly Salary",
                "Payment_Method": "Bank Transfer",
                "Transaction_Type": "Income",
                "Budget": base_monthly_income,
                "Savings": monthly_savings_target,
                "Merchant_Type": "Employer Payroll",
            })

        # Rent
        records.append({
            "Date": current_month_dt.replace(day=1).strftime("%Y-%m-%d"),
            "Income": base_monthly_income,
            "Expense_Amount": float(rng.uniform(1400.0, 1500.0)),
            "Category": "Rent",
            "Subcategory": "Apartment Rent",
            "Payment_Method": "Bank Transfer",
            "Transaction_Type": "Expense",
            "Budget": category_budgets["Rent"],
            "Savings": monthly_savings_target,
            "Merchant_Type": "Landlord",
        })

        # Utilities
        for u_sub in ["Electricity", "High-Speed Internet", "Water & Sewage"]:
            records.append({
                "Date": current_month_dt.replace(day=rng.integers(5, 10)).strftime("%Y-%m-%d"),
                "Income": base_monthly_income,
                "Expense_Amount": float(rng.uniform(40.0, 110.0)),
                "Category": "Utilities",
                "Subcategory": u_sub,
                "Payment_Method": "Debit Card",
                "Transaction_Type": "Expense",
                "Budget": category_budgets["Utilities"],
                "Savings": monthly_savings_target,
                "Merchant_Type": "Utility Company",
            })

        # Subscriptions
        for sub_name in ["Netflix", "Spotify", "Gym Membership"]:
            records.append({
                "Date": current_month_dt.replace(day=rng.integers(12, 18)).strftime("%Y-%m-%d"),
                "Income": base_monthly_income,
                "Expense_Amount": float(rng.uniform(12.0, 45.0)),
                "Category": "Subscriptions",
                "Subcategory": "Media & Fitness",
                "Payment_Method": "Credit Card",
                "Transaction_Type": "Expense",
                "Budget": category_budgets["Subscriptions"],
                "Savings": monthly_savings_target,
                "Merchant_Type": "Streaming Service",
            })

        # Advance to next month
        year = current_month_dt.year + (current_month_dt.month // 12)
        month = (current_month_dt.month % 12) + 1
        current_month_dt = datetime(year, month, 1)

    # 2. Generate random day-to-day transactions to reach total target
    remaining_count = max(n_transactions - len(records), 1000)

    for _ in range(remaining_count):
        # Pick random date
        random_days = rng.integers(0, total_days)
        txn_date = start_dt + timedelta(days=int(random_days))
        is_weekend = txn_date.weekday() in [4, 5, 6]  # Fri, Sat, Sun
        month = txn_date.month

        # Weighted category selection with weekend dining/entertainment bias
        adj_weights = list(category_weights)
        if is_weekend:
            adj_weights[8] *= 1.8  # Dining surge
            adj_weights[7] *= 1.6  # Entertainment surge
            adj_weights[6] *= 1.3  # Shopping surge

        # Normalize weights
        adj_weights = [w / sum(adj_weights) for w in adj_weights]
        cat = rng.choice(categories, p=adj_weights)

        # Base expense distribution depending on category
        if cat == "Groceries":
            amt = rng.gamma(shape=4.0, scale=20.0)  # ~$80 mean
        elif cat == "Dining":
            base = 45.0 if is_weekend else 22.0
            amt = rng.exponential(scale=base) + 6.0
        elif cat == "Shopping":
            amt = rng.lognormal(mean=4.1, sigma=0.6)  # ~$75-$150
            if month in [11, 12]:  # Holiday season surge
                amt *= 1.4
        elif cat == "Transport":
            amt = rng.uniform(5.0, 65.0)
        elif cat == "Healthcare":
            amt = rng.choice([rng.uniform(15.0, 45.0), rng.uniform(80.0, 280.0)], p=[0.7, 0.3])
        elif cat == "Education":
            amt = rng.uniform(25.0, 180.0)
        elif cat == "Entertainment":
            amt = rng.uniform(18.0, 120.0)
        elif cat == "Subscriptions":
            amt = rng.uniform(9.99, 29.99)
        elif cat == "Rent":
            amt = rng.uniform(1400.0, 1550.0)
        else:
            amt = rng.uniform(35.0, 140.0)

        # Rare large expense / anomaly (e.g., laptop purchase, emergency care, flight ticket)
        if rng.random() < 0.015:
            amt = float(rng.uniform(650.0, 2400.0))

        subcat = rng.choice(subcategories_map.get(cat, ["General"]))
        merchant = rng.choice(merchants_map.get(cat, ["Retailer"]))
        pay_method = rng.choice(payment_methods, p=payment_weights)

        # Income variability across months (promotions, bonus)
        user_income = base_monthly_income
        if month == 12:  # Year-end bonus
            user_income += 1200.0
        elif month == 6:  # Mid-year performance
            user_income += 500.0

        records.append({
            "Date": txn_date.strftime("%Y-%m-%d"),
            "Income": float(user_income),
            "Expense_Amount": float(round(amt, 2)),
            "Category": cat,
            "Subcategory": subcat,
            "Payment_Method": pay_method,
            "Transaction_Type": "Expense",
            "Budget": category_budgets.get(cat, 300.0),
            "Savings": monthly_savings_target,
            "Merchant_Type": merchant,
        })

    df = pd.DataFrame(records)

    # Sort chronologically
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Assign Sequential Transaction ID
    df["Transaction_ID"] = [f"TXN-{i+1:05d}" for i in range(len(df))]

    # Order columns logically
    cols = [
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
    df = df[cols]

    # Controlled data quality issues to test and demonstrate cleaning/preprocessing
    if introduce_quality_issues:
        n_rows = len(df)

        # 1. Missing values in non-critical columns (~2% in Subcategory, ~1.5% in Payment_Method)
        nan_subcat_idx = rng.choice(n_rows, size=int(n_rows * 0.02), replace=False)
        df.loc[nan_subcat_idx, "Subcategory"] = np.nan

        nan_pay_idx = rng.choice(n_rows, size=int(n_rows * 0.015), replace=False)
        df.loc[nan_pay_idx, "Payment_Method"] = np.nan

        # Missing values in Expense_Amount (~0.8%)
        nan_amt_idx = rng.choice(n_rows, size=int(n_rows * 0.008), replace=False)
        # Avoid zeroing out Income transactions
        nan_amt_idx = [i for i in nan_amt_idx if df.loc[i, "Transaction_Type"] == "Expense"]
        df.loc[nan_amt_idx, "Expense_Amount"] = np.nan

        # 2. Add duplicate transactions (e.g., 6 duplicate rows at random places)
        dup_indices = rng.choice(n_rows, size=6, replace=False)
        duplicates = df.iloc[dup_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)

    return df


def generate_and_save_sample(
    output_path: str = "data/sample_transactions.csv",
    n_transactions: int = 5500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate and save the sample dataset to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_transactions(n_transactions=n_transactions, seed=seed)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_transactions.csv")
    df = generate_and_save_sample(output_path=out_file)
    print(f"Generated sample dataset with {len(df)} records at {out_file}")
