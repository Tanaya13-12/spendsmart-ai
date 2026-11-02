"""
SpendSmart AI - Page 2: Expense Prediction
Interactive forecasting laboratory utilizing Multiple Linear Regression, Polynomial Regression,
Ridge, Lasso, and Ensemble models with chronological train/test split.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from src.evaluation import (
    apply_dark_layout,
    calculate_regression_metrics,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
    THEME_EMERALD,
    THEME_TEAL,
)
from src.preprocessing import chronological_train_test_split
from src.ui_helpers import (
    get_active_dataset,
    inject_custom_styles,
    render_disclaimer,
    render_metric_card,
    render_sidebar_data_source,
)

st.set_page_config(page_title="Expense Prediction | SpendSmart AI", page_icon="🔮", layout="wide")
inject_custom_styles()

st.title("🔮 Predictive Expense Forecasting")
st.markdown(
    "Forecast future transaction and category expenses using supervised regression algorithms. "
    "Evaluated via **chronological train/test splitting** to guarantee zero data leakage."
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

# Prepare feature matrix for regression
# Target: Expense_Amount
expense_only = feat_df[feat_df["Transaction_Type"] != "Income"].copy()
expense_only = expense_only.sort_values("Date").reset_index(drop=True)

# Selected predictor features
numeric_feature_cols = [
    "Income",
    "Budget",
    "Savings",
    "Rolling_7d_Expense",
    "Rolling_30d_Expense",
    "DayOfWeek",
    "IsWeekend",
    "Month",
    "Day",
]

# Ensure no NaNs in predictors
expense_only[numeric_feature_cols] = expense_only[numeric_feature_cols].fillna(0.0)

# Chronological Split (80% Train, 20% Test)
train_df, test_df = chronological_train_test_split(expense_only, test_size=0.2, date_col="Date")

X_train = train_df[numeric_feature_cols]
y_train = train_df["Expense_Amount"]
X_test = test_df[numeric_feature_cols]
y_test = test_df["Expense_Amount"]

# ----------------- SIDEBAR MODEL CONTROLS -----------------
with st.sidebar:
    st.header("⚙️ Model Configuration")
    model_choice = st.selectbox(
        "Select Forecasting Algorithm:",
        [
            "Multiple Linear Regression (Practical 3)",
            "Polynomial Regression [Degree 2] (Practical 4)",
            "Ridge Regression [L2] (Practical 5)",
            "Lasso Regression [L1] (Practical 5)",
            "Random Forest Regressor (Practical 10)",
            "HistGradientBoosting Regressor (Practical 10)",
        ],
        index=0,
    )

    alpha_param = 1.0
    if "Ridge" in model_choice or "Lasso" in model_choice:
        alpha_param = st.slider("Regularization Strength (alpha):", min_value=0.01, max_value=50.0, value=1.0, step=0.1)

    n_trees = 100
    if "Random Forest" in model_choice:
        n_trees = st.slider("Number of Trees (n_estimators):", min_value=20, max_value=200, value=100, step=10)

# Build & fit selected pipeline
if "Multiple Linear" in model_choice:
    pipeline = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
elif "Polynomial" in model_choice:
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
elif "Ridge" in model_choice:
    pipeline = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=alpha_param, random_state=42))])
elif "Lasso" in model_choice:
    pipeline = Pipeline([("scaler", StandardScaler()), ("reg", Lasso(alpha=alpha_param, random_state=42, max_iter=5000))])
elif "Random Forest" in model_choice:
    pipeline = Pipeline([("scaler", StandardScaler()), ("reg", RandomForestRegressor(n_estimators=n_trees, random_state=42))])
else:
    pipeline = Pipeline([("scaler", StandardScaler()), ("reg", HistGradientBoostingRegressor(random_state=42))])

# Fit on training data only
pipeline.fit(X_train, y_train)

# Predictions
y_test_pred = pipeline.predict(X_test)
y_train_pred = pipeline.predict(X_train)

metrics = calculate_regression_metrics(y_test.values, y_test_pred)

# ----------------- METRICS CARDS -----------------
st.subheader(f"📊 Evaluation Metrics: {model_choice.split('(')[0].strip()}")
st.caption(f"Evaluated on {len(X_test):,} unseen chronological test records ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}).")

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("R² Score (Explained Var)", f"{metrics['R2']:.4f}", "Test Performance", sub_color="#10b981" if metrics['R2'] > 0.4 else "#f59e0b")
with m2:
    render_metric_card("RMSE", f"${metrics['RMSE']:.2f}", "Root Mean Squared Error", sub_color="#3b82f6")
with m3:
    render_metric_card("MAE", f"${metrics['MAE']:.2f}", "Mean Absolute Error", sub_color="#14b8a6")
with m4:
    render_metric_card("MAPE", f"{metrics['MAPE']:.1f}%", "Mean Absolute % Error", sub_color="#a855f7")

# ----------------- VISUALIZATIONS -----------------
st.markdown("---")
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    fig_avp = plot_actual_vs_predicted(
        y_actual=y_test.values.tolist()[:300],
        y_pred=y_test_pred.tolist()[:300],
        title="Actual vs. Predicted Expenses (Test Sample)",
    )
    st.plotly_chart(fig_avp, use_container_width=True)

with col_plot2:
    residuals = (y_test.values - y_test_pred).tolist()[:300]
    fig_res = plot_residuals(
        y_pred=y_test_pred.tolist()[:300],
        residuals=residuals,
        title="Residual Diagnostic Plot",
    )
    st.plotly_chart(fig_res, use_container_width=True)

# ----------------- INTERACTIVE FORECASTING SIMULATOR -----------------
st.markdown("---")
st.subheader("🎯 Real-Time What-If Expense Estimator")
st.markdown("Tweak scenario parameters below to compute an instant predicted transaction expense.")

with st.form("prediction_form"):
    p1, p2, p3 = st.columns(3)
    with p1:
        sim_income = st.number_input("Monthly Income ($):", value=float(expense_only["Income"].median()), step=100.0)
        sim_budget = st.number_input("Allocated Budget ($):", value=float(expense_only["Budget"].median()), step=50.0)
        sim_savings = st.number_input("Target Savings ($):", value=float(expense_only["Savings"].median()), step=50.0)
    with p2:
        sim_roll7 = st.number_input("Rolling 7-Day Spend ($):", value=float(expense_only["Rolling_7d_Expense"].median()), step=10.0)
        sim_roll30 = st.number_input("Rolling 30-Day Spend ($):", value=float(expense_only["Rolling_30d_Expense"].median()), step=25.0)
        sim_month = st.slider("Month of Year:", 1, 12, 6)
    with p3:
        sim_day = st.slider("Day of Month:", 1, 31, 15)
        sim_dayofweek = st.selectbox("Day of Week:", [0, 1, 2, 3, 4, 5, 6], format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d])
        sim_isweekend = 1 if sim_dayofweek in [5, 6] else 0
        submit_btn = st.form_submit_button("Compute Predicted Expense")

if submit_btn:
    input_row = pd.DataFrame([{
        "Income": sim_income,
        "Budget": sim_budget,
        "Savings": sim_savings,
        "Rolling_7d_Expense": sim_roll7,
        "Rolling_30d_Expense": sim_roll30,
        "DayOfWeek": sim_dayofweek,
        "IsWeekend": sim_isweekend,
        "Month": sim_month,
        "Day": sim_day,
    }])
    pred_val = float(pipeline.predict(input_row)[0])
    pred_val = max(0.0, pred_val)

    st.success(f"Estimated Transaction Expense: **${pred_val:.2f}**")

# ----------------- BENCHMARK COMPARISON TABLE -----------------
st.markdown("---")
st.subheader("⚖️ Side-by-Side Model Benchmark on Unseen Test Split")

@st.cache_data
def run_all_models_benchmark():
    models_dict = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression (alpha=1.0)": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression (alpha=1.0)": Lasso(alpha=1.0, random_state=42, max_iter=5000),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    rows = []
    for name, m in models_dict.items():
        m.fit(X_tr_s, y_train)
        pred = m.predict(X_te_s)
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        rows.append({
            "Model Name": name,
            "R² Score": round(float(r2), 4),
            "RMSE ($)": round(float(rmse), 2),
            "MAE ($)": round(float(mae), 2),
        })
    return pd.DataFrame(rows)

benchmark_df = run_all_models_benchmark()
st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

# Download predictions CSV
download_df = test_df[["Transaction_ID", "Date", "Category", "Expense_Amount"]].copy()
download_df["Predicted_Amount"] = np.round(y_test_pred, 2)
download_df["Prediction_Residual"] = np.round(y_test.values - y_test_pred, 2)

csv_data = download_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Test Predictions CSV",
    data=csv_data,
    file_name="spendsmart_expense_predictions.csv",
    mime="text/csv",
)
