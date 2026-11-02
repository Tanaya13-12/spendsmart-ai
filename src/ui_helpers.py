"""
SpendSmart AI - UI Helper Functions and Theme Styling
Provides custom CSS injection, KPI metric card generation,
and session-cached data loading for Streamlit.
"""

from typing import Any, Dict, List, Optional
import os
import pandas as pd
import streamlit as st
from src.data_generator import generate_and_save_sample
from src.feature_engineering import create_monthly_spending_profiles, engineer_transaction_features
from src.preprocessing import validate_and_clean_data

# Custom CSS for dark modern fintech aesthetics
CUSTOM_CSS = """
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container background */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }
    
    /* Card Container */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #14b8a6;
        transform: translateY(-2px);
    }
    
    .metric-title {
        font-size: 0.825rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #10b981;
        margin-top: 6px;
    }
    
    /* Academic disclaimer banner */
    .disclaimer-banner {
        background-color: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #14b8a6;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-bottom: 24px;
        border: 1px solid #334155;
    }
    
    /* Custom section header badge */
    .badge-tag {
        display: inline-block;
        background-color: rgba(20, 184, 166, 0.15);
        color: #14b8a6;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    /* Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #334155;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 16px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #14b8a6 !important;
        font-weight: 600;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        color: #ffffff;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #14b8a6 0%, #2dd4bf 100%);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
    }
</style>
"""


def inject_custom_styles():
    """Injects custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_disclaimer():
    """Renders the educational and financial disclaimer banner."""
    st.markdown(
        """
        <div class="disclaimer-banner">
            🛡️ <b>Academic Project Notice:</b> SpendSmart AI provides automated financial analysis, machine learning forecasts,
            and reinforcement learning budget simulations for <b>educational purposes only</b>.
            All estimates and policies do not constitute certified professional financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, subtext: str = "", sub_color: str = "#10b981"):
    """Renders a styled dark KPI metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub" style="color: {sub_color};">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


DATA_SOURCE_SAMPLE = "Sample Financial Dataset (5,500+ records)"
DATA_SOURCE_CUSTOM = "Use my own transaction data (Upload CSV)"


@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Cached data loader that reads CSV, executes validation, cleaning,
    and feature engineering, returning cleaned transactions and monthly profiles.
    """
    raw_df = pd.read_csv(csv_path)
    clean_df, quality_report = validate_and_clean_data(raw_df)
    feat_df = engineer_transaction_features(clean_df)
    monthly_profiles = create_monthly_spending_profiles(feat_df)
    return feat_df, monthly_profiles, quality_report


@st.cache_data(show_spinner=False)
def load_and_preprocess_from_bytes(
    file_bytes: bytes, filename: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Cached data loader for in-memory uploaded CSV bytes.
    """
    import io
    raw_df = pd.read_csv(io.BytesIO(file_bytes))
    clean_df, quality_report = validate_and_clean_data(raw_df)
    feat_df = engineer_transaction_features(clean_df)
    monthly_profiles = create_monthly_spending_profiles(feat_df)
    return feat_df, monthly_profiles, quality_report


def render_sidebar_data_source():
    """
    Renders centralized data source selector in Streamlit sidebar across all pages.
    Guarantees that:
    1. Sample dataset is selected by default on first launch.
    2. If Custom CSV is selected but no file uploaded, friendly guidance is shown and sample data remains active.
    3. If valid CSV is uploaded, it is cleanly validated and used.
    4. Changing data source clears stale ML model results in session_state.
    """
    if "data_source_mode" not in st.session_state:
        st.session_state["data_source_mode"] = DATA_SOURCE_SAMPLE

    st.sidebar.markdown("### 💳 **SpendSmart AI**")
    st.sidebar.caption("AI-Powered Personal Finance & Spending Pattern Analyzer")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Data Source")

    current_idx = 0 if st.session_state.get("data_source_mode") == DATA_SOURCE_SAMPLE else 1

    selected_mode = st.sidebar.radio(
        "Financial Transactions Data:",
        [DATA_SOURCE_SAMPLE, DATA_SOURCE_CUSTOM],
        index=current_idx,
        key="data_source_radio_selector",
        help="SpendSmart AI uses a realistic 5,500+ transaction synthetic financial dataset by default.",
    )

    if selected_mode != st.session_state.get("data_source_mode"):
        st.session_state["data_source_mode"] = selected_mode
        # Invalidate stale model artifacts when data source switches
        st.session_state.pop("rl_results", None)
        st.session_state.pop("p11_rl", None)
        st.session_state.pop("cached_custom_data", None)

    if selected_mode == DATA_SOURCE_CUSTOM:
        uploaded_file = st.sidebar.file_uploader(
            "Upload transactions CSV:",
            type=["csv"],
            key="custom_csv_uploader_widget",
            help="Upload personal financial transactions with columns: Date, Expense_Amount, Category, etc.",
        )
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.getvalue()
                df_custom, profiles_custom, report_custom = load_and_preprocess_from_bytes(
                    file_bytes, uploaded_file.name
                )
                st.session_state["cached_custom_data"] = (df_custom, profiles_custom, report_custom)
                st.sidebar.success(f"✅ Loaded {len(df_custom):,} custom records!")
            except Exception as e:
                st.sidebar.error(f"⚠️ Validation error: {e}")
                st.sidebar.warning("Using built-in sample dataset to keep dashboard functional.")
                st.session_state.pop("cached_custom_data", None)
        else:
            st.session_state.pop("cached_custom_data", None)
            st.sidebar.info("ℹ️ No CSV uploaded yet. Using built-in sample dataset so all analytics and models remain immediately available.")
    else:
        st.sidebar.success("✅ Active: Built-in Sample Financial Dataset (5,506 verified transactions).")
        if st.sidebar.button("🔄 Reset / Regenerate Sample Data", key="btn_regenerate_sample"):
            sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_transactions.csv")
            generate_and_save_sample(output_path=sample_path)
            st.cache_data.clear()
            st.session_state.pop("rl_results", None)
            st.session_state.pop("p11_rl", None)
            st.sidebar.success("Regenerated 5,500+ records with a fresh seed!")
            st.rerun()

    st.sidebar.markdown("---")


def get_active_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Centralized dataset resolver.
    Returns (feat_df, monthly_profiles, quality_report).
    Guarantees that a valid dataset is ALWAYS returned.
    If custom data is selected and valid, returns it; otherwise returns the built-in sample dataset.
    """
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_transactions.csv")
    if not os.path.exists(sample_path):
        generate_and_save_sample(output_path=sample_path)

    # Check if custom data mode is active and has valid data
    if st.session_state.get("data_source_mode") == DATA_SOURCE_CUSTOM:
        if "cached_custom_data" in st.session_state and st.session_state["cached_custom_data"] is not None:
            return st.session_state["cached_custom_data"]

    # Default to built-in sample dataset
    return load_and_preprocess_dataset(sample_path)
