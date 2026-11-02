"""
SpendSmart AI - Page 4: Smart Budget Planner
Implements Practical 11: Reinforcement Learning (Q-Learning) Budget Simulator.
Dynamically balances essential spending, discretionary expenses, and savings targets.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.evaluation import apply_dark_layout, plot_rl_training_curves, THEME_BLUE, THEME_CORAL, THEME_EMERALD, THEME_TEAL
from src.reinforcement_learning import ACTION_NAMES, train_q_learning_agent
from src.ui_helpers import inject_custom_styles, render_disclaimer, render_metric_card, render_sidebar_data_source

st.set_page_config(page_title="Smart Budget Planner | SpendSmart AI", page_icon="🤖", layout="wide")
inject_custom_styles()

st.title("🤖 Smart Budget Planner & Q-Learning Simulator")
st.markdown(
    "Practical 11: Train a tabular **Reinforcement Learning (Q-Learning)** agent to discover adaptive monthly "
    "budget allocations under dynamic spending pressures while **strictly preserving essential expenses**."
)

render_disclaimer()

# Render centralized data source selector in sidebar
render_sidebar_data_source()

# ----------------- SECTION 1: USER BUDGET & ESSENTIAL CONSTRAINTS -----------------
st.subheader("💵 Financial Baseline & Essential Constraints")
st.caption("Define your monthly baseline cash flow parameters. Essential expenses are protected and cannot be underfunded.")

b_col1, b_col2, b_col3, b_col4 = st.columns(4)
with b_col1:
    user_income = st.number_input("Monthly Net Income ($):", min_value=1500.0, max_value=25000.0, value=5800.0, step=100.0)
with b_col2:
    user_rent = st.number_input("Fixed Housing / Rent ($):", min_value=500.0, max_value=10000.0, value=1450.0, step=50.0)
with b_col3:
    user_essentials = st.number_input("Utilities & Groceries ($):", min_value=200.0, max_value=5000.0, value=850.0, step=25.0)
with b_col4:
    user_target_savings = st.number_input("Target Monthly Savings ($):", min_value=100.0, max_value=10000.0, value=1200.0, step=50.0)

total_essentials = user_rent + user_essentials
discretionary_allowance = max(0.0, user_income - total_essentials - user_target_savings)
utilization_pct = (total_essentials / user_income) * 100

st.markdown("---")
u1, u2, u3, u4 = st.columns(4)
with u1:
    render_metric_card("Protected Essentials", f"${total_essentials:,.2f}", f"{utilization_pct:.1f}% of Total Income", sub_color="#3b82f6")
with u2:
    render_metric_card("Discretionary Buffer", f"${discretionary_allowance:,.2f}", "Dining, Shopping, Entertainment")
with u3:
    render_metric_card("Savings Target", f"${user_target_savings:,.2f}", f"{(user_target_savings/user_income*100):.1f}% Target Savings Rate")
with u4:
    valid_budget = (total_essentials + user_target_savings) <= user_income
    status_label = "Feasible Budget" if valid_budget else "Deficit! Reduce Target"
    status_color = "#10b981" if valid_budget else "#f43f5e"
    render_metric_card("Budget Viability", status_label, "Mathematical Constraint Check", sub_color=status_color)

if not valid_budget:
    st.error("⚠️ Warning: Your essential expenses plus target savings exceed total monthly income. Adjust baseline parameters.")

# ----------------- SECTION 2: Q-LEARNING SIMULATOR CONTROLS -----------------
st.markdown("---")
st.subheader("🧠 Tabular Q-Learning Hyperparameter Tuning")
st.markdown(
    """
    The agent updates state-action values using the Bellman optimality equation:
    $$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$$
    """
)

q_col1, q_col2, q_col3, q_col4 = st.columns(4)
with q_col1:
    lr = st.slider("Learning Rate (α):", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Speed of Q-value updates.")
with q_col2:
    gamma = st.slider("Discount Factor (γ):", min_value=0.7, max_value=0.99, value=0.95, step=0.01, help="Value placed on future monthly rewards.")
with q_col3:
    episodes = st.slider("Training Episodes:", min_value=200, max_value=3000, value=1000, step=100, help="Number of simulated 12-month budget cycles.")
with q_col4:
    decay = st.slider("Exploration Decay (ε-decay):", min_value=0.98, max_value=0.999, value=0.995, step=0.001, help="Rate at which agent shifts from exploration to exploitation.")

train_btn = st.button("🚀 Train Q-Learning Budget Agent")

# Train agent or retrieve from session state
if train_btn or "rl_results" not in st.session_state:
    with st.spinner("Simulating environment interactions and updating Q-Table..."):
        rl_res = train_q_learning_agent(
            episodes=episodes,
            learning_rate=lr,
            discount_factor=gamma,
            exploration_decay=decay,
            monthly_income=user_income,
            essential_expenses=total_essentials,
            target_savings=user_target_savings,
        )
        st.session_state["rl_results"] = rl_res
else:
    rl_res = st.session_state["rl_results"]

# ----------------- SECTION 3: RL TRAINING CONVERGENCE CHARTS -----------------
st.markdown("---")
st.subheader("📈 Q-Learning Training Curves")

fig_rl = plot_rl_training_curves(
    episode_rewards=rl_res["episode_rewards"],
    smoothed_rewards=rl_res["smoothed_rewards"],
    epsilon_history=rl_res["epsilon_history"],
    title=f"Reinforcement Learning Training Progression ({rl_res['episodes']} Episodes)",
)
st.plotly_chart(fig_rl, use_container_width=True)

# ----------------- SECTION 4: 12-MONTH SIMULATION TRAJECTORY -----------------
st.markdown("---")
st.subheader("⚔️ Simulation Benchmark: Learned Agent vs. Fixed Rigid Budget")

col_res1, col_res2 = st.columns(2)
with col_res1:
    render_metric_card(
        "Agent Cumulative Savings",
        f"${rl_res['agent_cumulative_savings']:,.2f}",
        "Adaptive Dynamic Actions Taken",
        sub_color="#10b981",
    )
with col_res2:
    render_metric_card(
        "Baseline Cumulative Savings",
        f"${rl_res['baseline_cumulative_savings']:,.2f}",
        "Rigid Fixed Allocations (No Adaptation)",
        sub_color="#94a3b8",
    )

agent_sim_df = pd.DataFrame(rl_res["agent_simulation"])
baseline_sim_df = pd.DataFrame(rl_res["baseline_simulation"])

fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(
    x=agent_sim_df["month"],
    y=agent_sim_df["savings_allocated"].cumsum(),
    mode="lines+markers",
    name="Learned Q-Agent Savings",
    line=dict(color=THEME_EMERALD, width=3),
))
fig_sim.add_trace(go.Scatter(
    x=baseline_sim_df["month"],
    y=baseline_sim_df["savings_allocated"].cumsum(),
    mode="lines+markers",
    name="Fixed Baseline Savings",
    line=dict(color="#475569", width=2, dash="dash"),
))
fig_sim.update_xaxes(title_text="Month of Simulation")
fig_sim.update_yaxes(title_text="Cumulative Savings ($)")
st.plotly_chart(apply_dark_layout(fig_sim, title="12-Month Cumulative Wealth Accumulation"), use_container_width=True)

# ----------------- SECTION 5: LEARNED POLICY & Q-TABLE -----------------
st.markdown("---")
st.subheader("🗺️ Learned Optimal Policy & Q-Table Inspection")
st.markdown("Review the converged state-action values across all 27 financial states:")

q_table_df = pd.DataFrame(rl_res["q_table"])
st.dataframe(q_table_df, use_container_width=True, hide_index=True)

st.download_button(
    label="📥 Download Converged Q-Table CSV",
    data=q_table_df.to_csv(index=False).encode("utf-8"),
    file_name="spendsmart_q_table_policy.csv",
    mime="text/csv",
)
