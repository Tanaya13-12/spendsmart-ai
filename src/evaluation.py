"""
SpendSmart AI - Evaluation and Visualization Utilities
Provides evaluation metric computation and polished Plotly visualizations
styled for modern dark UI aesthetics (navy/charcoal backgrounds, teal/emerald accents).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# Styling palette constants
THEME_BG = "#0f172a"        # Deep slate / navy
THEME_CARD_BG = "#1e293b"   # Slate card
THEME_TEXT = "#f8fafc"      # Off-white
THEME_MUTED = "#94a3b8"     # Muted slate
THEME_TEAL = "#14b8a6"      # Teal accent
THEME_EMERALD = "#10b981"   # Emerald green
THEME_BLUE = "#3b82f6"      # Blue accent
THEME_CORAL = "#f43f5e"     # Coral alert / noise
THEME_PURPLE = "#a855f7"    # Purple accent
THEME_AMBER = "#f59e0b"     # Amber accent


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression evaluation metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Safe MAPE
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if np.any(mask) else 0.0

    return {
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0, average="weighted"))
    rec = float(recall_score(y_true, y_pred, zero_division=0, average="weighted"))
    f1 = float(f1_score(y_true, y_pred, zero_division=0, average="weighted"))

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
    }


def apply_dark_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Applies cohesive polished dark theme to Plotly figures."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": THEME_TEXT, "family": "Inter, sans-serif"},
        },
        paper_bgcolor=THEME_CARD_BG,
        plot_bgcolor=THEME_BG,
        font={"color": THEME_MUTED, "family": "Inter, sans-serif"},
        margin=dict(l=40, r=40, t=60, b=40),
        height=height,
        xaxis=dict(
            gridcolor="#334155",
            zerolinecolor="#475569",
            tickfont=dict(color=THEME_MUTED),
            title=dict(font=dict(color=THEME_TEXT)),
        ),
        yaxis=dict(
            gridcolor="#334155",
            zerolinecolor="#475569",
            tickfont=dict(color=THEME_MUTED),
            title=dict(font=dict(color=THEME_TEXT)),
        ),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.7)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color=THEME_TEXT),
        ),
        hoverlabel=dict(
            bgcolor=THEME_BG,
            font_size=12,
            font_family="Inter, sans-serif",
        ),
    )
    return fig


def plot_actual_vs_predicted(
    y_actual: List[float],
    y_pred: List[float],
    title: str = "Actual vs. Predicted Expenses",
) -> go.Figure:
    """Plots scatter of actual vs predicted values with diagonal 45-degree reference line."""
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_actual,
        y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(
            color=THEME_TEAL,
            size=7,
            opacity=0.75,
            line=dict(width=1, color="#0d9488"),
        ),
        hovertemplate="Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>",
    ))

    # Perfect prediction identity line
    min_val = min(min(y_actual), min(y_pred))
    max_val = max(max(y_actual), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Identity (y = x)",
        line=dict(color=THEME_CORAL, dash="dash", width=2),
    ))

    fig.update_xaxes(title_text="Actual Expense ($)")
    fig.update_yaxes(title_text="Predicted Expense ($)")
    return apply_dark_layout(fig, title=title)


def plot_residuals(
    y_pred: List[float],
    residuals: List[float],
    title: str = "Residual Plot (Predicted vs. Residuals)",
) -> go.Figure:
    """Plots residual plot to diagnose homoscedasticity and non-linear patterns."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode="markers",
        name="Residuals",
        marker=dict(
            color=THEME_BLUE,
            size=7,
            opacity=0.7,
            line=dict(width=1, color="#2563eb"),
        ),
        hovertemplate="Predicted: $%{x:.2f}<br>Residual: $%{y:.2f}<extra></extra>",
    ))

    # Zero residual line
    fig.add_hline(y=0, line_dash="dash", line_color=THEME_CORAL, line_width=2)

    fig.update_xaxes(title_text="Predicted Values ($)")
    fig.update_yaxes(title_text="Residual (Actual - Predicted) ($)")
    return apply_dark_layout(fig, title=title)


def plot_confusion_matrix(
    cm: List[List[int]],
    labels: List[str] = ["Within Budget", "Over Budget"],
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Renders annotated confusion matrix heatmap."""
    z = np.array(cm)

    # Annotations
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=str(val),
                    font=dict(color=THEME_TEXT, size=16, family="Inter, sans-serif"),
                    showarrow=False,
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[[0, "#1e293b"], [0.5, "#0f766e"], [1.0, "#14b8a6"]],
        showscale=True,
        hoverongaps=False,
    ))

    fig.update_layout(annotations=annotations)
    fig.update_xaxes(title_text="Predicted Label")
    fig.update_yaxes(title_text="True Label", autorange="reversed")
    return apply_dark_layout(fig, title=title, height=380)


def plot_feature_importance(
    importances: Dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 12,
) -> go.Figure:
    """Renders ranked horizontal bar chart of feature importances."""
    sorted_items = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features = [item[0] for item in sorted_items][::-1]
    values = [item[1] for item in sorted_items][::-1]

    colors = [THEME_TEAL if v >= 0 else THEME_CORAL for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))

    fig.update_xaxes(title_text="Relative Importance / Coefficient")
    fig.update_yaxes(title_text="Feature")
    return apply_dark_layout(fig, title=title, height=400)


def plot_polynomial_comparison(
    poly_data: Dict[str, Any],
    title: str = "Polynomial Regression Degree Comparison (Bias-Variance Trade-off)",
) -> go.Figure:
    """Renders Polynomial Degrees 1, 2, 3 fit curves alongside actual observations."""
    fig = go.Figure()

    # Actual test points
    fig.add_trace(go.Scatter(
        x=poly_data["X_test_sorted"],
        y=poly_data["y_test_sorted"],
        mode="markers",
        name="Observations (Test)",
        marker=dict(color=THEME_MUTED, size=6, opacity=0.6),
    ))

    colors = [THEME_BLUE, THEME_TEAL, THEME_CORAL]
    for i, (name, curve) in enumerate(poly_data["curves"].items()):
        fig.add_trace(go.Scatter(
            x=curve["grid_x"],
            y=curve["grid_y"],
            mode="lines",
            name=name,
            line=dict(color=colors[i % len(colors)], width=2.5),
        ))

    fig.update_xaxes(title_text="Scaled Input Feature")
    fig.update_yaxes(title_text="Expense Amount ($)")
    return apply_dark_layout(fig, title=title)


def plot_2d_projection(
    plot_data: List[Dict[str, Any]],
    x_col: str,
    y_col: str,
    color_col: str,
    title: str = "2D Projection",
) -> go.Figure:
    """Renders 2D scatter plot for PCA, LDA, K-Means, or DBSCAN."""
    df_plot = pd.DataFrame(plot_data)

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_sequence=[THEME_TEAL, THEME_BLUE, THEME_AMBER, THEME_PURPLE, THEME_EMERALD, THEME_CORAL],
        hover_data=df_plot.columns.tolist(),
    )
    return apply_dark_layout(fig, title=title, height=450)


def plot_rl_training_curves(
    episode_rewards: List[float],
    smoothed_rewards: List[float],
    epsilon_history: List[float],
    title: str = "Q-Learning Agent Training Progression",
) -> go.Figure:
    """Plots episodic rewards, moving average, and epsilon decay curves."""
    fig = go.Figure()

    episodes = list(range(1, len(episode_rewards) + 1))

    # Raw reward
    fig.add_trace(go.Scatter(
        x=episodes,
        y=episode_rewards,
        mode="lines",
        name="Episodic Reward",
        line=dict(color="#334155", width=1),
        opacity=0.5,
    ))

    # Smoothed reward
    fig.add_trace(go.Scatter(
        x=episodes,
        y=smoothed_rewards,
        mode="lines",
        name="30-Ep Moving Avg Reward",
        line=dict(color=THEME_EMERALD, width=2.5),
    ))

    # Epsilon decay on secondary y-axis
    fig.add_trace(go.Scatter(
        x=episodes,
        y=epsilon_history,
        mode="lines",
        name="Exploration Rate (ε)",
        line=dict(color=THEME_AMBER, width=2, dash="dot"),
        yaxis="y2",
    ))

    fig.update_layout(
        yaxis2=dict(
            title=dict(text="Exploration Rate (ε)", font=dict(color=THEME_AMBER)),
            overlaying="y",
            side="right",
            range=[0, 1.05],
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=THEME_AMBER),
        ),
    )

    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Cumulative Reward")
    return apply_dark_layout(fig, title=title, height=450)
