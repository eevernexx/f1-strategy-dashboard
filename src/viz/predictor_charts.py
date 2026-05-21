"""
Race Predictor Visualization
=============================
Chart builders for the race outcome predictor page.
Dark F1 theme consistent with ml_charts.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.config import DRIVER_COLORS


_BG       = "#0D0D0D"
_BG_PAPER = "#111111"
_GRID     = "#1E1E1E"
_TEXT     = "#CCCCCC"
_FONT     = "Barlow, system-ui, sans-serif"
_FONT_T   = "Barlow Condensed, system-ui, sans-serif"
_ACCENT   = "#E8002D"

LAYOUT_BASE = dict(
    paper_bgcolor=_BG_PAPER,
    plot_bgcolor=_BG,
    font=dict(family=_FONT, color=_TEXT, size=12),
    margin=dict(l=60, r=20, t=50, b=40),
)

_CLS_COLORS = {
    "DNF":     "#888888",
    "Podium":  "#E8002D",
    "Points":  "#FF8000",
    "Outside": "#444444",
}


def build_outcome_probability_bar(proba_df: pd.DataFrame) -> go.Figure | None:
    """Horizontal stacked bar per driver showing outcome probabilities."""
    if proba_df is None or len(proba_df) == 0:
        return None

    df = proba_df.sort_values("prob_podium", ascending=True).reset_index(drop=True)
    drivers = df["driver_code"].tolist()

    fig = go.Figure()
    for col, label, color in [
        ("prob_dnf",     "DNF",     _CLS_COLORS["DNF"]),
        ("prob_podium",  "Podium",  _CLS_COLORS["Podium"]),
        ("prob_points",  "Points",  _CLS_COLORS["Points"]),
        ("prob_outside", "Outside", _CLS_COLORS["Outside"]),
    ]:
        vals = (df[col] * 100).tolist()
        fig.add_trace(go.Bar(
            y=drivers,
            x=vals,
            name=label,
            orientation="h",
            marker=dict(color=color, line=dict(color="#000", width=0.3)),
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}%<extra></extra>",
        ))

    h = max(400, len(drivers) * 28)
    fig.update_layout(
        **LAYOUT_BASE,
        barmode="stack",
        height=h,
        title=dict(
            text="Race Outcome Probabilities",
            font=dict(family=_FONT_T, size=15),
            x=0.01,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )
    fig.update_xaxes(
        title=dict(text="Probability (%)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
        range=[0, 100],
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)", zerolinecolor=_GRID,
        tickfont=dict(size=11, family=_FONT_T),
    )
    return fig


def build_win_probability_bar(proba_df: pd.DataFrame) -> go.Figure | None:
    """Vertical bar chart of podium probability per driver."""
    if proba_df is None or len(proba_df) == 0:
        return None

    df = proba_df.sort_values("prob_podium", ascending=False).reset_index(drop=True)
    drivers = df["driver_code"].tolist()
    vals = (df["prob_podium"] * 100).tolist()
    colors = [DRIVER_COLORS.get(d, _ACCENT) for d in drivers]

    fig = go.Figure(go.Bar(
        x=drivers,
        y=vals,
        marker=dict(color=colors, line=dict(color="#000", width=0.5)),
        hovertemplate="<b>%{x}</b><br>Podium: %{y:.1f}%<extra></extra>",
    ))

    for i, (d, v) in enumerate(zip(drivers, vals)):
        fig.add_annotation(
            x=d, y=v, text=f"{v:.1f}%",
            showarrow=False, yshift=10,
            font=dict(size=9, color="#AAA"),
        )

    fig.update_layout(
        **LAYOUT_BASE,
        height=420,
        title=dict(
            text="Podium Probability by Driver",
            font=dict(family=_FONT_T, size=15),
            x=0.01,
        ),
        showlegend=False,
    )
    fig.update_xaxes(
        gridcolor="rgba(0,0,0,0)", zerolinecolor=_GRID,
        tickfont=dict(size=10, family=_FONT_T),
    )
    fig.update_yaxes(
        title=dict(text="Podium Probability (%)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    return fig


def build_shap_importance_bar(
    shap_values,
    feature_names: list[str],
) -> go.Figure | None:
    """Global feature importance via mean |SHAP| (or raw importances array)."""
    if shap_values is None or not feature_names:
        return None

    arr = np.asarray(shap_values)

    if arr.ndim == 3:
        importances = np.abs(arr).mean(axis=(0, 1))
    elif arr.ndim == 2:
        importances = np.abs(arr).mean(axis=0)
    elif arr.ndim == 1:
        importances = arr
    else:
        return None

    if len(importances) != len(feature_names):
        return None

    order = np.argsort(importances)
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = importances[order]

    fig = go.Figure(go.Bar(
        x=sorted_vals,
        y=sorted_names,
        orientation="h",
        marker=dict(color=_ACCENT, line=dict(color="#000", width=0.5)),
        text=[f"{v:.4f}" for v in sorted_vals],
        textposition="outside",
        textfont=dict(size=9, color="#AAA"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=max(320, len(feature_names) * 24 + 100),
        title=dict(
            text="Global Feature Importance (mean |SHAP|)",
            font=dict(family=_FONT_T, size=15),
            x=0.01,
        ),
        showlegend=False,
    )
    fig.update_xaxes(
        title=dict(text="Mean |SHAP value|", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)", zerolinecolor=_GRID,
        tickfont=dict(size=10, family=_FONT),
    )
    return fig


def build_shap_waterfall(
    shap_values,
    expected_value: float,
    feature_names: list[str],
    driver_idx: int,
    class_idx: int = 1,
) -> go.Figure | None:
    """Waterfall chart of SHAP contributions for one driver and one class."""
    if shap_values is None or expected_value is None:
        return None

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        if driver_idx >= arr.shape[0] or class_idx >= arr.shape[1]:
            return None
        vals = arr[driver_idx, class_idx, :]
    elif arr.ndim == 2:
        if driver_idx >= arr.shape[0]:
            return None
        vals = arr[driver_idx, :]
    else:
        return None

    if len(vals) != len(feature_names):
        return None

    top_k = min(10, len(vals))
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    order = order[::-1]

    names = [feature_names[i] for i in order]
    contributions = [vals[i] for i in order]
    colors = ["#27AE60" if v >= 0 else "#E8002D" for v in contributions]

    fig = go.Figure(go.Bar(
        y=names,
        x=contributions,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#000", width=0.5)),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=max(300, top_k * 32 + 80),
        title=dict(
            text=f"SHAP Contributions (base: {expected_value:.3f})",
            font=dict(family=_FONT_T, size=14),
            x=0.01,
        ),
        showlegend=False,
    )
    fig.update_xaxes(
        title=dict(text="SHAP contribution", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)", zerolinecolor=_GRID,
        tickfont=dict(size=10, family=_FONT),
    )
    return fig


def build_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: list[str],
) -> go.Figure | None:
    """Annotated heatmap of the confusion matrix (row-normalized percentages)."""
    if cm is None or len(class_names) == 0:
        return None

    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100

    annotations = []
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            annotations.append(dict(
                x=class_names[j],
                y=class_names[i],
                text=f"{cm_pct[i, j]:.0f}%",
                showarrow=False,
                font=dict(color="#FFFFFF" if cm_pct[i, j] > 30 else "#999999", size=12),
            ))

    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=class_names,
        y=class_names,
        colorscale=[[0, "#111111"], [1, "#E8002D"]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>%{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=400,
        title=dict(
            text="Confusion Matrix (row-normalized %)",
            font=dict(family=_FONT_T, size=15),
            x=0.01,
        ),
        annotations=annotations,
    )
    fig.update_xaxes(
        title=dict(text="Predicted", font=dict(size=11)),
        gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, family=_FONT_T),
        side="bottom",
    )
    fig.update_yaxes(
        title=dict(text="Actual", font=dict(size=11)),
        gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, family=_FONT_T),
        autorange="reversed",
    )
    return fig
