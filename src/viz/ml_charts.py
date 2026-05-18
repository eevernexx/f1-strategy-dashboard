"""
ML Visualization
================
Chart builders untuk hasil model degradasi ban & pit optimizer:
- Predicted vs actual scatter
- Pit window optimization curve
- Feature importance bar
"""

import pandas as pd
import plotly.graph_objects as go


_BG       = "#0D0D0D"
_BG_PAPER = "#111111"
_GRID     = "#1E1E1E"
_TEXT     = "#CCCCCC"
_FONT     = "Barlow, system-ui, sans-serif"

LAYOUT_BASE = dict(
    paper_bgcolor=_BG_PAPER,
    plot_bgcolor=_BG,
    font=dict(family=_FONT, color=_TEXT, size=12),
    margin=dict(l=60, r=20, t=50, b=40),
)


def build_predicted_vs_actual(
    actual: list[float],
    predicted: list[float],
    r2: float | None = None,
) -> go.Figure | None:
    """Scatter actual lap time vs predicted, dengan garis y=x diagonal."""
    if not actual or not predicted or len(actual) != len(predicted):
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted,
        mode="markers",
        marker=dict(color="#E8002D", size=5, opacity=0.55,
                    line=dict(color="#000", width=0.3)),
        hovertemplate=(
            "Actual: %{x:.3f}s<br>"
            "Predicted: %{y:.3f}s<extra></extra>"
        ),
        showlegend=False,
    ))

    # Diagonal y=x reference line
    if actual:
        lo = min(min(actual), min(predicted))
        hi = max(max(actual), max(predicted))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(color="#FFD700", width=1.5, dash="dash"),
            name="Perfect prediction",
            showlegend=False,
            hoverinfo="skip",
        ))

    title_text = "Predicted vs Actual lap time (test set)"
    if r2 is not None:
        title_text += f"   R² = {r2:.3f}"

    fig.update_layout(
        **LAYOUT_BASE,
        height=400,
        title=dict(text=title_text, font=dict(size=13), x=0.01),
    )
    fig.update_xaxes(
        title=dict(text="Actual lap time (s)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        title=dict(text="Predicted lap time (s)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    return fig


def build_pit_window_chart(
    candidates: list[dict],
    optimal_pit_lap: int,
    baseline_no_pit: float,
) -> go.Figure | None:
    """
    Line chart: x = pit lap, y = total remaining race time (sec).
    Highlight optimal pit lap with marker. Baseline (no pit) sebagai dashed line.
    """
    if not candidates:
        return None

    df = pd.DataFrame(candidates)
    if "pit_lap" not in df.columns or "total_time" not in df.columns:
        return None

    fig = go.Figure()

    # Main pit window curve
    fig.add_trace(go.Scatter(
        x=df["pit_lap"], y=df["total_time"],
        mode="lines+markers",
        line=dict(color="#3DA9FF", width=2),
        marker=dict(size=5, color="#3DA9FF"),
        name="Total time if pit at this lap",
        hovertemplate=(
            "Pit at lap %{x}<br>"
            "Total remaining: %{y:.2f}s<extra></extra>"
        ),
    ))

    # Optimal marker (gold)
    optimal_row = df[df["pit_lap"] == optimal_pit_lap]
    if len(optimal_row) > 0:
        fig.add_trace(go.Scatter(
            x=optimal_row["pit_lap"], y=optimal_row["total_time"],
            mode="markers",
            marker=dict(size=14, color="#FFD700",
                        line=dict(color="#000", width=1.5)),
            name=f"Optimal: lap {optimal_pit_lap}",
            hovertemplate=(
                f"<b>OPTIMAL</b><br>Pit at lap {optimal_pit_lap}<br>"
                "Total: %{y:.2f}s<extra></extra>"
            ),
        ))

    # Baseline (no pit)
    fig.add_hline(
        y=baseline_no_pit,
        line=dict(color="#888", width=1, dash="dot"),
        annotation_text=f"No pit: {baseline_no_pit:.1f}s",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#AAA"),
    )

    fig.update_layout(
        **LAYOUT_BASE,
        height=380,
        title=dict(
            text="Pit window — total remaining race time per pit lap choice",
            font=dict(size=13), x=0.01,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )
    fig.update_xaxes(
        title=dict(text="Pit lap (when to pit)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        title=dict(text="Total remaining race time (s)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    return fig


def build_feature_importance(importance_df: pd.DataFrame) -> go.Figure | None:
    """
    Horizontal bar chart of feature importance.
    Expects DataFrame dengan kolom Feature, Importance, Method.
    """
    if importance_df is None or len(importance_df) == 0:
        return None
    if "Feature" not in importance_df.columns or "Importance" not in importance_df.columns:
        return None

    # Sort ascending → most important at top setelah reverse y-axis
    df = importance_df.sort_values("Importance", ascending=True)
    method = df["Method"].iloc[0] if "Method" in df.columns and len(df) > 0 else "?"

    fig = go.Figure(go.Bar(
        x=df["Importance"], y=df["Feature"],
        orientation="h",
        marker=dict(color="#E8002D", line=dict(color="#000", width=0.5)),
        text=[f"{v:.3f}" for v in df["Importance"]],
        textposition="outside",
        textfont=dict(size=10, color="#AAA"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Importance: %{x:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=max(280, 28 * len(df) + 110),
        title=dict(
            text=f"Feature importance ({method})",
            font=dict(size=13), x=0.01,
        ),
        showlegend=False,
    )
    fig.update_xaxes(
        title=dict(text="Mean |impact| on prediction", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)", zerolinecolor=_GRID,
        tickfont=dict(size=11, family=_FONT),
    )
    return fig
