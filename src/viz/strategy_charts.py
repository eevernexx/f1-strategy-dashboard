"""
Strategy Visualisation
======================
Chart builders untuk page Tyre / Race Strategy:
- Stint breakdown (Gantt-style per driver)
- Compound usage summary
- Pit stop timeline
- Tyre degradation scatter (lap time vs tyre age, per compound)
- Stint pace evolution
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.config import COMPOUND_COLORS, DRIVER_COLORS


# ── Styling defaults ──────────────────────────────────────────────────────────

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
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#333",
        borderwidth=0.5,
        font=dict(size=11),
    ),
)


def _driver_color(driver: str) -> str:
    return DRIVER_COLORS.get(driver, "#AAAAAA")


def _compound_color(compound: str | None) -> str:
    if not isinstance(compound, str):
        return "#888"
    return COMPOUND_COLORS.get(compound.upper(), "#888")


def _fmt_laptime(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    s = float(seconds)
    if s >= 60:
        m = int(s // 60)
        return f"{m}:{s % 60:06.3f}"
    return f"{s:.3f}"


# ── Stint breakdown chart (signature F1 viz) ─────────────────────────────────

def build_stint_breakdown(
    stints_df: pd.DataFrame,
    drivers_order: list[str] | None = None,
) -> go.Figure | None:
    """
    Horizontal Gantt-style per driver: bar berwarna compound, panjang = stint length.
    """
    if stints_df is None or len(stints_df) == 0:
        return None

    if drivers_order:
        # Filter ke driver yang ada di data + preserve order
        in_data = set(stints_df["Driver"].unique())
        drivers = [d for d in drivers_order if d in in_data]
    else:
        drivers = sorted(stints_df["Driver"].unique().tolist())

    if not drivers:
        return None

    fig = go.Figure()

    # Track compound yang sudah ditambah ke legend (avoid duplicate legend entries)
    legend_seen: set[str] = set()

    for driver in drivers:
        drv_stints = stints_df[stints_df["Driver"] == driver].sort_values("LapStart")
        for _, st in drv_stints.iterrows():
            compound = st.get("Compound")
            if not isinstance(compound, str):
                continue
            color = _compound_color(compound)
            try:
                lap_start = int(st["LapStart"])
                lap_end = int(st["LapEnd"])
                laps_count = int(st["Laps"])
            except (TypeError, ValueError):
                continue

            show_in_legend = compound.upper() not in legend_seen
            legend_seen.add(compound.upper())

            fig.add_trace(go.Bar(
                y=[driver],
                x=[laps_count],
                base=[lap_start - 1],  # offset agar bar mulai di lap_start
                orientation="h",
                marker=dict(color=color, line=dict(color="#000", width=0.5)),
                name=compound.title(),
                legendgroup=compound.upper(),
                showlegend=show_in_legend,
                hovertemplate=(
                    f"<b>{driver}</b> · {compound.title()}<br>"
                    f"Lap {lap_start}–{lap_end} ({laps_count} laps)<extra></extra>"
                ),
                text=compound[0].upper() if compound else "",
                textposition="inside",
                textfont=dict(size=10, color="#000",
                              family="Barlow Condensed, sans-serif"),
                insidetextanchor="middle",
            ))

    n_drivers = len(drivers)
    fig.update_layout(
        **LAYOUT_BASE,
        height=max(280, 26 * n_drivers + 130),
        title=dict(
            text="Stint breakdown — compound per driver",
            font=dict(size=13), x=0.01,
        ),
        barmode="overlay",   # bars sudah di-offset via `base`, jangan stack
        bargap=0.25,
    )
    fig.update_xaxes(
        title=dict(text="Lap", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        autorange="reversed",  # driver pertama di atas
        gridcolor="rgba(0,0,0,0)", zerolinecolor=_GRID,
        tickfont=dict(size=11, family=_FONT),
    )
    return fig


# ── Compound usage summary ───────────────────────────────────────────────────

def build_compound_usage(stints_df: pd.DataFrame) -> go.Figure | None:
    """Total laps per compound across drivers — bar chart sorted descending."""
    if stints_df is None or len(stints_df) == 0:
        return None
    if "Compound" not in stints_df.columns or "Laps" not in stints_df.columns:
        return None

    df = stints_df.dropna(subset=["Compound"]).copy()
    if len(df) == 0:
        return None

    summary = (
        df.groupby("Compound")["Laps"].sum()
        .sort_values(ascending=False)
    )
    if len(summary) == 0:
        return None

    fig = go.Figure(go.Bar(
        x=[c.title() if isinstance(c, str) else "?" for c in summary.index],
        y=summary.values,
        marker=dict(
            color=[_compound_color(c) for c in summary.index],
            line=dict(color="#000", width=0.5),
        ),
        text=summary.values,
        textposition="outside",
        textfont=dict(size=11, color="#CCC"),
        hovertemplate="<b>%{x}</b><br>%{y} total laps<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=300,
        title=dict(
            text="Compound usage — total laps across all drivers",
            font=dict(size=13), x=0.01,
        ),
        showlegend=False,
    )
    fig.update_xaxes(
        title=dict(text="Compound", font=dict(size=11)),
        gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11),
    )
    fig.update_yaxes(
        title=dict(text="Total laps", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    return fig


# ── Pit stop timeline ────────────────────────────────────────────────────────

def build_pit_timeline(
    pit_laps_df: pd.DataFrame,
    drivers_order: list[str] | None = None,
    total_laps: int | None = None,
) -> go.Figure | None:
    """
    Marker (diamond) di lap tiap driver pit-in. Y-axis = driver, X = lap.
    `pit_laps_df`: DataFrame dengan kolom Driver, LapNumber, optional Compound.
    """
    if pit_laps_df is None or len(pit_laps_df) == 0:
        return None

    df = pit_laps_df.dropna(subset=["LapNumber", "Driver"]).copy()
    if len(df) == 0:
        return None

    if drivers_order:
        in_data = set(df["Driver"].unique())
        drivers = [d for d in drivers_order if d in in_data]
    else:
        drivers = sorted(df["Driver"].unique().tolist())

    if not drivers:
        return None

    fig = go.Figure()

    for driver in drivers:
        drv_pits = df[df["Driver"] == driver].sort_values("LapNumber")
        if len(drv_pits) == 0:
            continue
        color = _driver_color(driver)
        # Compound after pit (kalau tersedia)
        compounds_after = (
            drv_pits["CompoundAfter"].tolist()
            if "CompoundAfter" in drv_pits.columns else [None] * len(drv_pits)
        )
        hover_texts = []
        for lap, comp in zip(drv_pits["LapNumber"], compounds_after):
            comp_str = f" → {comp.title()}" if isinstance(comp, str) else ""
            hover_texts.append(f"<b>{driver}</b> · Pit at lap {int(lap)}{comp_str}")

        # Marker color = compound after kalau ada, else driver color
        marker_colors = [
            _compound_color(c) if isinstance(c, str) else color
            for c in compounds_after
        ]

        fig.add_trace(go.Scatter(
            x=drv_pits["LapNumber"],
            y=[driver] * len(drv_pits),
            mode="markers",
            marker=dict(
                symbol="diamond", size=11,
                color=marker_colors,
                line=dict(color=color, width=1.5),
            ),
            name=driver,
            showlegend=False,
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=max(280, 22 * len(drivers) + 110),
        title=dict(
            text="Pit stop timeline — diamond color = new compound",
            font=dict(size=13), x=0.01,
        ),
        showlegend=False,
    )
    xax_kwargs = dict(
        title=dict(text="Lap", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    if total_laps is not None and total_laps > 0:
        xax_kwargs["range"] = [0, total_laps + 1]
    fig.update_xaxes(**xax_kwargs)
    fig.update_yaxes(
        autorange="reversed",
        gridcolor="rgba(0,0,0,0)", zerolinecolor=_GRID,
        tickfont=dict(size=11, family=_FONT),
    )
    return fig


# ── Tyre degradation scatter ─────────────────────────────────────────────────

def build_tyre_degradation_scatter(laps_df: pd.DataFrame) -> go.Figure | None:
    """
    Scatter lap time vs tyre age, di-color by compound, plus polynomial trend
    line (degree 2) per compound — visualize degradation curve & cliff.
    """
    needed = ("TyreLife", "LapTimeSeconds", "Compound")
    if any(c not in laps_df.columns for c in needed):
        return None

    df = laps_df.dropna(subset=list(needed)).copy()
    if len(df) == 0:
        return None

    # Filter outliers: lap times > 2 std above median per compound (in/out laps)
    df["_keep"] = True
    for compound, grp in df.groupby("Compound"):
        median = grp["LapTimeSeconds"].median()
        std = grp["LapTimeSeconds"].std()
        if pd.notna(std):
            threshold = median + 2 * std
            df.loc[(df["Compound"] == compound) &
                   (df["LapTimeSeconds"] > threshold), "_keep"] = False
    df = df[df["_keep"]].drop(columns=["_keep"])
    if len(df) == 0:
        return None

    fig = go.Figure()

    for compound in df["Compound"].unique():
        if not isinstance(compound, str):
            continue
        comp_df = df[df["Compound"] == compound]
        if len(comp_df) == 0:
            continue
        color = _compound_color(compound)

        # Scatter samples
        fig.add_trace(go.Scatter(
            x=comp_df["TyreLife"],
            y=comp_df["LapTimeSeconds"],
            mode="markers",
            marker=dict(color=color, size=5, opacity=0.55,
                        line=dict(color="#000", width=0.3)),
            name=compound.title(),
            legendgroup=compound.upper(),
            hovertemplate=(
                f"<b>{compound.title()}</b><br>"
                "Tyre age: %{x:.0f} laps<br>"
                "Lap time: %{y:.3f}s<extra></extra>"
            ),
        ))

        # Polynomial trend line (degree 2) kalau cukup samples
        if len(comp_df) >= 6:
            try:
                x_fit = comp_df["TyreLife"].astype(float).values
                y_fit = comp_df["LapTimeSeconds"].astype(float).values
                coeffs = np.polyfit(x_fit, y_fit, deg=2)
                x_line = np.linspace(x_fit.min(), x_fit.max(), 60)
                y_line = np.polyval(coeffs, x_line)
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines",
                    line=dict(color=color, width=2.5, dash="dash"),
                    name=f"{compound.title()} trend",
                    legendgroup=compound.upper(),
                    showlegend=False,
                    hoverinfo="skip",
                ))
            except Exception:
                pass

    fig.update_layout(
        **LAYOUT_BASE,
        height=440,
        title=dict(
            text="Tyre degradation — lap time vs tyre age",
            font=dict(size=13), x=0.01,
        ),
        hovermode="closest",
        hoverlabel=dict(font_size=10, namelength=-1),
    )
    fig.update_xaxes(
        title=dict(text="Tyre age (laps)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        title=dict(text="Lap time (s)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    return fig


# ── Stint pace evolution ─────────────────────────────────────────────────────

def build_stint_pace_evolution(
    laps_df: pd.DataFrame,
    drivers_filter: list[str] | None = None,
) -> go.Figure | None:
    """
    Tiap line = 1 stint (1 driver × 1 stint), x=TyreLife, y=LapTime.
    Color by compound. Multi-stint dari driver yang sama overlay, jadi mudah
    lihat konsistensi degradation per stint.
    """
    needed = ("LapTimeSeconds", "Compound", "Stint", "TyreLife", "Driver")
    if any(c not in laps_df.columns for c in needed):
        return None

    df = laps_df.dropna(subset=list(needed)).copy()
    if drivers_filter:
        df = df[df["Driver"].isin(drivers_filter)]
    if len(df) == 0:
        return None

    # Filter outlier per (driver, stint) — drop laps > 2 std above stint median
    keep_mask = pd.Series(True, index=df.index)
    for (driver, stint), grp in df.groupby(["Driver", "Stint"]):
        if len(grp) < 3:
            continue
        median = grp["LapTimeSeconds"].median()
        std = grp["LapTimeSeconds"].std()
        if pd.notna(std):
            threshold = median + 2 * std
            keep_mask.loc[grp.index] = grp["LapTimeSeconds"] <= threshold
    df = df[keep_mask]
    if len(df) == 0:
        return None

    fig = go.Figure()

    # Track compound for legend (one entry per compound across all stints)
    legend_seen: set[str] = set()

    for (driver, stint), grp in df.groupby(["Driver", "Stint"]):
        if len(grp) < 2:
            continue
        compound = grp["Compound"].iloc[0]
        if not isinstance(compound, str):
            continue
        color = _compound_color(compound)
        # Sort by TyreLife untuk smooth line
        grp_sorted = grp.sort_values("TyreLife")

        show_legend = compound.upper() not in legend_seen
        legend_seen.add(compound.upper())

        fig.add_trace(go.Scatter(
            x=grp_sorted["TyreLife"],
            y=grp_sorted["LapTimeSeconds"],
            mode="lines+markers",
            line=dict(color=color, width=1.3),
            marker=dict(size=3.5, color=color, line=dict(width=0)),
            opacity=0.55,
            name=compound.title(),
            legendgroup=compound.upper(),
            showlegend=show_legend,
            hovertemplate=(
                f"<b>{driver}</b> · Stint {int(stint)} ({compound.title()})<br>"
                "Tyre age: %{x:.0f} laps<br>"
                "Lap time: %{y:.3f}s<extra></extra>"
            ),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=400,
        title=dict(
            text="Stint pace evolution — each line = one stint",
            font=dict(size=13), x=0.01,
        ),
        hovermode="closest",
        hoverlabel=dict(font_size=10, namelength=-1),
    )
    fig.update_xaxes(
        title=dict(text="Tyre age (laps)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        title=dict(text="Lap time (s)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    return fig
