"""
Driver Comparison Visualization
================================
Chart builders untuk page Driver Comparison (Mode A — Single Race H2H).
- build_lap_delta_chart  : cumulative time delta (D2 vs D1) per lap
- build_pace_distribution: side-by-side box plot 2 driver
- build_h2h_radar        : radar / spider chart 5 metric normalisasi
- build_stint_compare    : horizontal stint bar 2 driver
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import DRIVER_COLORS, COMPOUND_COLORS


# ── Styling — konsisten dengan race_charts.py & telemetry_charts.py ──────────

_BG       = "#0D0D0D"
_BG_PAPER = "#111111"
_GRID     = "#1E1E1E"
_TEXT     = "#CCCCCC"
_FONT     = "Barlow, system-ui, sans-serif"

LAYOUT_BASE = dict(
    paper_bgcolor=_BG_PAPER,
    plot_bgcolor=_BG,
    font=dict(family=_FONT, color=_TEXT, size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#333",
        borderwidth=0.5,
        font=dict(size=11),
    ),
)


def _driver_color(driver: str) -> str:
    return DRIVER_COLORS.get(driver, "#AAAAAA")


# ── Lap-by-lap cumulative time delta ─────────────────────────────────────────

def build_lap_delta_chart(
    laps: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure | None:
    """
    Cumulative time delta antara 2 driver per lap.
    `laps` harus punya kolom: Driver, LapNumber, Time (timedelta atau Timestamp).

    Pakai `Time` (timestamp end-of-lap dari awal session) langsung — pit stop &
    safety car otomatis ikut terhitung. Delta = elapsed_B − elapsed_A.
    Positif = B lebih lambat (di belakang), negatif = B lebih cepat (di depan).

    Returns None kalau salah satu driver tidak punya lap data yang valid.
    """
    needed = {"Driver", "LapNumber", "Time"}
    if not needed.issubset(laps.columns):
        return None

    df = laps[list(needed)].copy()
    df = df.dropna(subset=["Time", "LapNumber"])
    if len(df) == 0:
        return None

    # Convert Time → elapsed seconds (handle both timedelta & timestamp)
    try:
        df["ElapsedSeconds"] = df["Time"].dt.total_seconds()
    except AttributeError:
        return None

    laps_a = df[df["Driver"] == driver_a].sort_values("LapNumber")
    laps_b = df[df["Driver"] == driver_b].sort_values("LapNumber")

    if len(laps_a) == 0 or len(laps_b) == 0:
        return None

    # Merge on LapNumber — hanya lap yang ada di kedua driver
    merged = pd.merge(
        laps_a[["LapNumber", "ElapsedSeconds"]].rename(
            columns={"ElapsedSeconds": "elapsed_a"}
        ),
        laps_b[["LapNumber", "ElapsedSeconds"]].rename(
            columns={"ElapsedSeconds": "elapsed_b"}
        ),
        on="LapNumber",
        how="inner",
    ).sort_values("LapNumber")

    if len(merged) == 0:
        return None

    merged["Delta"] = merged["elapsed_b"] - merged["elapsed_a"]

    color_a = _driver_color(driver_a)
    color_b = _driver_color(driver_b)

    fig = go.Figure()

    # Zero line = reference (driver A's position)
    fig.add_hline(
        y=0, line=dict(color=color_a, width=1, dash="solid"),
        annotation_text=f"{driver_a} (reference)",
        annotation_position="top right",
        annotation_font=dict(size=10, color=color_a),
    )

    # Driver B delta vs A — fill above/below zero supaya visual ahead/behind clear
    fig.add_trace(go.Scatter(
        x=merged["LapNumber"],
        y=merged["Delta"],
        mode="lines+markers",
        name=f"{driver_b} vs {driver_a}",
        line=dict(color=color_b, width=2),
        marker=dict(size=5, color=color_b),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(color_b)},0.12)",
        hovertemplate=(
            "Lap %{x}<br>"
            f"<b>{driver_b}</b>: %{{y:+.2f}}s vs <b>{driver_a}</b>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=360,
        title=dict(
            text=f"Cumulative time delta — {driver_b} vs {driver_a}",
            font=dict(size=13), x=0.01,
        ),
        xaxis=dict(
            title="Lap",
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Gap (seconds) · negative = ahead",
            gridcolor=_GRID, zerolinecolor="#444",
            tickfont=dict(size=10),
            autorange="reversed",  # negative on top — match intuition "ahead = up"
        ),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


# ── Pace distribution box plot ───────────────────────────────────────────────

def build_pace_distribution(
    clean_laps: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure | None:
    """
    Side-by-side box plot lap time distribution dari clean laps.
    `clean_laps` = output dari get_laps() (pit & outlier sudah difilter).

    Returns None kalau salah satu driver punya <3 clean laps (insufficient).
    """
    needed = {"Driver", "LapTimeSeconds"}
    if not needed.issubset(clean_laps.columns):
        return None

    df = clean_laps[list(needed)].dropna(subset=["LapTimeSeconds"]).copy()
    if len(df) == 0:
        return None

    laps_a = df[df["Driver"] == driver_a]["LapTimeSeconds"]
    laps_b = df[df["Driver"] == driver_b]["LapTimeSeconds"]

    if len(laps_a) < 3 or len(laps_b) < 3:
        return None

    fig = go.Figure()

    for driver, vals in [(driver_a, laps_a), (driver_b, laps_b)]:
        color = _driver_color(driver)
        fig.add_trace(go.Box(
            y=vals,
            name=driver,
            marker=dict(color=color),
            line=dict(color=color),
            fillcolor=f"rgba({_hex_to_rgb(color)},0.25)",
            boxmean="sd",      # show mean + std dev
            boxpoints="outliers",
            jitter=0.3,
            pointpos=0,
            hovertemplate=(
                f"<b>{driver}</b><br>%{{y:.3f}}s<extra></extra>"
            ),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=380,
        title=dict(
            text="Lap time distribution (clean laps)",
            font=dict(size=13), x=0.01,
        ),
        xaxis=dict(
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Lap time (seconds)",
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(size=10),
        ),
        showlegend=False,
    )
    return fig


# ── Radar / spider chart H2H ─────────────────────────────────────────────────

def build_h2h_radar(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    driver_a: str,
    driver_b: str,
) -> go.Figure | None:
    """
    Radar chart 5 metric (atau lebih) — value sudah dinormalisasi 0-1 di caller
    di mana 1 = best of the two.

    `metrics_a` / `metrics_b` = dict {label: float in [0,1]}.
    Kedua dict harus punya keys yang sama (caller jaga itu).
    """
    if not metrics_a or not metrics_b:
        return None
    labels = list(metrics_a.keys())
    if labels != list(metrics_b.keys()):
        return None
    if len(labels) < 3:
        # Radar < 3 axis terlihat aneh
        return None

    vals_a = [metrics_a[k] for k in labels]
    vals_b = [metrics_b[k] for k in labels]

    color_a = _driver_color(driver_a)
    color_b = _driver_color(driver_b)

    fig = go.Figure()

    # Tutup polygon dengan repeat first point
    fig.add_trace(go.Scatterpolar(
        r=vals_a + [vals_a[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=driver_a,
        line=dict(color=color_a, width=2),
        fillcolor=f"rgba({_hex_to_rgb(color_a)},0.22)",
        hovertemplate=f"<b>{driver_a}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b + [vals_b[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=driver_b,
        line=dict(color=color_b, width=2),
        fillcolor=f"rgba({_hex_to_rgb(color_b)},0.22)",
        hovertemplate=f"<b>{driver_b}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=_BG_PAPER,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color=_TEXT, size=12),
        margin=dict(l=40, r=40, t=40, b=40),
        height=420,
        title=dict(
            text="Performance comparison · normalised (1.0 = best of pair)",
            font=dict(size=13), x=0.01,
        ),
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(
                range=[0, 1.05],
                tickfont=dict(size=9, color="#666"),
                gridcolor=_GRID,
                linecolor=_GRID,
                showticklabels=True,
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=_TEXT, family=_FONT),
                gridcolor=_GRID,
                linecolor=_GRID,
            ),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#333",
            borderwidth=0.5,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


# ── Stint comparison bar (2 driver) ──────────────────────────────────────────

def build_stint_compare(
    stints_df: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure | None:
    """
    Horizontal bar comparing tyre stint sequence untuk 2 driver.
    `stints_df` schema: Driver, Stint, Compound, LapStart, LapEnd, Laps.
    """
    if stints_df is None or len(stints_df) == 0:
        return None
    needed = {"Driver", "Stint", "Compound", "LapStart", "LapEnd"}
    if not needed.issubset(stints_df.columns):
        return None

    fig = go.Figure()

    # Render order: top = driver_a, bottom = driver_b
    # (Plotly bar y-axis renders bottom→top → put A last in the loop)
    drivers_ordered = [driver_b, driver_a]
    any_data = False

    legend_seen: set[str] = set()
    for driver in drivers_ordered:
        drv = stints_df[stints_df["Driver"] == driver].sort_values("Stint")
        if len(drv) == 0:
            continue
        any_data = True
        for _, row in drv.iterrows():
            compound = row["Compound"] if isinstance(row.get("Compound"), str) else "UNKNOWN"
            comp_key = compound.upper()
            color = COMPOUND_COLORS.get(comp_key, COMPOUND_COLORS["UNKNOWN"])
            try:
                lap_start = int(row["LapStart"])
                lap_end = int(row["LapEnd"])
            except (TypeError, ValueError):
                continue
            width = max(1, lap_end - lap_start + 1)
            show_legend = comp_key not in legend_seen
            legend_seen.add(comp_key)

            # Text color: gelap untuk compound terang (M/H), terang untuk lainnya
            text_color = "#000" if comp_key in ("MEDIUM", "HARD") else "#FFF"
            fig.add_trace(go.Bar(
                y=[driver],
                x=[width],
                base=[lap_start - 1],   # offset so bar starts at LapStart
                orientation="h",
                marker=dict(
                    color=color,
                    line=dict(color="#000", width=0.5),
                ),
                name=compound.title(),
                legendgroup=comp_key,
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>{driver}</b> · {compound.title()}<br>"
                    f"Lap {lap_start}–{lap_end} ({width} laps)"
                    "<extra></extra>"
                ),
                text=compound[0].upper() if len(compound) > 0 else "?",
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=text_color, size=11, family=_FONT),
            ))

    if not any_data:
        return None

    fig.update_layout(
        **LAYOUT_BASE,
        height=220,
        title=dict(text="Stint comparison", font=dict(size=13), x=0.01),
        barmode="overlay",   # bars sudah di-offset via `base`, jangan stack
        bargap=0.3,
        xaxis=dict(
            title="Lap",
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            tickfont=dict(size=12, family=_FONT),
            gridcolor="rgba(0,0,0,0)",
            zerolinecolor=_GRID,
        ),
    )
    return fig


# ── Cumulative points chart (Mode B) ─────────────────────────────────────────

def build_cumulative_points_chart(
    cum_a: list[tuple[int, float]],
    cum_b: list[tuple[int, float]],
    driver_a: str,
    driver_b: str,
    *,
    sprint_rounds: set[int] | None = None,
    race_names: dict[int, str] | None = None,
) -> go.Figure | None:
    """
    Line chart: cumulative points (race + sprint) per round untuk 2 driver.

    `cum_a` / `cum_b` = list of (round, cumulative_total).
    `sprint_rounds` = set of round numbers yang punya sprint (untuk marker khusus).
    `race_names` = dict {round: GP name} untuk hover tooltip.
    """
    if not cum_a or not cum_b:
        return None

    color_a = _driver_color(driver_a)
    color_b = _driver_color(driver_b)
    sprint_rounds = sprint_rounds or set()
    race_names = race_names or {}

    fig = go.Figure()

    for driver, cum, color in [
        (driver_a, cum_a, color_a),
        (driver_b, cum_b, color_b),
    ]:
        x = [r for r, _ in cum]
        y = [p for _, p in cum]
        # Hover text per-point: round + GP name + cumulative pts
        hover = [
            f"<b>{driver}</b><br>"
            f"R{r} · {race_names.get(r, '—')}<br>"
            f"Total: {p:.0f} pts"
            + (" · 🏁 Sprint" if r in sprint_rounds else "")
            for r, p in cum
        ]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines+markers",
            name=driver,
            line=dict(color=color, width=2.4),
            marker=dict(
                size=[7 if r in sprint_rounds else 5 for r in x],
                color=color,
                symbol=[
                    "diamond" if r in sprint_rounds else "circle"
                    for r in x
                ],
                line=dict(color="#000", width=0.5),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    # Mark sprint rounds dengan tipis di X axis (annotation)
    sprint_anno = []
    if sprint_rounds:
        max_y = max(
            max((p for _, p in cum_a), default=0),
            max((p for _, p in cum_b), default=0),
        )
        for r in sorted(sprint_rounds):
            sprint_anno.append(dict(
                x=r, y=max_y * 1.02,
                text="S", showarrow=False,
                font=dict(size=9, color="#666"),
                yanchor="bottom",
            ))

    # LAYOUT_BASE already defines `legend` — strip out before unpacking
    # supaya tidak double-pass kwarg.
    layout_kwargs = {k: v for k, v in LAYOUT_BASE.items() if k != "legend"}
    fig.update_layout(
        **layout_kwargs,
        height=420,
        title=dict(
            text="Cumulative season points (race + sprint)",
            font=dict(size=13), x=0.01,
        ),
        xaxis=dict(
            title="Round",
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(size=10),
            dtick=1,
        ),
        yaxis=dict(
            title="Cumulative points",
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(size=10),
        ),
        hovermode="x unified",
        annotations=sprint_anno,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#333",
            borderwidth=0.5,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom", y=1.0,
            xanchor="right", x=1.0,
        ),
    )
    return fig


# ── Circuit finish positions chart (Mode C) ──────────────────────────────────

def build_circuit_finish_chart(
    circuit_df: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> go.Figure | None:
    """
    Grouped bar: race finish position per year untuk 2 driver di satu circuit.
    Y-axis dibalik (P1 di atas). DNC / DNF (pos None) → bar di-skip.

    `circuit_df` = output build_circuit_h2h dengan hidden kolom _r_a, _r_b, Year.
    """
    if circuit_df is None or len(circuit_df) == 0:
        return None
    needed = {"Year", "_r_a", "_r_b"}
    if not needed.issubset(circuit_df.columns):
        return None

    years = circuit_df["Year"].tolist()
    pos_a = circuit_df["_r_a"].tolist()
    pos_b = circuit_df["_r_b"].tolist()

    # Kalau tidak ada satu pun posisi valid, tidak usah render
    if all(p is None for p in pos_a) and all(p is None for p in pos_b):
        return None

    color_a = _driver_color(driver_a)
    color_b = _driver_color(driver_b)

    # Text label per bar: "P3" atau "—" untuk None
    def _labels(positions):
        return [f"P{p}" if p is not None else "" for p in positions]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(y) for y in years],
        y=[p if p is not None else None for p in pos_a],
        name=driver_a,
        marker=dict(color=color_a, line=dict(color="#000", width=0.5)),
        text=_labels(pos_a),
        textposition="outside",
        textfont=dict(size=11, color=color_a, family=_FONT),
        hovertemplate=f"<b>{driver_a}</b> %{{x}}<br>Finish: P%{{y}}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[str(y) for y in years],
        y=[p if p is not None else None for p in pos_b],
        name=driver_b,
        marker=dict(color=color_b, line=dict(color="#000", width=0.5)),
        text=_labels(pos_b),
        textposition="outside",
        textfont=dict(size=11, color=color_b, family=_FONT),
        hovertemplate=f"<b>{driver_b}</b> %{{x}}<br>Finish: P%{{y}}<extra></extra>",
    ))

    # Max position untuk range axis (sedikit headroom)
    valid_pos = [p for p in (pos_a + pos_b) if p is not None]
    max_pos = max(valid_pos) if valid_pos else 20

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k != "legend"},
        height=380,
        title=dict(
            text="Race finish position by year",
            font=dict(size=13), x=0.01,
        ),
        barmode="group",
        bargap=0.3,
        bargroupgap=0.1,
        xaxis=dict(
            title="Year",
            gridcolor="rgba(0,0,0,0)", zerolinecolor=_GRID,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Finish position",
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(size=10),
            # Reversed via range order [high, low] → P1 di atas.
            # (Jangan pakai autorange='reversed' bareng range — konflik.)
            range=[max_pos + 1.5, 0.5],
            dtick=2,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#333", borderwidth=0.5,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom", y=1.0,
            xanchor="right", x=1.0,
        ),
    )
    return fig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' → 'R,G,B' string untuk pakai di rgba()."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "170,170,170"
    try:
        return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"
    except ValueError:
        return "170,170,170"
