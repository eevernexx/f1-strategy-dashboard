"""
Telemetry Visualization
=======================
All Plotly figure builders for the telemetry page.
Each function returns a go.Figure — no Streamlit calls here.
That separation keeps the viz layer testable and reusable.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import DRIVER_COLORS


# ── Styling defaults ──────────────────────────────────────────────────────────

_BG        = "#0D0D0D"
_BG_PAPER  = "#111111"
_GRID      = "#1E1E1E"
_TEXT      = "#CCCCCC"
_FONT      = "Barlow, system-ui, sans-serif"  # konsisten dengan race_charts.py & app

# Y-axis range default per channel — mencegah autoscale yang inkonsisten
# antar lap/sesi. Nilainya rentang realistic F1.
_CHANNEL_RANGES: dict[str, list[float]] = {
    "Speed":    [0, 360],
    "Throttle": [0, 105],
    "Brake":    [0, 105],
    "Gear":     [0, 9],
    "RPM":      [0, 13500],
}

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
    xaxis=dict(
        gridcolor=_GRID,
        zerolinecolor=_GRID,
        tickfont=dict(size=10),
        title_font=dict(size=11),
    ),
    yaxis=dict(
        gridcolor=_GRID,
        zerolinecolor=_GRID,
        tickfont=dict(size=10),
        title_font=dict(size=11),
    ),
)


def _driver_color(driver: str) -> str:
    return DRIVER_COLORS.get(driver, "#AAAAAA")


# ── Multi-channel telemetry (Speed / Throttle / Brake / Gear / RPM) ──────────

def build_telemetry_multichannel(
    tel_data: dict[str, pd.DataFrame],   # {driver: telemetry_df}
    channels: list[str],
    *,
    corners: pd.DataFrame | None = None,
    sector_distances: list[float] | None = None,
    drs_zones: list[tuple[float, float]] | None = None,
) -> go.Figure:
    """
    Stacked subplot showing multiple telemetry channels vs distance.
    tel_data: dict mapping driver abbreviation → telemetry DataFrame
    channels: subset of ['Speed', 'Throttle', 'Brake', 'Gear', 'RPM']

    Optional overlays:
      corners — DataFrame dari `session.get_circuit_info().corners`
        (kolom Number, Distance). Render T1/T2/... di top subplot.
      sector_distances — [s1_end_dist, s2_end_dist] meter di mana S1 & S2 berakhir
        → render dashed gold line + label.
      drs_zones — [(start_dist, end_dist), ...] → render shaded blue rect.
    """
    n = len(channels)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,  # naik dari 0.03 → kasih ruang title antar subplot
        subplot_titles=[_channel_label(c) for c in channels],
    )

    # Subplot titles: kecilin font + warna lebih lembut supaya tidak dominan
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#888", family=_FONT)

    for driver, tel in tel_data.items():
        color = _driver_color(driver)
        legend_shown = False
        for i, channel in enumerate(channels, start=1):
            if channel not in tel.columns:
                continue

            y = tel[channel]
            # Brake is boolean (0/1) → scale to 0–100 for visual consistency
            if channel == "Brake":
                y = y.astype(float) * 100

            fig.add_trace(
                go.Scatter(
                    x=tel["Distance"],
                    y=y,
                    mode="lines",
                    name=driver,
                    line=dict(color=color, width=1.4),
                    showlegend=not legend_shown,  # one legend entry per driver
                    hovertemplate=(
                        # Kompak 1 baris per trace — Distance sudah jadi header
                        # "x unified", supaya tooltip tidak overflow saat 3 driver
                        # × 5 channel = 15 entries.
                        f"<b>{driver}</b> {_channel_label(channel)}: %{{y:.1f}}"
                        "<extra></extra>"
                    ),
                ),
                row=i, col=1,
            )
            legend_shown = True

    # Apply base layout (tanpa xaxis/yaxis — subplot pakai update_xaxes/yaxes
    # per-row di bawah, jadi nilai global akan ke-override percuma).
    layout_no_axes = {k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **layout_no_axes,
        height=120 + 130 * n,
        title=dict(text="Telemetry comparison", font=dict(size=13), x=0.01),
        hovermode="x unified",
    )

    for i, channel in enumerate(channels, start=1):
        yaxis_kwargs = dict(
            title_text=_channel_label(channel),
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            tickfont=dict(size=9),
            title_font=dict(size=10),
        )
        if channel in _CHANNEL_RANGES:
            yaxis_kwargs["range"] = _CHANNEL_RANGES[channel]
        fig.update_yaxes(**yaxis_kwargs, row=i, col=1)
        fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID, row=i, col=1)

    fig.update_xaxes(title_text="Distance (m)", row=n, col=1)

    # ── Overlays: DRS zones (paling bawah layer) ─────────────────────────────
    if drs_zones:
        for z_start, z_end in drs_zones:
            fig.add_vrect(
                x0=z_start, x1=z_end,
                fillcolor="rgba(34,139,230,0.07)",
                line_width=0,
                layer="below",
                row="all", col=1,
            )
        # DRS label di top subplot saja
        if drs_zones:
            mid = (drs_zones[0][0] + drs_zones[0][1]) / 2
            fig.add_annotation(
                x=mid, y=1.02, xref="x", yref="paper",
                text="<b>DRS</b>",
                showarrow=False,
                font=dict(size=9, color="#3DA9FF", family=_FONT),
            )

    # ── Overlays: Sector splits (vertical dashed gold) ───────────────────────
    if sector_distances:
        for j, sd in enumerate(sector_distances, start=1):
            fig.add_vline(
                x=sd,
                line=dict(color="rgba(255,215,0,0.55)", width=1, dash="dash"),
                row="all", col=1,
            )
            # Label "S2"/"S3" = batas mulai sektor berikutnya
            fig.add_annotation(
                x=sd, y=1.02, xref="x", yref="paper",
                text=f"<b>S{j+1}</b>",
                showarrow=False,
                font=dict(size=10, color="#FFD700", family=_FONT),
            )

    # ── Overlays: Corner labels (T1, T2, ...) di top subplot ─────────────────
    if corners is not None and len(corners) > 0:
        for _, c in corners.iterrows():
            num  = c.get("Number")
            dist = c.get("Distance")
            if pd.isna(num) or pd.isna(dist):
                continue
            # Light vertical guide spanning all subplots
            fig.add_vline(
                x=float(dist),
                line=dict(color="rgba(255,255,255,0.05)", width=0.6),
                row="all", col=1,
            )
            # Label di atas top subplot
            fig.add_annotation(
                x=float(dist), y=1.06, xref="x", yref="paper",
                text=f"T{int(num)}",
                showarrow=False,
                font=dict(size=8, color="#888", family=_FONT),
            )

    return fig


def _channel_label(channel: str) -> str:
    return {
        "Speed":   "Speed (km/h)",
        "Throttle":"Throttle (%)",
        "Brake":   "Brake (%)",
        "Gear":    "Gear",
        "RPM":     "RPM",
        "DRS":     "DRS",
    }.get(channel, channel)


# ── Delta time chart ─────────────────────────────────────────────────────────

def build_delta_time(
    tel_ref: pd.DataFrame,
    tel_cmps: dict[str, pd.DataFrame],
    driver_ref: str,
) -> go.Figure:
    """
    Time delta chart: tiap driver di `tel_cmps` di-plot vs `tel_ref`.
    Positive = cmp lebih lambat dari ref di posisi trek itu.

    Saat 1 cmp: keep visual lama (line putih + green/red fill regions)
                supaya jelas binary fast/slow per region.
    Saat 2+ cmp: line per driver pakai warna driver — lebih bersih kalau
                 ada multi-comparison.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#444", width=1, dash="dot"))

    # Reference distance & time. PENTING: time DI-normalize ke elapsed-from-lap-start
    # supaya delta = 0 di awal lap. Tanpa ini, kalau ref & cmp lapnya dimulai di
    # session-time berbeda, delta-nya bias konstan terlepas dari pace riil.
    dist_ref = tel_ref["Distance"].values - tel_ref["Distance"].values[0]
    t_ref_raw = tel_ref["Time"].dt.total_seconds().values
    t_ref_raw = t_ref_raw - t_ref_raw[0]  # elapsed dari lap start
    idx_ref = np.argsort(dist_ref)

    # Common grid: pakai distance terpendek antara ref & semua cmp
    max_dists = [float(dist_ref.max())]
    for cmp_d, tel_cmp in tel_cmps.items():
        d = tel_cmp["Distance"].values - tel_cmp["Distance"].values[0]
        max_dists.append(float(d.max()))
    common_max = min(max_dists) if max_dists else 0
    grid = np.linspace(0, common_max, 1000)
    t_ref = np.interp(grid, dist_ref[idx_ref], t_ref_raw[idx_ref])

    cmp_list = list(tel_cmps.items())
    n_cmp = len(cmp_list)

    if n_cmp == 1:
        # Visual lama untuk 1-vs-1: white line + green/red regions
        cmp_d, tel_cmp = cmp_list[0]
        dist_cmp = tel_cmp["Distance"].values - tel_cmp["Distance"].values[0]
        t_cmp_raw = tel_cmp["Time"].dt.total_seconds().values
        t_cmp_raw = t_cmp_raw - t_cmp_raw[0]  # elapsed dari lap start
        idx_cmp = np.argsort(dist_cmp)
        t_cmp = np.interp(grid, dist_cmp[idx_cmp], t_cmp_raw[idx_cmp])
        delta = t_cmp - t_ref

        fig.add_trace(go.Scatter(
            x=grid, y=delta,
            mode="lines",
            line=dict(color="#FFFFFF", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,255,255,0.05)",
            hovertemplate=(
                f"<b>{cmp_d}</b> vs {driver_ref}: %{{y:+.3f}}s<br>"
                "Distance: %{x:.0f}m<extra></extra>"
            ),
            name=f"Δ {cmp_d} − {driver_ref}",
        ))
        # Green region kalau cmp lebih cepat
        fig.add_trace(go.Scatter(
            x=grid, y=np.where(delta < 0, delta, 0),
            mode="lines", line=dict(width=0),
            fill="tozeroy", fillcolor="rgba(50,200,100,0.15)",
            showlegend=False, hoverinfo="skip",
        ))
        # Red region kalau cmp lebih lambat
        fig.add_trace(go.Scatter(
            x=grid, y=np.where(delta > 0, delta, 0),
            mode="lines", line=dict(width=0),
            fill="tozeroy", fillcolor="rgba(220,50,50,0.15)",
            showlegend=False, hoverinfo="skip",
        ))
        title = f"Time delta — {cmp_d} vs {driver_ref}"
    else:
        # 2+ comparisons: line per driver pakai warna driver, no fill
        for cmp_d, tel_cmp in cmp_list:
            dist_cmp = tel_cmp["Distance"].values - tel_cmp["Distance"].values[0]
            t_cmp_raw = tel_cmp["Time"].dt.total_seconds().values
            t_cmp_raw = t_cmp_raw - t_cmp_raw[0]  # elapsed dari lap start
            idx_cmp = np.argsort(dist_cmp)
            t_cmp = np.interp(grid, dist_cmp[idx_cmp], t_cmp_raw[idx_cmp])
            delta = t_cmp - t_ref

            color = _driver_color(cmp_d)
            fig.add_trace(go.Scatter(
                x=grid, y=delta,
                mode="lines",
                line=dict(color=color, width=1.6),
                hovertemplate=(
                    f"<b>{cmp_d}</b> vs {driver_ref}: %{{y:+.3f}}s<br>"
                    "Distance: %{x:.0f}m<extra></extra>"
                ),
                name=f"Δ {cmp_d} − {driver_ref}",
            ))
        title = f"Time delta — vs {driver_ref} (reference)"

    fig.update_layout(
        **LAYOUT_BASE,
        height=270,
        title=dict(text=title, font=dict(size=13), x=0.01),
        xaxis_title="Distance (m)",
        yaxis_title="Δ Time (s)",
        hovermode="x unified",
        hoverlabel=dict(font_size=10, namelength=-1),
    )
    return fig


# ── Track map colored by speed ────────────────────────────────────────────────

def build_track_speed_map(tel: pd.DataFrame, driver: str) -> go.Figure:
    """
    X/Y track map with color encoding for speed.
    Uses positional data (X, Y) from FastF1.
    """
    if "X" not in tel.columns or "Y" not in tel.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tel["X"],
        y=tel["Y"],
        mode="markers",
        marker=dict(
            color=tel["Speed"],
            colorscale=[
                [0.0, "#1a0a00"],
                [0.2, "#7f1d1d"],
                [0.5, "#f59e0b"],
                [0.8, "#34d399"],
                [1.0, "#06b6d4"],
            ],
            size=3,
            colorbar=dict(
                title=dict(text="Speed (km/h)", font=dict(size=10)),
                tickfont=dict(size=9),
                thickness=12,
            ),
        ),
        hovertemplate=(
            f"<b>{driver}</b><br>Speed: %{{marker.color:.0f}} km/h<extra></extra>"
        ),
        showlegend=False,
    ))

    layout_no_axes = {k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **layout_no_axes,
        height=400,
        title=dict(text=f"Track map — {driver} speed", font=dict(size=13), x=0.01),
    )
    fig.update_xaxes(visible=False, scaleanchor="y")
    fig.update_yaxes(visible=False)
    return fig


# ── Track dominance map (mini-sector winner) ─────────────────────────────────

def build_track_dominance_map(
    pos_tel_per_driver: dict[str, pd.DataFrame],
    tel_per_driver: dict[str, pd.DataFrame],
    n_sectors: int = 24,
) -> go.Figure | None:
    """
    Track map dengan setiap mini-sector di-color by driver tercepat di sektor itu.

    Cara kerja:
      1. Bagi panjang lap jadi N mini-sectors (default 24).
      2. Untuk tiap sector, hitung waktu tiap driver lewat sector itu (interpolasi
         dari Distance ↔ Time di tel masing-masing).
      3. Driver dengan time terkecil = "winner" sector itu, warna trace di plot
         pakai warna driver tersebut.

    Memerlukan minimal 2 driver yang punya tel + pos data.
    """
    drivers = list(tel_per_driver.keys())
    if len(drivers) < 2:
        return None

    ref_driver = drivers[0]
    ref_pos = pos_tel_per_driver.get(ref_driver)
    if ref_pos is None or "X" not in ref_pos.columns or "Y" not in ref_pos.columns:
        return None

    # Cumulative distance dari (X, Y) — pos_tel kadang tidak punya kolom Distance
    x = ref_pos["X"].values
    y = ref_pos["Y"].values
    if len(x) < 2:
        return None
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    pos_dist = np.cumsum(np.sqrt(dx**2 + dy**2))

    # Distance → Time interpolators per driver
    interps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for d, tel in tel_per_driver.items():
        if "Distance" not in tel.columns or "Time" not in tel.columns or len(tel) < 2:
            continue
        sorted_tel = tel.sort_values("Distance")
        dist = sorted_tel["Distance"].values
        time = sorted_tel["Time"].dt.total_seconds().values
        interps[d] = (dist, time - time[0])

    if len(interps) < 2:
        return None

    max_d = float(pos_dist.max())
    boundaries = np.linspace(0, max_d, n_sectors + 1)

    # Winner per mini-sector
    sector_winners: list[str | None] = []
    for i in range(n_sectors):
        d0, d1 = boundaries[i], boundaries[i + 1]
        best_t = float("inf")
        winner: str | None = None
        for d, (dist_arr, time_arr) in interps.items():
            t0 = float(np.interp(d0, dist_arr, time_arr))
            t1 = float(np.interp(d1, dist_arr, time_arr))
            sector_t = t1 - t0
            if sector_t < best_t:
                best_t = sector_t
                winner = d
        sector_winners.append(winner)

    # Color per pos sample = warna winner mini-sector tempat sample itu berada
    point_colors: list[str] = []
    for d in pos_dist:
        idx = min(int(d / max_d * n_sectors), n_sectors - 1)
        winner = sector_winners[idx]
        point_colors.append(_driver_color(winner) if winner else "#444")

    fig = go.Figure()

    # Background outline track (gelap, biar warna driver muncul kontras di atasnya)
    fig.add_trace(go.Scatter(
        x=ref_pos["X"], y=ref_pos["Y"],
        mode="lines",
        line=dict(color="#1A1A1A", width=10),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Foreground colored markers
    fig.add_trace(go.Scatter(
        x=ref_pos["X"], y=ref_pos["Y"],
        mode="markers",
        marker=dict(color=point_colors, size=4),
        customdata=pos_dist,
        hovertemplate="Distance: %{customdata:.0f}m<extra></extra>",
        showlegend=False,
    ))

    # Legend traces (invisible point per driver) — supaya legenda warna keluar
    for d in interps.keys():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=_driver_color(d), size=10),
            name=d,
            showlegend=True,
        ))

    layout_no_axes = {k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **layout_no_axes,
        height=420,
        title=dict(
            text=f"Track dominance — tercepat per mini-sector ({n_sectors} segments)",
            font=dict(size=13), x=0.01,
        ),
    )
    fig.update_xaxes(visible=False, scaleanchor="y")
    fig.update_yaxes(visible=False)
    return fig


# ── Lap consistency overlay ──────────────────────────────────────────────────

def build_lap_consistency_chart(
    driver: str,
    laps_data: list[dict],
    channel: str = "Speed",
) -> go.Figure | None:
    """
    Overlay channel trace dari N lap tercepat 1 driver — visualize konsistensi.

    laps_data: list of {lap_num, lap_time, tel} sudah disort by lap_time ascending.
    Lap tercepat: solid + opacity 1; lap berikutnya: ghost dengan opacity menurun.
    """
    if not laps_data:
        return None

    # Hover format per channel
    if channel == "Speed":
        y_fmt, unit = "%{y:.0f}", " km/h"
    elif channel in ("Throttle", "Brake"):
        y_fmt, unit = "%{y:.0f}", "%"
    elif channel == "Gear":
        y_fmt, unit = "%{y:.0f}", ""
    elif channel == "RPM":
        y_fmt, unit = "%{y:.0f}", " rpm"
    else:
        y_fmt, unit = "%{y:.1f}", ""

    fig = go.Figure()
    base_color = _driver_color(driver)
    n = len(laps_data)

    for i, ld in enumerate(laps_data):
        tel = ld.get("tel")
        if tel is None or channel not in tel.columns:
            continue
        # Fastest = solid full color; sisa-nya semakin redup sesuai ranking
        opacity = 1.0 if i == 0 else max(0.18, 0.65 - i * 0.09)
        width   = 2.5 if i == 0 else 1.3
        lap_t   = ld["lap_time"]
        lap_t_str = (
            f"{int(lap_t // 60)}:{lap_t % 60:06.3f}" if pd.notna(lap_t) else "—"
        )

        fig.add_trace(go.Scatter(
            x=tel["Distance"],
            y=tel[channel],
            mode="lines",
            name=f"Lap {ld['lap_num']} · {lap_t_str}",
            line=dict(color=base_color, width=width),
            opacity=opacity,
            hovertemplate=(
                # Compact 1-line per trace — supaya tooltip di-x-unified mode
                # tidak overflow saat user pilih max laps (10).
                f"<b>Lap {ld['lap_num']}</b> · {lap_t_str} · {y_fmt}{unit}"
                "<extra></extra>"
            ),
        ))

    # Strip xaxis/yaxis/legend — semua di-override di bawah, kalau di-spread
    # dari LAYOUT_BASE jadi duplicate kwarg di update_layout.
    layout_clean = {
        k: v for k, v in LAYOUT_BASE.items()
        if k not in ("xaxis", "yaxis", "legend")
    }
    fig.update_layout(
        **layout_clean,
        # Naik tinggi proporsional ke n_laps — 10 laps × ~16px line = 160px
        # tooltip + chart area minimal 340px
        height=max(420, 340 + n * 18),
        title=dict(
            text=f"Lap consistency — {driver} fastest {n} laps",
            font=dict(size=13), x=0.01,
        ),
        xaxis_title="Distance (m)",
        yaxis_title=_channel_label(channel),
        hovermode="x unified",
        # Font lebih kecil + namelength=-1 supaya nama lap & lap time tidak terpotong
        hoverlabel=dict(font_size=10, namelength=-1, align="left"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="#333",
            borderwidth=0.5,
            font=dict(size=10),
            # Pindah legend ke samping kalau banyak lap supaya gak makan space chart
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.01,
        ),
    )
    if channel in _CHANNEL_RANGES:
        fig.update_yaxes(range=_CHANNEL_RANGES[channel])
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    return fig


# ── Lap time distribution ─────────────────────────────────────────────────────

def build_lap_time_dist(laps_df: pd.DataFrame, drivers: list[str]) -> go.Figure:
    """
    Horizontal box plot — driver disort by median pace (tercepat di atas).
    Box menunjukkan IQR (25-75%), garis tebal = median, garis dashed = mean,
    titik kecil di luar whisker = outlier laps.

    Visual enhancements:
      - X-axis tick labels formatted as mm:ss.SSS (bukan raw detik)
      - Vertical dashed reference line di fastest median + label
      - Median time per driver di-annotate di samping kanan box
      - Y-axis label: driver name + delta vs fastest
    """
    fig = go.Figure()

    # Compute median lap time per driver (untuk sorting)
    median_pace: dict[str, float] = {}
    for driver in drivers:
        d = laps_df[laps_df["Driver"] == driver]["LapTimeSeconds"].dropna()
        if len(d) >= 1:
            median_pace[driver] = float(d.median())

    if not median_pace:
        fig.add_annotation(
            text="No clean laps available for selected drivers.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="#666", family=_FONT),
        )
        fig.update_layout(
            **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
            height=180,
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    # Sort ascending by median = tercepat duluan
    sorted_drivers = sorted(median_pace.keys(), key=lambda x: median_pace[x])
    fastest_median = median_pace[sorted_drivers[0]]

    def _fmt_lap(s: float) -> str:
        if pd.isna(s):
            return "—"
        if s >= 60:
            m = int(s // 60)
            return f"{m}:{s % 60:06.3f}"
        return f"{s:.3f}"

    # Plotly menempatkan trace pertama paling bawah; reverse supaya tercepat di atas
    for driver in reversed(sorted_drivers):
        d = laps_df[laps_df["Driver"] == driver]["LapTimeSeconds"].dropna()
        if len(d) == 0:
            continue
        color = _driver_color(driver)
        try:
            r, g, b = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
            fillcolor = f"rgba({r},{g},{b},0.22)"
        except ValueError:
            fillcolor = "rgba(170,170,170,0.22)"

        # Y-axis label: driver name + delta + median
        median_v = median_pace[driver]
        delta = median_v - fastest_median
        if delta < 0.001:
            label_extras = (
                f"<span style='color:#FFD700;font-size:10px;font-weight:700'>"
                f"FASTEST</span> "
                f"<span style='color:#888;font-size:10px'>{_fmt_lap(median_v)}</span>"
            )
        else:
            label_extras = (
                f"<span style='color:#999;font-size:10px;font-weight:600'>"
                f"+{delta:.3f}s</span> "
                f"<span style='color:#666;font-size:10px'>{_fmt_lap(median_v)}</span>"
            )
        y_label = f"<b style='color:{color}'>{driver}</b>  {label_extras}"

        fig.add_trace(go.Box(
            x=d,
            name=y_label,
            orientation="h",
            marker=dict(color=color, size=3.5, opacity=0.6, line=dict(width=0)),
            line=dict(color=color, width=1.8),
            fillcolor=fillcolor,
            boxmean=True,             # mean ditampilkan sebagai dashed marker
            boxpoints="outliers",     # cuma outlier yang ditampilin sebagai dot
            whiskerwidth=0.65,
            width=0.55,
            hovertemplate=(
                f"<b>{driver}</b><br>"
                "Lap time: <b>%{x:.3f}s</b>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    n_drivers = len(sorted_drivers)
    layout_no_axes = {k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")}

    # Vertical reference line di fastest median + label
    fig.add_vline(
        x=fastest_median,
        line=dict(color="rgba(255,215,0,0.4)", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=fastest_median, y=1.02, xref="x", yref="paper",
        text=f"<b>Fastest median</b> · {_fmt_lap(fastest_median)}",
        showarrow=False,
        font=dict(size=10, color="#FFD700", family=_FONT),
        xanchor="left", xshift=6,
    )

    fig.update_layout(
        **layout_no_axes,
        height=max(280, 46 * n_drivers + 130),
        title=dict(
            text="Lap time distribution — sorted by median pace",
            font=dict(size=13), x=0.01,
        ),
        boxgap=0.35,
        hoverlabel=dict(font_size=11, namelength=-1),
    )

    # Custom X-ticks: format detik jadi mm:ss.SSS untuk readability
    all_times = laps_df["LapTimeSeconds"].dropna().values
    if len(all_times) > 0:
        vmin, vmax = float(np.min(all_times)), float(np.max(all_times))
        span = vmax - vmin
        if span > 0:
            # Step ~0.5s untuk range pendek, ~1s untuk medium, ~2s untuk panjang
            step = 0.5 if span < 3 else (1.0 if span < 8 else 2.0)
            first_tick = float(np.floor(vmin / step) * step)
            last_tick = float(np.ceil(vmax / step) * step)
            tick_vals = list(np.arange(first_tick, last_tick + step / 2, step))
            tick_text = [_fmt_lap(v) for v in tick_vals]
            fig.update_xaxes(
                title=dict(text="Lap time", font=dict(size=11)),
                gridcolor=_GRID, zerolinecolor=_GRID,
                tickfont=dict(size=10, family="monospace"),
                tickmode="array", tickvals=tick_vals, ticktext=tick_text,
            )
        else:
            fig.update_xaxes(
                title=dict(text="Lap time (s)", font=dict(size=11)),
                gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10),
            )

    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)",   # hide horizontal grid lines — bersih
        zerolinecolor=_GRID,
        tickfont=dict(size=11, family=_FONT),
    )
    return fig
