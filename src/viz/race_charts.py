"""
Race Overview Visualization
============================
Chart builders for:
- Lap-by-lap position chart
- Gap to leader per lap
- Fastest laps table
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import DRIVER_COLORS


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
    # NOTE: xaxis/yaxis defaults sengaja tidak di sini — setiap chart
    # selalu override-nya, kalau dipasang di sini akan jadi duplicate kwarg
    # saat di-spread `**LAYOUT_BASE` bersama `xaxis=...` di update_layout.
)


def _driver_color(driver: str) -> str:
    return DRIVER_COLORS.get(driver, "#AAAAAA")


def _fmt_laptime(seconds: float) -> str:
    if pd.isna(seconds):
        return "—"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:06.3f}"


# ── Lap-by-lap position chart ─────────────────────────────────────────────────

def build_position_chart(
    laps: pd.DataFrame,
    drivers: list[str],
    *,
    race_events: list[tuple[str, int, int]] | None = None,
    prev_year_laps: pd.DataFrame | None = None,
    prev_year_label: str = "",
) -> go.Figure:
    """
    Line chart: X = lap number, Y = position (inverted so P1 is on top).

    Tooltip strategy: pre-compute ranking sorted per-lap dan inject sebagai
    `customdata` — supaya tooltip menampilkan klasemen lap-itu sorted by
    posisi (bukan trace order yang statis). Pakai `hovermode="closest"` plus
    x-axis spike line untuk indikator visual lap.

    `race_events`: list of (event_type, start_lap, end_lap) untuk shaded bands.
    event_type "SC" / "VSC" / "RED" / "RAIN" — render shaded vrect + label.

    `prev_year_laps`: optional DataFrame of laps dari sesi prev year (sama GP).
        Di-render sebagai dashed ghosted line di belakang trace current year,
        hanya untuk driver yang sama-sama ada di kedua tahun.
    `prev_year_label`: string label untuk legend (e.g., "2023").
    """
    fig = go.Figure()

    # Prev year ghost lines (paling belakang biar tidak overlap dengan current)
    if prev_year_laps is not None and len(prev_year_laps) > 0:
        prev_drivers_set = set(prev_year_laps["Driver"].unique())
        # Hanya tampilkan driver yang ada di kedua tahun (apple-to-apple)
        common = [d for d in drivers if d in prev_drivers_set]
        for driver in common:
            prev_d = prev_year_laps[prev_year_laps["Driver"] == driver].sort_values("LapNumber")
            if len(prev_d) == 0:
                continue
            color = _driver_color(driver)
            fig.add_trace(go.Scatter(
                x=prev_d["LapNumber"],
                y=prev_d["Position"],
                mode="lines",
                name=f"{driver} ({prev_year_label})",
                line=dict(color=color, width=1.2, dash="dot"),
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",  # avoid tooltip clutter
            ))

    # Pre-compute sorted ranking per lap → string siap render di tooltip
    laps_in_chart = laps[laps["Driver"].isin(drivers)]
    rankings_by_lap: dict[int, str] = {}
    for lap, group in laps_in_chart.groupby("LapNumber"):
        sorted_grp = group.sort_values("Position")
        lines = [
            f"<b>{r['Driver']}</b> P{int(r['Position'])}"
            for _, r in sorted_grp.iterrows()
        ]
        rankings_by_lap[int(lap)] = "<br>".join(lines)

    # Trace order pakai final position — efek visual untuk legend & line stack
    last_pos = (
        laps.sort_values("LapNumber").groupby("Driver")["Position"].last()
    )
    drivers_ordered = sorted(drivers, key=lambda d: last_pos.get(d, 999))

    for driver in drivers_ordered:
        d = laps[laps["Driver"] == driver].sort_values("LapNumber")
        if len(d) == 0:
            continue
        color = _driver_color(driver)
        customdata = [rankings_by_lap.get(int(l), "") for l in d["LapNumber"]]

        fig.add_trace(go.Scatter(
            x=d["LapNumber"],
            y=d["Position"],
            mode="lines+markers",
            name=driver,
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            customdata=customdata,
            hovertemplate=(
                "<b>Lap %{x}</b><br>%{customdata}<extra></extra>"
            ),
        ))

    total_laps = int(laps["LapNumber"].max()) if len(laps) > 0 else 60
    max_pos    = int(laps["Position"].max()) if len(laps) > 0 else 20

    fig.update_layout(
        **LAYOUT_BASE,
        height=560,
        title=dict(text="Lap-by-lap positions", font=dict(size=13), x=0.01),
        xaxis=dict(
            title="Lap",
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            range=[1, total_laps],
            tickfont=dict(size=10),
            # Spike line vertikal untuk indikator lap saat hover
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#666",
            spikethickness=1,
            spikedash="dot",
        ),
        yaxis=dict(
            title="Position",
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            autorange="reversed",  # P1 at top
            tickvals=list(range(1, max_pos + 1)),
            ticktext=[f"P{i}" for i in range(1, max_pos + 1)],
            tickfont=dict(size=10),
        ),
        # closest mode: tooltip dari trace terdekat. Karena setiap trace di lap
        # yang sama bawa customdata identik (= klasemen sorted), user lihat
        # ranking yang sama-correct lap apapun yang di-hover.
        hovermode="closest",
        hoverlabel=dict(font_size=10, namelength=-1),
    )

    # ── Race events overlay (SC / VSC / Red flag / Rain bands) ──────────────
    if race_events:
        event_styles = {
            "SC":   ("rgba(255,200,0,0.10)", "#FFC800", "SC"),
            "VSC":  ("rgba(255,200,0,0.05)", "#FFC800", "VSC"),
            "RED":  ("rgba(255,40,40,0.12)", "#FF3030", "RED"),
            "RAIN": ("rgba(0,103,255,0.08)", "#3DA9FF", "RAIN"),
        }
        for ev_type, start_lap, end_lap in race_events:
            fillcolor, label_color, label = event_styles.get(
                ev_type, ("rgba(180,180,180,0.05)", "#999", ev_type)
            )
            try:
                s, e = int(start_lap), int(end_lap)
            except (TypeError, ValueError):
                continue
            if e < s:
                s, e = e, s
            fig.add_vrect(
                x0=s, x1=e,
                fillcolor=fillcolor,
                line_width=0,
                layer="below",
            )
            mid = (s + e) / 2
            fig.add_annotation(
                x=mid, y=1.02, xref="x", yref="paper",
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=9, color=label_color, family=_FONT),
            )

    return fig


# ── Gap to leader ─────────────────────────────────────────────────────────────

def build_gap_to_leader(laps: pd.DataFrame, drivers: list[str]) -> go.Figure:
    """
    Gap to race leader per lap.

    Expected columns: Driver, LapNumber, ElapsedSeconds.
    `ElapsedSeconds` = timestamp end-of-lap dari awal session (= session.laps["Time"]
    yang sudah dikonversi ke detik). PENTING: jangan kirim hasil `get_laps()`
    yang sudah memfilter pit/outlier laps — itu bikin cum time underestimate
    waktu race riil dan gap-nya jadi salah. Pakai raw laps.

    Tooltip: sama seperti `build_position_chart` — ranking di-pre-compute per
    lap, di-inject lewat customdata, render dengan closest-hover supaya
    tooltip sorted leader → backmarker secara dinamis per-lap.
    """
    fig = go.Figure()

    # Elapsed time (= waktu sejak awal session saat driver crossing line di akhir lap)
    elapsed: dict[str, pd.Series] = {}
    for driver in drivers:
        d = laps[laps["Driver"] == driver].sort_values("LapNumber")
        d = d.dropna(subset=["ElapsedSeconds"])
        if len(d) == 0:
            continue
        elapsed[driver] = d.set_index("LapNumber")["ElapsedSeconds"]

    if not elapsed:
        return fig

    all_laps = sorted(set().union(*[s.index for s in elapsed.values()]))

    # Pre-compute gap per (lap, driver) + ranking string per lap
    gap_lookup: dict[int, dict[str, float]] = {}
    rankings_by_lap: dict[int, str] = {}
    for lap in all_laps:
        leader_time = min(
            t[lap] for t in elapsed.values() if lap in t.index
        )
        gaps_at_lap: dict[str, float] = {}
        for drv, t in elapsed.items():
            if lap in t.index:
                gaps_at_lap[drv] = t[lap] - leader_time
        gap_lookup[int(lap)] = gaps_at_lap
        sorted_pairs = sorted(gaps_at_lap.items(), key=lambda kv: kv[1])
        rankings_by_lap[int(lap)] = "<br>".join(
            f"<b>{drv}</b> +{gap:.3f}s" for drv, gap in sorted_pairs
        )

    # Trace order pakai elapsed time di lap terakhir = posisi finish efektif
    last_lap = all_laps[-1]
    drivers_ordered = sorted(
        elapsed.keys(),
        key=lambda d: elapsed[d][last_lap] if last_lap in elapsed[d].index
                       else elapsed[d].iloc[-1] + 1e9,  # DNF → jauh di belakang
    )

    for driver in drivers_ordered:
        t = elapsed[driver]
        color = _driver_color(driver)
        gaps_y = []
        laps_x = []
        customdata = []

        for lap in all_laps:
            if lap not in t.index:
                continue
            gaps_y.append(gap_lookup[int(lap)][driver])
            laps_x.append(lap)
            customdata.append(rankings_by_lap[int(lap)])

        fig.add_trace(go.Scatter(
            x=laps_x,
            y=gaps_y,
            mode="lines",
            name=driver,
            line=dict(color=color, width=1.8),
            customdata=customdata,
            hovertemplate=(
                "<b>Lap %{x}</b><br>%{customdata}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line=dict(color="#444", width=1, dash="dot"))

    fig.update_layout(
        **LAYOUT_BASE,
        height=560,
        title=dict(text="Gap to leader", font=dict(size=13), x=0.01),
        xaxis=dict(
            title="Lap",
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            tickfont=dict(size=10),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#666",
            spikethickness=1,
            spikedash="dot",
        ),
        yaxis=dict(title="Gap (s)", gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10)),
        hovermode="closest",
        hoverlabel=dict(font_size=10, namelength=-1),
    )
    return fig


# ── Fastest laps table ────────────────────────────────────────────────────────

def build_fastest_laps_table(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a clean DataFrame of fastest lap per driver, sorted by lap time.
    Empty input → empty DataFrame (caller should handle the empty case).
    """
    valid = laps.dropna(subset=["LapTimeSeconds"])
    if len(valid) == 0:
        return pd.DataFrame(columns=["Driver", "Fastest Lap", "Gap to Fastest"])

    fastest = (
        valid.groupby("Driver")["LapTimeSeconds"]
        .min()
        .reset_index()
        .sort_values("LapTimeSeconds")
        .reset_index(drop=True)
    )
    fastest.index += 1  # rank starts at 1
    fastest["LapTime"] = fastest["LapTimeSeconds"].apply(_fmt_laptime)

    # Gap to fastest
    fastest_time = fastest["LapTimeSeconds"].iloc[0]
    fastest["Gap"] = fastest["LapTimeSeconds"].apply(
        lambda x: "—" if x == fastest_time else f"+{x - fastest_time:.3f}s"
    )

    return fastest[["Driver", "LapTime", "Gap"]].rename(
        columns={"LapTime": "Fastest Lap", "Gap": "Gap to Fastest"}
    )


# ── Session result / final classification ────────────────────────────────────

def build_session_results(session) -> pd.DataFrame:
    """
    Final classification dari `session.results`.
    Race  → kolom Time/Gap (leader = total race time, sisanya = gap)
    Quali → kolom Best Lap (Q3 → Q2 → Q1, sesuai sesi terjauh yang ditembus)
    Tambah kolom Grid + Δ kalau GridPosition tersedia (race only biasanya).
    """
    res = getattr(session, "results", None)
    if res is None or len(res) == 0:
        return pd.DataFrame()

    df = res.copy().dropna(subset=["Position"]).sort_values("Position")
    if len(df) == 0:
        return pd.DataFrame()

    out: dict = {
        "Pos": df["Position"].astype(int).map(lambda p: f"P{p}").tolist(),
    }
    if "Abbreviation" in df.columns:
        out["Driver"] = df["Abbreviation"].fillna("—").tolist()
    if "TeamName" in df.columns:
        out["Team"] = df["TeamName"].fillna("—").tolist()

    # Grid → Finish delta — biasanya cuma ada di Race, di Quali GridPosition NaN/0
    if "GridPosition" in df.columns and df["GridPosition"].notna().any():
        grid_pos = df["GridPosition"].astype(float)
        final_pos = df["Position"].astype(int)
        # GridPosition 0 di FastF1 = pit lane start, format jadi "PL"
        out["Grid"] = [
            ("PL" if g == 0 else f"P{int(g)}") if pd.notna(g) else "—"
            for g in grid_pos
        ]
        deltas = []
        for g, f in zip(grid_pos, final_pos):
            if pd.isna(g) or g == 0:
                deltas.append("—")
                continue
            d = int(g) - int(f)
            if d > 0:
                deltas.append(f"+{d}")     # gained positions
            elif d < 0:
                deltas.append(f"{d}")       # lost (already has minus)
            else:
                deltas.append("0")
        out["Δ"] = deltas

    # Time/Gap (race) atau Best Lap (quali)
    has_race_time = "Time" in df.columns and df["Time"].notna().any()
    if has_race_time:
        def _fmt_race(td, is_leader: bool) -> str:
            if pd.isna(td):
                return "—"
            s = td.total_seconds()
            if is_leader:
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                return f"{h}:{m:02d}:{s % 60:06.3f}"
            return f"+{s:.3f}s"
        out["Time/Gap"] = [
            _fmt_race(t, int(p) == 1)
            for t, p in zip(df["Time"], df["Position"])
        ]
    elif any(q in df.columns for q in ("Q3", "Q2", "Q1")):
        def _best_q(row) -> str:
            for q in ("Q3", "Q2", "Q1"):
                if q not in df.columns:
                    continue
                t = row[q]
                if pd.notna(t):
                    s = t.total_seconds()
                    return f"{int(s // 60)}:{s % 60:06.3f}"
            return "—"
        out["Best Lap"] = df.apply(_best_q, axis=1).tolist()

    if "Status" in df.columns:
        out["Status"] = df["Status"].fillna("—").tolist()
    if "Points" in df.columns:
        out["Pts"] = df["Points"].fillna(0).astype(int).tolist()

    return pd.DataFrame(out).reset_index(drop=True)


# ── Race pace ranking ────────────────────────────────────────────────────────

def build_race_pace_ranking(
    laps_df: pd.DataFrame,
    drivers_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Average + median clean lap time per driver, sorted by median (terkencang
    dulu). Caller harus passing laps_df hasil `get_laps()` (sudah clean dari
    pit-laps & outliers).
    """
    if "LapTimeSeconds" not in laps_df.columns or len(laps_df) == 0:
        return pd.DataFrame(columns=["Rank", "Driver", "Avg pace", "Median", "Δ vs leader", "Laps"])

    df = laps_df.copy()
    if drivers_filter:
        df = df[df["Driver"].isin(drivers_filter)]
    df = df.dropna(subset=["LapTimeSeconds"])
    if len(df) == 0:
        return pd.DataFrame(columns=["Rank", "Driver", "Avg pace", "Median", "Δ vs leader", "Laps"])

    grp = (
        df.groupby("Driver")["LapTimeSeconds"]
        .agg(mean="mean", median="median", count="count")
        .sort_values("median")
        .reset_index()
    )
    fastest_median = float(grp["median"].iloc[0])
    grp["Rank"] = range(1, len(grp) + 1)
    grp["Avg pace"] = grp["mean"].apply(_fmt_laptime)
    grp["Median"] = grp["median"].apply(_fmt_laptime)
    grp["Δ vs leader"] = grp["median"].apply(
        lambda x: "—" if x == fastest_median else f"+{x - fastest_median:.3f}s"
    )
    grp["Laps"] = grp["count"]
    return grp[["Rank", "Driver", "Avg pace", "Median", "Δ vs leader", "Laps"]]


# ── Top speed leaderboard ────────────────────────────────────────────────────

def build_top_speed_table(
    session,
    drivers_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Top speed per driver dari speed trap (kolom SpeedST = end-of-straight).
    """
    try:
        if "SpeedST" not in session.laps.columns:
            return pd.DataFrame(columns=["Rank", "Driver", "Top Speed"])
        laps = session.laps[["Driver", "SpeedST"]].copy()
    except Exception:
        return pd.DataFrame(columns=["Rank", "Driver", "Top Speed"])

    if drivers_filter:
        laps = laps[laps["Driver"].isin(drivers_filter)]
    laps = laps.dropna(subset=["SpeedST"])
    if len(laps) == 0:
        return pd.DataFrame(columns=["Rank", "Driver", "Top Speed"])

    grp = (
        laps.groupby("Driver")["SpeedST"].max()
        .sort_values(ascending=False).reset_index()
    )
    grp["Rank"] = range(1, len(grp) + 1)
    grp["Top Speed"] = grp["SpeedST"].apply(
        lambda x: f"{x:.0f} km/h" if pd.notna(x) else "—"
    )
    return grp[["Rank", "Driver", "Top Speed"]]


# ── Sector pace ranking ──────────────────────────────────────────────────────

def build_sector_pace_table(
    session,
    drivers_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Min sector time per driver across all session laps (numeric = seconds).
    Sorted by sum of best sectors (ranking).
    Returns numeric DataFrame — caller format & highlight per kebutuhan.
    """
    try:
        cols_needed = ["Driver", "Sector1Time", "Sector2Time", "Sector3Time"]
        if not all(c in session.laps.columns for c in cols_needed):
            return pd.DataFrame()
        laps = session.laps[cols_needed].copy()
    except Exception:
        return pd.DataFrame()

    if drivers_filter:
        laps = laps[laps["Driver"].isin(drivers_filter)]
    if len(laps) == 0:
        return pd.DataFrame()

    rows = []
    for driver, drv_laps in laps.groupby("Driver"):
        s1 = drv_laps["Sector1Time"].dropna().min()
        s2 = drv_laps["Sector2Time"].dropna().min()
        s3 = drv_laps["Sector3Time"].dropna().min()
        rows.append({
            "Driver": driver,
            "S1": s1.total_seconds() if pd.notna(s1) else float("nan"),
            "S2": s2.total_seconds() if pd.notna(s2) else float("nan"),
            "S3": s3.total_seconds() if pd.notna(s3) else float("nan"),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame()
    df["Total"] = df[["S1", "S2", "S3"]].sum(axis=1, min_count=1)
    df = df.sort_values("Total", na_position="last").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


# ── Pit stop summary ─────────────────────────────────────────────────────────

def build_pit_stops_summary(
    session,
    drivers_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Setiap row = 1 pit stop event.
    Detect dari row dengan PitInTime not null. Compound after = lap berikutnya.
    Duration = PitOutTime (lap N+1) - PitInTime (lap N) → total waktu di pit lane.
    """
    try:
        cols_needed = ["Driver", "LapNumber", "PitInTime", "PitOutTime",
                       "Compound", "TyreLife"]
        avail = [c for c in cols_needed if c in session.laps.columns]
        if "PitInTime" not in avail or "LapNumber" not in avail or "Driver" not in avail:
            return pd.DataFrame()
        all_laps = session.laps[avail].copy()
    except Exception:
        return pd.DataFrame()

    if drivers_filter:
        all_laps = all_laps[all_laps["Driver"].isin(drivers_filter)]

    pit_rows = []
    for driver, drv_laps in all_laps.groupby("Driver"):
        drv = drv_laps.sort_values("LapNumber").reset_index(drop=True)
        stop_num = 0
        for i in range(len(drv)):
            row = drv.iloc[i]
            pit_in = row.get("PitInTime")
            if pd.isna(pit_in):
                continue
            stop_num += 1
            compound_in = row.get("Compound") if "Compound" in drv.columns else None
            tyre_age = row.get("TyreLife") if "TyreLife" in drv.columns else None
            compound_out = None
            pit_out = None
            if i + 1 < len(drv):
                next_lap = drv.iloc[i + 1]
                if "Compound" in drv.columns:
                    compound_out = next_lap.get("Compound")
                if "PitOutTime" in drv.columns:
                    pit_out = next_lap.get("PitOutTime")

            # Total time in pit lane (drive in + stop + drive out)
            duration = None
            if pd.notna(pit_in) and pd.notna(pit_out):
                try:
                    duration = float((pit_out - pit_in).total_seconds())
                except Exception:
                    duration = None

            pit_rows.append({
                "Driver":   driver,
                "Stop":     stop_num,
                "Lap":      int(row["LapNumber"]) if pd.notna(row["LapNumber"]) else None,
                "Tyre Off": compound_in.title() if isinstance(compound_in, str) else "—",
                "Age":      int(tyre_age) if pd.notna(tyre_age) else None,
                "Tyre On":  compound_out.title() if isinstance(compound_out, str) else "—",
                "Duration": f"{duration:.1f}s" if duration is not None else "—",
            })

    if not pit_rows:
        return pd.DataFrame()
    df = pd.DataFrame(pit_rows).sort_values(["Lap", "Driver"]).reset_index(drop=True)
    return df


# ── Race pace heatmap (driver × lap, deviation vs lap median) ───────────────

def build_pace_heatmap(
    laps_df: pd.DataFrame,
    drivers_filter: list[str] | None = None,
) -> go.Figure | None:
    """
    Heatmap: rows = driver (sorted by median pace), cols = lap.
    Cell color = lap time deviation dari race-median lap itu.
    Biru = lebih cepat, merah = lebih lambat. Pakai laps_df hasil `get_laps()`
    (sudah clean dari pit/outliers).
    """
    if "LapTimeSeconds" not in laps_df.columns or len(laps_df) == 0:
        return None

    df = laps_df.copy()
    if drivers_filter:
        df = df[df["Driver"].isin(drivers_filter)]
    df = df.dropna(subset=["LapTimeSeconds", "LapNumber"])
    if len(df) == 0:
        return None

    # Sort drivers by median pace (cepat di atas)
    medians = df.groupby("Driver")["LapTimeSeconds"].median().sort_values()
    drivers_order = medians.index.tolist()
    if not drivers_order:
        return None

    pivot = df.pivot_table(
        index="Driver", columns="LapNumber", values="LapTimeSeconds", aggfunc="first"
    ).reindex(drivers_order)

    # Median per lap (across drivers) — center of color scale
    median_per_lap = pivot.median(axis=0)
    deviation = pivot.sub(median_per_lap, axis=1)

    fig = go.Figure(data=go.Heatmap(
        z=deviation.values,
        x=deviation.columns.tolist(),
        y=deviation.index.tolist(),
        colorscale=[
            [0.0, "#0EA5E9"],
            [0.5, "#1A1A1A"],
            [1.0, "#DC2626"],
        ],
        zmid=0,
        zmin=-2.0,
        zmax=2.0,
        colorbar=dict(
            title=dict(text="Δ vs lap median (s)", font=dict(size=10)),
            tickfont=dict(size=9),
            thickness=10,
        ),
        hovertemplate=(
            "<b>%{y}</b> · Lap %{x}<br>"
            "Δ vs median: %{z:+.3f}s<extra></extra>"
        ),
    ))

    n = len(drivers_order)
    layout_no_axes = {k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **layout_no_axes,
        height=max(280, 25 * n + 110),
        title=dict(
            text="Race pace heatmap — Δ vs lap median",
            font=dict(size=13), x=0.01,
        ),
    )
    fig.update_xaxes(
        title=dict(text="Lap", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=9),
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)", zerolinecolor=_GRID, tickfont=dict(size=10),
    )
    return fig


# ── Weather timeline (track temp / air temp / wind / rain) ───────────────────

def build_weather_timeline(session) -> go.Figure | None:
    """
    Stacked subplot: temperature + (wind & rainfall flag) over session time.
    Pakai `session.weather_data` — kalau tidak tersedia, return None.
    """
    try:
        wx = session.weather_data
    except Exception:
        return None
    if wx is None or len(wx) == 0 or "Time" not in wx.columns:
        return None

    # Time as session-minute
    try:
        time_min = wx["Time"].dt.total_seconds() / 60.0
    except Exception:
        return None

    has_temp  = ("TrackTemp" in wx.columns and wx["TrackTemp"].notna().any()) or \
                ("AirTemp"   in wx.columns and wx["AirTemp"].notna().any())
    has_wind  = "WindSpeed" in wx.columns and wx["WindSpeed"].notna().any()
    has_rain  = "Rainfall"  in wx.columns and wx["Rainfall"].any()

    if not (has_temp or has_wind or has_rain):
        return None

    n_rows = 2 if (has_wind or has_rain) else 1
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            ["Temperature"] + (["Wind & rain"] if n_rows == 2 else [])
        ),
    )
    # Soften subplot title font
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#888", family=_FONT)

    # Row 1: temperatures
    if "TrackTemp" in wx.columns and wx["TrackTemp"].notna().any():
        fig.add_trace(go.Scatter(
            x=time_min, y=wx["TrackTemp"], name="Track temp",
            mode="lines", line=dict(color="#FF6B35", width=1.8),
            hovertemplate="Track: %{y:.1f}°C<extra></extra>",
        ), row=1, col=1)
    if "AirTemp" in wx.columns and wx["AirTemp"].notna().any():
        fig.add_trace(go.Scatter(
            x=time_min, y=wx["AirTemp"], name="Air temp",
            mode="lines", line=dict(color="#3DA9FF", width=1.8),
            hovertemplate="Air: %{y:.1f}°C<extra></extra>",
        ), row=1, col=1)

    # Row 2: wind speed + rainfall flag (kalau ada)
    if n_rows == 2:
        if has_wind:
            fig.add_trace(go.Scatter(
                x=time_min, y=wx["WindSpeed"], name="Wind (m/s)",
                mode="lines", line=dict(color="#999", width=1.5),
                hovertemplate="Wind: %{y:.1f} m/s<extra></extra>",
            ), row=2, col=1)
        if has_rain:
            # Rainfall = bool; scale jadi 0-max(wind) untuk visual overlay
            try:
                rain_vals = wx["Rainfall"].astype(float)
                if has_wind:
                    scale_y = float(wx["WindSpeed"].max()) * 0.8
                else:
                    scale_y = 1.0
                rain_y = rain_vals * scale_y
                fig.add_trace(go.Scatter(
                    x=time_min, y=rain_y, name="Rainfall",
                    mode="lines", line=dict(color="#0067FF", width=0),
                    fill="tozeroy", fillcolor="rgba(0,103,255,0.20)",
                    hovertemplate="Rainfall: %{customdata}<extra></extra>",
                    customdata=["YES" if x else "no" for x in rain_vals],
                ), row=2, col=1)
            except Exception:
                pass

    layout_clean = {
        k: v for k, v in LAYOUT_BASE.items()
        if k not in ("xaxis", "yaxis", "legend")
    }
    fig.update_layout(
        **layout_clean,
        height=350 if n_rows == 2 else 240,
        title=dict(text="Weather timeline", font=dict(size=13), x=0.01),
        hovermode="x unified",
        hoverlabel=dict(font_size=10, namelength=-1),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#333", borderwidth=0.5,
            font=dict(size=10),
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )

    fig.update_xaxes(
        title=dict(text="Session time (min)", font=dict(size=11)),
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=9),
        row=n_rows, col=1,
    )
    for r in range(1, n_rows + 1):
        fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=9), row=r, col=1)

    return fig
