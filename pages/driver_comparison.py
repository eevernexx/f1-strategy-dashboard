"""
Driver Comparison Page
======================
Head-to-head 2 driver dengan 3 mode filter.

Mode A — Single Race H2H (FastF1 session):
- Year + GP + Driver 1 + Driver 2
- Banner cards, Quali H2H (Q1/Q2/Q3 + gap), Race H2H (finish/points/FL/status)
- Lap-by-lap cumulative time delta, pace distribution box, stint compare
- Radar: pace, consistency, fastest lap, top speed, sector wins

Mode B — Season H2H (Ergast):
- Year + Driver 1 + Driver 2
- Summary cards (points/wins/podiums/poles/FL/DNF), cumulative points chart,
  win-loss tally, per-GP H2H table. Points = race + sprint.

Mode C — Circuit history H2H (Ergast):
- Circuit + Driver 1 + Driver 2, lintas musim 2022-2024
- Per-year quali/race detail, career stats, finish-position-by-year chart
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.pipeline.loader import (
    load_session,
    get_laps,
    get_available_rounds,
    get_session_drivers,
)
from src.pipeline.season_loader import (
    get_season_drivers as load_season_drivers,
    get_season_sprints,
    build_driver_season_summary,
    build_per_gp_h2h,
    compute_h2h_tally,
    get_circuits_for_years,
    build_circuit_h2h,
    build_circuit_career_stats,
)
from src.viz.comparison_charts import (
    build_lap_delta_chart,
    build_pace_distribution,
    build_h2h_radar,
    build_stint_compare,
    build_cumulative_points_chart,
    build_circuit_finish_chart,
)
from src.utils.config import (
    DRIVER_COLORS,
    SESSION_LABELS,
    SUPPORTED_YEARS,
)


# ── Format helpers ───────────────────────────────────────────────────────────

def _fmt_laptime(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    s = float(seconds)
    if s >= 60:
        m = int(s // 60)
        return f"{m}:{s % 60:06.3f}"
    return f"{s:.3f}"


def _fmt_timedelta(td) -> str:
    """Convert pandas Timedelta → mm:ss.fff string."""
    if td is None or pd.isna(td):
        return "—"
    try:
        s = td.total_seconds()
    except AttributeError:
        try:
            s = float(td)
        except (TypeError, ValueError):
            return "—"
    return _fmt_laptime(s)


def _fmt_pos(p) -> str:
    if pd.isna(p):
        return "—"
    try:
        return f"P{int(p)}"
    except (TypeError, ValueError):
        return "—"


def _fmt_grid(p) -> str:
    """Grid position — 0 = pit lane."""
    if pd.isna(p):
        return "—"
    try:
        v = int(p)
    except (TypeError, ValueError):
        return "—"
    return "PL" if v == 0 else f"P{v}"


# ── Driver lookup helpers ────────────────────────────────────────────────────

def _driver_row(session, driver: str) -> pd.Series | None:
    """
    Return session.results row untuk driver tertentu, or None.
    Match by Abbreviation (string driver code, e.g. 'VER').
    """
    try:
        res = session.results
        if res is None or len(res) == 0 or "Abbreviation" not in res.columns:
            return None
        match = res[res["Abbreviation"] == driver]
        if len(match) == 0:
            return None
        return match.iloc[0]
    except Exception:
        return None


def _team_for_driver(session, driver: str) -> str | None:
    """Try session.results first, then session.laps; return None if both fail."""
    row = _driver_row(session, driver)
    if row is not None:
        team = row.get("TeamName")
        if pd.notna(team):
            return str(team)
    # Fallback ke laps
    try:
        drv_laps = session.laps[session.laps["Driver"] == driver]
        if len(drv_laps) > 0 and "Team" in drv_laps.columns:
            teams = drv_laps["Team"].dropna()
            if len(teams) > 0:
                return str(teams.iloc[0])
    except Exception:
        pass
    return None


# ── Race H2H stats ───────────────────────────────────────────────────────────

def _race_stats(session, driver: str) -> dict:
    """Per-driver race summary stats."""
    out: dict = {
        "grid": None, "finish": None, "points": None, "status": None,
        "fastest_lap": None, "fastest_lap_num": None,
        "median_lap": None, "std_lap": None,
        "top_speed": None, "best_s1": None, "best_s2": None, "best_s3": None,
        "n_clean_laps": 0,
    }

    row = _driver_row(session, driver)
    if row is not None:
        out["grid"]    = row.get("GridPosition")
        out["finish"]  = row.get("Position")
        out["points"]  = row.get("Points")
        out["status"]  = (
            str(row.get("Status")) if pd.notna(row.get("Status")) else None
        )

    # Lap-level stats — pakai session.laps (raw)
    try:
        drv_laps = session.laps[session.laps["Driver"] == driver].copy()
    except Exception:
        drv_laps = pd.DataFrame()

    if len(drv_laps) > 0:
        # Fastest lap (raw, ignoring outliers — pakai LapTime asli)
        try:
            valid = drv_laps[drv_laps["LapTime"].notna()].copy()
            if len(valid) > 0:
                valid["s"] = valid["LapTime"].dt.total_seconds()
                idx = valid["s"].idxmin()
                out["fastest_lap"]     = float(valid.loc[idx, "s"])
                out["fastest_lap_num"] = (
                    int(valid.loc[idx, "LapNumber"])
                    if pd.notna(valid.loc[idx, "LapNumber"]) else None
                )
        except Exception:
            pass

        # Top speed (SpeedST = end-of-main-straight speed trap)
        try:
            if "SpeedST" in drv_laps.columns:
                spd = drv_laps["SpeedST"].dropna()
                if len(spd) > 0:
                    out["top_speed"] = float(spd.max())
        except Exception:
            pass

        # Best sector times
        try:
            for sec_col, out_key in [
                ("Sector1Time", "best_s1"),
                ("Sector2Time", "best_s2"),
                ("Sector3Time", "best_s3"),
            ]:
                if sec_col in drv_laps.columns:
                    secs = drv_laps[sec_col].dropna()
                    if len(secs) > 0:
                        try:
                            out[out_key] = float(secs.min().total_seconds())
                        except AttributeError:
                            pass
        except Exception:
            pass

    return out


def _clean_lap_stats(clean_laps: pd.DataFrame, driver: str) -> dict:
    """Median & std lap time from filtered clean laps."""
    out = {"median_lap": None, "std_lap": None, "n_clean_laps": 0}
    if clean_laps is None or len(clean_laps) == 0:
        return out
    if "Driver" not in clean_laps.columns or "LapTimeSeconds" not in clean_laps.columns:
        return out
    drv = clean_laps[
        (clean_laps["Driver"] == driver)
        & clean_laps["LapTimeSeconds"].notna()
    ]
    if len(drv) == 0:
        return out
    out["n_clean_laps"] = len(drv)
    out["median_lap"]   = float(drv["LapTimeSeconds"].median())
    if len(drv) >= 2:
        std = drv["LapTimeSeconds"].std()
        if pd.notna(std):
            out["std_lap"] = float(std)
    return out


# ── Quali H2H (separate session) ─────────────────────────────────────────────

def _quali_stats(quali_session, driver: str) -> dict:
    """Q1/Q2/Q3 times + best + final position dari session.results."""
    out = {
        "q1": None, "q2": None, "q3": None,
        "best": None, "position": None,
    }
    if quali_session is None:
        return out
    row = _driver_row(quali_session, driver)
    if row is None:
        return out

    for col, key in [("Q1", "q1"), ("Q2", "q2"), ("Q3", "q3")]:
        if col in row.index:
            v = row.get(col)
            if pd.notna(v):
                try:
                    out[key] = float(v.total_seconds())
                except AttributeError:
                    pass

    # Best = min dari Q1/Q2/Q3
    candidates = [v for v in (out["q1"], out["q2"], out["q3"]) if v is not None]
    if candidates:
        out["best"] = min(candidates)

    pos = row.get("Position")
    if pd.notna(pos):
        try:
            out["position"] = int(pos)
        except (TypeError, ValueError):
            pass

    return out


# ── Stints (mirror dari tyre_strategy._stints_df) ────────────────────────────

def _stints_df_for_pair(session, drivers: list[str]) -> pd.DataFrame:
    """Stint summary filtered ke 2 driver yang dipilih."""
    try:
        needed = ["Driver", "LapNumber", "Compound", "Stint"]
        if not all(c in session.laps.columns for c in needed):
            return pd.DataFrame()
        laps = session.laps[needed].copy()
    except Exception:
        return pd.DataFrame()

    laps = laps.dropna(subset=["Compound", "Stint", "LapNumber", "Driver"])
    laps = laps[laps["Driver"].isin(drivers)]
    if len(laps) == 0:
        return pd.DataFrame()

    stints = (
        laps.groupby(["Driver", "Stint", "Compound"])
        .agg(
            LapStart=("LapNumber", "min"),
            LapEnd=("LapNumber", "max"),
            Laps=("LapNumber", "count"),
        )
        .reset_index()
        .sort_values(["Driver", "Stint"])
        .reset_index(drop=True)
    )
    return stints


# ── Radar metric computation ─────────────────────────────────────────────────

def _radar_metrics(
    stats_a: dict, stats_b: dict,
    clean_a: dict, clean_b: dict,
) -> tuple[dict[str, float], dict[str, float], dict]:
    """
    Build normalized 5-axis radar metrics + raw values for caption.

    Normalisasi: untuk tiap metric, 'better' driver = 1.0, other = ratio
    (worse/better untuk "higher better" metric, atau best/worst untuk inverted).
    Selalu rasio min/max → naturally 0-1 dengan 1 = best.

    Returns (norm_a, norm_b, raw_pair_dict).
    """
    labels = [
        "Race pace",
        "Consistency",
        "Fastest lap",
        "Top speed",
        "Sector wins",
    ]

    # Raw values — (a_value, b_value, lower_is_better)
    raw_pairs: list[tuple[float | None, float | None, bool]] = []

    # 1. Race pace (median lap, lower better)
    raw_pairs.append((clean_a.get("median_lap"), clean_b.get("median_lap"), True))

    # 2. Consistency (std lap, lower better)
    raw_pairs.append((clean_a.get("std_lap"), clean_b.get("std_lap"), True))

    # 3. Fastest lap (lower better)
    raw_pairs.append((stats_a.get("fastest_lap"), stats_b.get("fastest_lap"), True))

    # 4. Top speed (higher better)
    raw_pairs.append((stats_a.get("top_speed"), stats_b.get("top_speed"), False))

    # 5. Sector wins (best sectors won out of 3, higher better)
    #    Driver yang lebih cepat di sektor itu dapat 1 poin.
    sec_wins_a = 0
    sec_wins_b = 0
    sec_counted = 0
    for sec_key in ("best_s1", "best_s2", "best_s3"):
        sa = stats_a.get(sec_key)
        sb = stats_b.get(sec_key)
        if sa is None or sb is None:
            continue
        sec_counted += 1
        if sa < sb:
            sec_wins_a += 1
        elif sb < sa:
            sec_wins_b += 1
        # tie = nobody scores
    # Kalau tidak ada sektor valid sama sekali, biarkan None → di-handle bawah
    if sec_counted > 0:
        raw_pairs.append((float(sec_wins_a), float(sec_wins_b), False))
    else:
        raw_pairs.append((None, None, False))

    norm_a: dict[str, float] = {}
    norm_b: dict[str, float] = {}
    raw_out: dict[str, tuple] = {}

    for label, (va, vb, lower_better) in zip(labels, raw_pairs):
        raw_out[label] = (va, vb, lower_better)
        if va is None or vb is None:
            # Hilangkan axis kalau data tidak ada
            continue
        if va <= 0 and vb <= 0:
            continue
        if lower_better:
            # Better = min. Better gets 1.0, other = better/other ≤ 1.
            best = min(va, vb)
            if best <= 0:
                continue
            na = best / va if va > 0 else 0.0
            nb = best / vb if vb > 0 else 0.0
        else:
            # Better = max. Better gets 1.0, other = other/best ≤ 1.
            best = max(va, vb)
            if best <= 0:
                # Both zero → score 0/0 — give both 0
                na = 0.0
                nb = 0.0
            else:
                na = va / best
                nb = vb / best
        # Floor di 0.05 supaya radar polygon tidak hilang ke titik origin
        norm_a[label] = max(0.05, min(1.0, na))
        norm_b[label] = max(0.05, min(1.0, nb))

    return norm_a, norm_b, raw_out


# ── Banner card renderer ─────────────────────────────────────────────────────

def _render_driver_card(driver: str, team: str | None, role: str = ""):
    """Render single driver header card."""
    color = DRIVER_COLORS.get(driver, "#888888")
    team_str = team if team else "—"
    role_html = (
        f"<div style='color:#666;font-size:9px;font-weight:700;"
        f"letter-spacing:0.18em;text-transform:uppercase;"
        f"font-family:Barlow Condensed,sans-serif;margin-bottom:4px'>{role}</div>"
    ) if role else ""

    st.markdown(
        f"""
        <div style='border-left:4px solid {color};
                   padding:14px 18px;margin:6px 0;
                   background:rgba(255,255,255,0.02);
                   border-radius:0 4px 4px 0'>
            {role_html}
            <div style='display:flex;align-items:baseline;gap:14px;flex-wrap:wrap'>
                <span style='color:{color};font-size:30px;font-weight:800;
                            letter-spacing:0.04em;font-family:Barlow Condensed,sans-serif'>
                    {driver}
                </span>
                <span style='color:#888;font-size:13px;letter-spacing:0.05em'>
                    {team_str}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Mode A: Single Race H2H renderer ─────────────────────────────────────────

def _render_mode_a():
    """Single-race head-to-head."""

    # ── Selectors ───────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_year = st.selectbox(
            "Season",
            options=sorted(SUPPORTED_YEARS, reverse=True),
            index=0,
            key="selected_year",  # shared dengan page lain
        )
    with col2:
        rounds = get_available_rounds(selected_year)
        gp_label = st.selectbox(
            "Grand Prix",
            options=[f"R{k} · {v}" for k, v in rounds.items()],
            index=0,
            key="dc_gp",
        )
        try:
            gp_round = int(gp_label.split("·")[0].strip()[1:])
        except (ValueError, IndexError):
            st.error("Invalid GP selection.")
            return
        gp_name = rounds[gp_round]

    st.divider()
    load_btn = st.button("Load Session", type="primary", key="dc_load")

    race_session_id = f"{selected_year}_{gp_name}_R"
    race_state_key  = f"dc_race_{race_session_id}"

    if not (load_btn or race_state_key in st.session_state):
        st.markdown(
            """
            <div style='text-align:center;padding:80px 0;color:#333'>
                <div style='font-size:48px'>⚔️</div>
                <div style='font-size:18px;margin-top:16px;color:#444'>
                    Select a Grand Prix and load to start a head-to-head
                </div>
                <div style='font-size:13px;margin-top:8px;color:#2a2a2a'>
                    Qualifying gap · Race finish · Lap delta · Pace box · Stint compare · Radar
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Load both Race + Qualifying sessions ────────────────────────────────
    with st.spinner(f"Loading {gp_name} Race {selected_year}..."):
        race = load_session(selected_year, gp_name, "R")
    if race is None:
        st.error("Race session unavailable. Try a different round.")
        return

    with st.spinner(f"Loading {gp_name} Qualifying {selected_year}..."):
        quali = load_session(selected_year, gp_name, "Q")
    # Quali optional — jangan return kalau Q gagal, cukup hide section nanti

    st.session_state[race_state_key] = True

    # ── Sticky header ───────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='
            position: sticky; top: 0; z-index: 99;
            background: rgba(8,8,8,0.92);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 6px 14px;
            margin: -4px -1rem 12px;
            border-bottom: 1px solid #1A1A1A;
            font-family: Barlow Condensed, sans-serif;
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        '>
            <span style='color:#E8002D;font-weight:800'>{gp_name}</span>
            <span style='color:#666;margin:0 6px'>·</span>
            <span style='color:#CCC;font-weight:600'>Race · Quali</span>
            <span style='color:#666;margin:0 6px'>·</span>
            <span style='color:#888'>{selected_year}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Driver list (union dari race & quali results) ───────────────────────
    drivers_race = get_session_drivers(race, f"{race_session_id}")
    drivers_quali = (
        get_session_drivers(quali, f"{selected_year}_{gp_name}_Q")
        if quali is not None else []
    )
    drivers_all = sorted(set(drivers_race) | set(drivers_quali))

    if len(drivers_all) < 2:
        st.error("Not enough drivers in this session to compare.")
        return

    # ── Driver picker ───────────────────────────────────────────────────────
    st.subheader("Pick two drivers")

    # Default ke 2 driver pertama (sorted by finish position kalau bisa)
    default_a, default_b = drivers_all[0], drivers_all[1]
    try:
        res = race.results.dropna(subset=["Position"]).sort_values("Position")
        if "Abbreviation" in res.columns and len(res) >= 2:
            top_finishers = [
                str(v) for v in res["Abbreviation"].tolist() if pd.notna(v)
            ]
            avail = [d for d in top_finishers if d in drivers_all]
            if len(avail) >= 2:
                default_a, default_b = avail[0], avail[1]
    except Exception:
        pass

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        driver_a = st.selectbox(
            "Driver 1",
            options=drivers_all,
            index=drivers_all.index(default_a) if default_a in drivers_all else 0,
            key="dc_drv_a",
        )
    with pcol2:
        # Default index untuk B: jangan sama dengan A
        b_options = [d for d in drivers_all if d != driver_a]
        if not b_options:
            st.error("Need at least 2 distinct drivers in the session.")
            return
        default_b_idx = (
            b_options.index(default_b) if default_b in b_options else 0
        )
        driver_b = st.selectbox(
            "Driver 2",
            options=b_options,
            index=default_b_idx,
            key="dc_drv_b",
        )

    if driver_a == driver_b:
        st.warning("Select two different drivers.")
        return

    st.divider()

    # ── Banner cards ────────────────────────────────────────────────────────
    team_a = _team_for_driver(race, driver_a)
    team_b = _team_for_driver(race, driver_b)
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        _render_driver_card(driver_a, team_a, role="Driver 1")
    with bcol2:
        _render_driver_card(driver_b, team_b, role="Driver 2")

    # ── Qualifying H2H ──────────────────────────────────────────────────────
    if quali is not None:
        st.subheader(
            "Qualifying head-to-head",
            help=(
                "Q1/Q2/Q3 lap times + final qualifying position. "
                "'Best' = waktu tercepat lintas Q1/Q2/Q3 yang dicatat. "
                "Gap = selisih ke driver yang lebih cepat (negatif kalau "
                "driver yang lebih cepat). Kalau salah satu tidak lolos ke "
                "Q2/Q3, kolom-nya '—'."
            ),
        )
        q_a = _quali_stats(quali, driver_a)
        q_b = _quali_stats(quali, driver_b)

        # Build comparison table
        gap_str = "—"
        if q_a["best"] is not None and q_b["best"] is not None:
            delta = q_b["best"] - q_a["best"]
            if abs(delta) < 0.001:
                gap_str = "≈ equal"
            elif delta > 0:
                gap_str = f"{driver_a} +{delta:.3f}s ahead"
            else:
                gap_str = f"{driver_b} +{-delta:.3f}s ahead"

        q_df = pd.DataFrame([
            {
                "Metric": "Q1",
                driver_a: _fmt_laptime(q_a["q1"]),
                driver_b: _fmt_laptime(q_b["q1"]),
            },
            {
                "Metric": "Q2",
                driver_a: _fmt_laptime(q_a["q2"]),
                driver_b: _fmt_laptime(q_b["q2"]),
            },
            {
                "Metric": "Q3",
                driver_a: _fmt_laptime(q_a["q3"]),
                driver_b: _fmt_laptime(q_b["q3"]),
            },
            {
                "Metric": "Best",
                driver_a: _fmt_laptime(q_a["best"]),
                driver_b: _fmt_laptime(q_b["best"]),
            },
            {
                "Metric": "Quali position",
                driver_a: _fmt_pos(q_a["position"]),
                driver_b: _fmt_pos(q_b["position"]),
            },
        ])
        st.dataframe(q_df, use_container_width=True, hide_index=True)
        st.caption(f"Quali gap: **{gap_str}**")

    # ── Race H2H summary ────────────────────────────────────────────────────
    st.subheader(
        "Race head-to-head",
        help=(
            "Race finish summary + key metrics dari lap data. 'Fastest lap' = "
            "lap tercepat driver di race ini. 'Top speed' = SpeedST (end-of-"
            "main-straight speed trap). 'Median pace' & 'Std' dari clean laps "
            "(pit & outliers difilter)."
        ),
    )

    with st.spinner("Processing race lap data..."):
        clean_laps = get_laps(race, race_session_id)

    stats_a  = _race_stats(race, driver_a)
    stats_b  = _race_stats(race, driver_b)
    clean_a  = _clean_lap_stats(clean_laps, driver_a)
    clean_b  = _clean_lap_stats(clean_laps, driver_b)

    # Merge dicts
    stats_a.update(clean_a)
    stats_b.update(clean_b)

    race_rows = [
        {
            "Metric": "Grid",
            driver_a: _fmt_grid(stats_a["grid"]),
            driver_b: _fmt_grid(stats_b["grid"]),
        },
        {
            "Metric": "Finish",
            driver_a: _fmt_pos(stats_a["finish"]),
            driver_b: _fmt_pos(stats_b["finish"]),
        },
        {
            "Metric": "Points",
            driver_a: (
                f"{stats_a['points']:.0f}"
                if pd.notna(stats_a.get("points")) else "—"
            ),
            driver_b: (
                f"{stats_b['points']:.0f}"
                if pd.notna(stats_b.get("points")) else "—"
            ),
        },
        {
            "Metric": "Status",
            driver_a: stats_a["status"] or "—",
            driver_b: stats_b["status"] or "—",
        },
        {
            "Metric": "Fastest lap",
            driver_a: (
                _fmt_laptime(stats_a["fastest_lap"])
                + (f" · L{stats_a['fastest_lap_num']}" if stats_a["fastest_lap_num"] else "")
            ),
            driver_b: (
                _fmt_laptime(stats_b["fastest_lap"])
                + (f" · L{stats_b['fastest_lap_num']}" if stats_b["fastest_lap_num"] else "")
            ),
        },
        {
            "Metric": "Median pace (clean)",
            driver_a: _fmt_laptime(stats_a["median_lap"]),
            driver_b: _fmt_laptime(stats_b["median_lap"]),
        },
        {
            "Metric": "Std (consistency)",
            driver_a: (
                f"{stats_a['std_lap']:.3f}s"
                if stats_a["std_lap"] is not None else "—"
            ),
            driver_b: (
                f"{stats_b['std_lap']:.3f}s"
                if stats_b["std_lap"] is not None else "—"
            ),
        },
        {
            "Metric": "Top speed (km/h)",
            driver_a: (
                f"{stats_a['top_speed']:.0f}"
                if stats_a["top_speed"] is not None else "—"
            ),
            driver_b: (
                f"{stats_b['top_speed']:.0f}"
                if stats_b["top_speed"] is not None else "—"
            ),
        },
        {
            "Metric": "Clean laps",
            driver_a: str(stats_a["n_clean_laps"]),
            driver_b: str(stats_b["n_clean_laps"]),
        },
    ]
    st.dataframe(pd.DataFrame(race_rows), use_container_width=True, hide_index=True)

    # ── Lap-by-lap delta chart ──────────────────────────────────────────────
    st.subheader(
        "Lap-by-lap time delta",
        help=(
            f"Cumulative time gap antara {driver_b} dan {driver_a}. Garis "
            f"horizontal = {driver_a} (reference). Y axis dibalik supaya "
            f"intuitif: 'ahead' = atas, 'behind' = bawah. Pit stop, safety "
            f"car, dll otomatis ikut terhitung karena pakai timestamp "
            f"end-of-lap raw."
        ),
    )
    try:
        gap_laps = race.laps[["Driver", "LapNumber", "Time"]].copy()
        fig_delta = build_lap_delta_chart(gap_laps, driver_a, driver_b)
    except Exception:
        fig_delta = None

    if fig_delta is not None:
        st.plotly_chart(
            fig_delta, use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("Lap delta unavailable — drivers may have no overlapping lap data.")

    # ── Pace distribution ───────────────────────────────────────────────────
    st.subheader(
        "Pace distribution",
        help=(
            "Box plot lap time clean (pit & outlier sudah difilter). Box = "
            "Q1-Q3, garis tebal = median, '+' = mean ± 1 std. Titik = outlier. "
            "Box yang lebih sempit = lebih konsisten."
        ),
    )
    fig_pace = build_pace_distribution(clean_laps, driver_a, driver_b)
    if fig_pace is not None:
        st.plotly_chart(
            fig_pace, use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info(
            "Pace distribution unavailable — perlu minimal 3 clean laps "
            "per driver."
        )

    # ── Stint comparison ────────────────────────────────────────────────────
    stints = _stints_df_for_pair(race, [driver_a, driver_b])
    if len(stints) > 0:
        st.subheader(
            "Tyre strategy comparison",
            help=(
                "Stint breakdown 2 driver side-by-side. Warna = compound "
                "(S=Soft merah, M=Medium kuning, H=Hard putih, I=Inter hijau, "
                "W=Wet biru). Panjang bar = jumlah lap di stint itu."
            ),
        )
        fig_stints = build_stint_compare(stints, driver_a, driver_b)
        if fig_stints is not None:
            st.plotly_chart(
                fig_stints, use_container_width=True,
                config={"displayModeBar": False},
            )

    # ── Radar chart (5 metric) ──────────────────────────────────────────────
    norm_a, norm_b, raw = _radar_metrics(stats_a, stats_b, clean_a, clean_b)
    if len(norm_a) >= 3 and len(norm_b) >= 3:
        st.subheader(
            "Performance radar",
            help=(
                "5-axis comparison dengan normalisasi 0-1 (1.0 = best of the "
                "pair). Race pace = median lap, Consistency = std (lower better), "
                "Fastest lap = peak single lap, Top speed = SpeedST max, "
                "Sector wins = jumlah dari 3 sektor di mana driver lebih cepat. "
                "Axis yang data-nya missing untuk salah satu driver di-skip."
            ),
        )
        fig_radar = build_h2h_radar(norm_a, norm_b, driver_a, driver_b)
        if fig_radar is not None:
            st.plotly_chart(
                fig_radar, use_container_width=True,
                config={"displayModeBar": False},
            )

            # Raw value caption — lebih useful daripada cuma normalised
            lines = []
            for label, (va, vb, lower_better) in raw.items():
                if va is None or vb is None:
                    continue
                # Pilih formatter berdasarkan label
                if label == "Top speed":
                    a_str = f"{va:.0f} km/h"
                    b_str = f"{vb:.0f} km/h"
                elif label == "Consistency":
                    a_str = f"{va:.3f}s"
                    b_str = f"{vb:.3f}s"
                elif label == "Sector wins":
                    a_str = f"{int(va)}/3"
                    b_str = f"{int(vb)}/3"
                else:  # lap-time-like
                    a_str = _fmt_laptime(va)
                    b_str = _fmt_laptime(vb)
                lines.append(
                    f"<b>{label}</b>: "
                    f"<span style='color:{DRIVER_COLORS.get(driver_a, '#CCC')}'>{driver_a}</span> "
                    f"{a_str} · "
                    f"<span style='color:{DRIVER_COLORS.get(driver_b, '#CCC')}'>{driver_b}</span> "
                    f"{b_str}"
                )
            if lines:
                st.markdown(
                    f"<div style='font-size:12px;color:#888;line-height:1.7'>"
                    f"{' &nbsp;·&nbsp; '.join(lines)}</div>",
                    unsafe_allow_html=True,
                )


# ── Mode B: Season H2H renderer ──────────────────────────────────────────────

def _render_mode_b():
    """Season head-to-head pakai Ergast API."""

    col1, col2 = st.columns([1, 3])
    with col1:
        # Key HARUS unik dari Mode A — st.tabs me-render semua tab di run yang
        # sama, jadi tidak boleh dua widget pakai key="selected_year".
        # Inisialisasi index dari shared selected_year supaya tahun sinkron
        # dengan tab/page lain saat pertama dibuka.
        _year_opts = sorted(SUPPORTED_YEARS, reverse=True)
        _cur_year = st.session_state.get("selected_year", _year_opts[0])
        selected_year = st.selectbox(
            "Season",
            options=_year_opts,
            index=_year_opts.index(_cur_year) if _cur_year in _year_opts else 0,
            key="dc_b_year",
        )
    with col2:
        st.caption(
            "Data via Ergast API (race + sprint results). Cached 24 jam — "
            "fetch pertama ~5-15 detik, selanjutnya instant."
        )

    st.divider()
    load_btn = st.button("Load Season", type="primary", key="dc_b_load")

    state_key = f"dc_b_season_{selected_year}"
    if not (load_btn or state_key in st.session_state):
        st.markdown(
            """
            <div style='text-align:center;padding:80px 0;color:#333'>
                <div style='font-size:48px'>🏆</div>
                <div style='font-size:18px;margin-top:16px;color:#444'>
                    Pick a season and load to compare full-year stats
                </div>
                <div style='font-size:13px;margin-top:8px;color:#2a2a2a'>
                    Points · Wins · Podiums · Poles · FL · DNFs · Per-GP H2H · Cumulative chart
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner(f"Fetching {selected_year} season results via Ergast..."):
        drivers_meta = load_season_drivers(selected_year)
        sprints = get_season_sprints(selected_year)

    if not drivers_meta:
        st.error(
            "Could not fetch season data — Ergast API mungkin unreachable, "
            "atau musim tidak tersedia."
        )
        return

    st.session_state[state_key] = True

    driver_codes = [d["code"] for d in drivers_meta]
    name_map = {d["code"]: d["full_name"] for d in drivers_meta}

    if len(driver_codes) < 2:
        st.error("Not enough drivers in this season to compare.")
        return

    # ── Driver picker ───────────────────────────────────────────────────────
    st.subheader("Pick two drivers")

    pcol1, pcol2 = st.columns(2)

    def _label(code: str) -> str:
        return f"{code} — {name_map.get(code, code)}"

    with pcol1:
        driver_a = st.selectbox(
            "Driver 1",
            options=driver_codes,
            index=0,
            key="dc_b_drv_a",
            format_func=_label,
        )
    with pcol2:
        b_options = [d for d in driver_codes if d != driver_a]
        if not b_options:
            st.error("Need at least 2 distinct drivers in the season.")
            return
        # Pick reasonable default driver_b — kalau ada teammate, pakai teammate
        default_b_idx = 0
        a_meta = next((d for d in drivers_meta if d["code"] == driver_a), None)
        if a_meta and a_meta.get("team"):
            for i, code in enumerate(b_options):
                meta = next((d for d in drivers_meta if d["code"] == code), None)
                if meta and meta.get("team") == a_meta["team"]:
                    default_b_idx = i
                    break
        driver_b = st.selectbox(
            "Driver 2",
            options=b_options,
            index=default_b_idx,
            key="dc_b_drv_b",
            format_func=_label,
        )

    if driver_a == driver_b:
        st.warning("Select two different drivers.")
        return

    st.divider()

    # ── Aggregate stats ─────────────────────────────────────────────────────
    with st.spinner("Computing season summaries..."):
        sum_a = build_driver_season_summary(selected_year, driver_a)
        sum_b = build_driver_season_summary(selected_year, driver_b)
        per_gp = build_per_gp_h2h(selected_year, driver_a, driver_b)

    if not sum_a or not sum_b:
        st.error("Could not build season summary for one of the drivers.")
        return

    # ── Banner cards ────────────────────────────────────────────────────────
    team_a = next(
        (d["team"] for d in drivers_meta if d["code"] == driver_a), None
    )
    team_b = next(
        (d["team"] for d in drivers_meta if d["code"] == driver_b), None
    )
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        _render_driver_card(driver_a, team_a, role=f"Driver 1 · {selected_year}")
    with bcol2:
        _render_driver_card(driver_b, team_b, role=f"Driver 2 · {selected_year}")

    # ── Headline metrics row ────────────────────────────────────────────────
    st.subheader(
        "Season summary",
        help=(
            "Total points includes both race and sprint points (real F1 "
            "standings convention). Wins/Podiums = race only. Poles = "
            "qualifying P1 (catatan: Ergast count berdasarkan quali position, "
            "tidak meng-exclude grid penalty demotion). Fastest laps = laps "
            "yang dapat +1 bonus point (rank 1 dengan finish P10 atau lebih)."
        ),
    )

    metrics = [
        ("Total points", "total_points", "{:.0f}"),
        ("Wins",         "wins",         "{:d}"),
        ("Podiums",      "podiums",      "{:d}"),
        ("Poles",        "poles",        "{:d}"),
        ("Fastest laps", "fastest_laps", "{:d}"),
        ("DNFs",         "dnfs",         "{:d}"),
    ]

    mcols = st.columns(len(metrics))
    for i, (label, key, fmt) in enumerate(metrics):
        va = sum_a.get(key, 0)
        vb = sum_b.get(key, 0)
        # Untuk DNF, lebih sedikit = better. Sisanya: lebih banyak = better.
        better_a = (va < vb) if key == "dnfs" else (va > vb)
        better_b = (vb < va) if key == "dnfs" else (vb > va)

        col_a = DRIVER_COLORS.get(driver_a, "#888")
        col_b = DRIVER_COLORS.get(driver_b, "#888")

        # Highlight winner per metric
        a_style = f"color:{col_a};font-weight:700" if better_a else "color:#888"
        b_style = f"color:{col_b};font-weight:700" if better_b else "color:#888"

        try:
            va_str = fmt.format(va)
            vb_str = fmt.format(vb)
        except (ValueError, TypeError):
            va_str = str(va)
            vb_str = str(vb)

        with mcols[i]:
            st.markdown(
                f"""
                <div style='border:1px solid #1F1F1F;border-radius:4px;
                           padding:8px 10px;background:rgba(255,255,255,0.012)'>
                    <div style='color:#666;font-size:9px;font-weight:700;
                               letter-spacing:0.15em;text-transform:uppercase;
                               font-family:Barlow Condensed,sans-serif;
                               margin-bottom:6px'>{label}</div>
                    <div style='display:flex;justify-content:space-between;
                               align-items:baseline;gap:8px'>
                        <span style='font-size:18px;{a_style};
                                    font-family:Barlow Condensed,sans-serif'>{va_str}</span>
                        <span style='color:#444;font-size:11px'>vs</span>
                        <span style='font-size:18px;{b_style};
                                    font-family:Barlow Condensed,sans-serif'>{vb_str}</span>
                    </div>
                    <div style='display:flex;justify-content:space-between;
                               font-size:10px;color:#555;margin-top:2px'>
                        <span>{driver_a}</span><span>{driver_b}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Race count note
    n_a = sum_a.get("n_races_entered", 0)
    n_b = sum_b.get("n_races_entered", 0)
    if n_a != n_b:
        st.caption(
            f"⚠️ Note: {driver_a} entered {n_a} races, {driver_b} entered {n_b}. "
            f"Total counts may not be directly comparable."
        )

    # ── Cumulative points chart ─────────────────────────────────────────────
    st.subheader(
        "Cumulative points",
        help=(
            "Total poin kumulatif (race + sprint) sepanjang musim. "
            "Diamond marker = round dengan Sprint race. "
            "Crossover = lead changes."
        ),
    )
    sprint_rounds = set(sprints.keys()) if sprints else set()
    race_names_map = {}
    for _, row in per_gp.iterrows():
        race_names_map[int(row["Round"])] = str(row["GP"])

    fig_cum = build_cumulative_points_chart(
        sum_a.get("cumulative", []),
        sum_b.get("cumulative", []),
        driver_a, driver_b,
        sprint_rounds=sprint_rounds,
        race_names=race_names_map,
    )
    if fig_cum is not None:
        st.plotly_chart(
            fig_cum, use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Win-loss tally ──────────────────────────────────────────────────────
    tally = compute_h2h_tally(per_gp, driver_a, driver_b)
    col_a = DRIVER_COLORS.get(driver_a, "#888")
    col_b = DRIVER_COLORS.get(driver_b, "#888")

    st.subheader(
        "Head-to-head tally",
        help=(
            "Berapa kali driver A finish di depan driver B di Race & Qualifying. "
            "Ties = kedua DNF / sama posisi (jarang). Tidak counted: salah satu "
            "driver tidak start / DNF sebelum yang lain finish."
        ),
    )
    tcol1, tcol2 = st.columns(2)
    for label, key_a, key_b, key_tie, col in [
        ("Race wins (head-to-head)", "race_a", "race_b", "race_tied", tcol1),
        ("Qualifying head-to-head",  "quali_a", "quali_b", "quali_tied", tcol2),
    ]:
        wa = tally[key_a]
        wb = tally[key_b]
        wt = tally[key_tie]
        leader = driver_a if wa > wb else (driver_b if wb > wa else "—")
        leader_color = (
            col_a if leader == driver_a else (col_b if leader == driver_b else "#888")
        )
        with col:
            st.markdown(
                f"""
                <div style='border-left:3px solid {leader_color};
                           padding:10px 14px;margin:4px 0;
                           background:rgba(255,255,255,0.012)'>
                    <div style='color:#666;font-size:10px;font-weight:700;
                               letter-spacing:0.15em;text-transform:uppercase;
                               font-family:Barlow Condensed,sans-serif;
                               margin-bottom:6px'>{label}</div>
                    <div style='font-family:Barlow Condensed,sans-serif;font-size:22px;
                               font-weight:700;letter-spacing:0.05em'>
                        <span style='color:{col_a}'>{driver_a} {wa}</span>
                        <span style='color:#444;font-size:14px;margin:0 8px'>—</span>
                        <span style='color:{col_b}'>{wb} {driver_b}</span>
                    </div>
                    <div style='color:#666;font-size:11px;margin-top:4px'>
                        Ties: {wt}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Per-GP H2H table ────────────────────────────────────────────────────
    st.subheader(
        "Per-GP head-to-head",
        help=(
            "Setiap row = 1 GP. Quali winner & Race winner berdasarkan "
            "qualifying & race position. Points = race + sprint (kalau ada). "
            "Sprint kolom: True kalau GP itu punya sprint race."
        ),
    )
    if len(per_gp) > 0:
        # Pilih kolom yang ditampilkan (drop helper kolom underscore)
        display_cols = [
            "Round", "GP", "Sprint",
            f"Quali {driver_a}", f"Quali {driver_b}", "Quali winner",
            f"Race {driver_a}",  f"Race {driver_b}",  "Race winner",
            f"Pts {driver_a}",   f"Pts {driver_b}",
        ]
        display_cols = [c for c in display_cols if c in per_gp.columns]
        display_df = per_gp[display_cols].copy()

        # Format points columns
        for pts_col in [f"Pts {driver_a}", f"Pts {driver_b}"]:
            if pts_col in display_df.columns:
                display_df[pts_col] = display_df[pts_col].apply(
                    lambda v: f"{v:.0f}" if pd.notna(v) else "0"
                )

        # Style: highlight winner cells
        def _highlight_winners(row):
            styles = [""] * len(row)
            cols = list(row.index)
            # Race winner highlight
            if "Race winner" in cols:
                idx = cols.index("Race winner")
                w = row["Race winner"]
                if w == driver_a:
                    styles[idx] = (
                        f"background-color: rgba({_color_rgb(col_a)}, 0.25); "
                        f"color: {col_a}; font-weight: 700"
                    )
                elif w == driver_b:
                    styles[idx] = (
                        f"background-color: rgba({_color_rgb(col_b)}, 0.25); "
                        f"color: {col_b}; font-weight: 700"
                    )
            # Quali winner highlight
            if "Quali winner" in cols:
                idx = cols.index("Quali winner")
                w = row["Quali winner"]
                if w == driver_a:
                    styles[idx] = (
                        f"background-color: rgba({_color_rgb(col_a)}, 0.18); "
                        f"color: {col_a}; font-weight: 700"
                    )
                elif w == driver_b:
                    styles[idx] = (
                        f"background-color: rgba({_color_rgb(col_b)}, 0.18); "
                        f"color: {col_b}; font-weight: 700"
                    )
            return styles

        try:
            styled = display_df.style.apply(_highlight_winners, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def _color_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' → 'R,G,B' string."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "170,170,170"
    try:
        return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"
    except ValueError:
        return "170,170,170"


def _render_mode_c():
    """Circuit history H2H — compare 2 driver di satu sirkuit lintas musim."""

    years = tuple(sorted(SUPPORTED_YEARS))

    st.caption(
        f"Compare 2 drivers at one circuit across {min(years)}–{max(years)}. "
        "Data via Ergast API (cached 24h)."
    )

    load_btn = st.button("Load Circuits", type="primary", key="dc_c_load")
    state_key = "dc_c_loaded"

    if not (load_btn or state_key in st.session_state):
        st.markdown(
            """
            <div style='text-align:center;padding:80px 0;color:#333'>
                <div style='font-size:48px'>📍</div>
                <div style='font-size:18px;margin-top:16px;color:#444'>
                    Load to compare two drivers' record at a circuit
                </div>
                <div style='font-size:13px;margin-top:8px;color:#2a2a2a'>
                    Per-year quali & race · Career stats · Finish-position chart
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Fetching circuit calendar across seasons..."):
        circuits = get_circuits_for_years(years)

    if not circuits:
        st.error(
            "Could not fetch circuit data — Ergast API mungkin unreachable."
        )
        return

    st.session_state[state_key] = True

    # ── Circuit picker ──────────────────────────────────────────────────────
    # Sort by GP name, label dengan jumlah tahun tersedia
    circuit_items = sorted(
        circuits.items(),
        key=lambda kv: kv[1].get("gp_name", kv[0]),
    )
    circuit_ids = [cid for cid, _ in circuit_items]

    def _circuit_label(cid: str) -> str:
        info = circuits[cid]
        n_years = len(info.get("years", {}))
        gp = info.get("gp_name", cid)
        return f"{gp} ({n_years} season{'s' if n_years != 1 else ''})"

    selected_circuit = st.selectbox(
        "Circuit",
        options=circuit_ids,
        index=0,
        key="dc_c_circuit",
        format_func=_circuit_label,
    )

    # ── Driver list: union dari semua driver di SUPPORTED_YEARS ──────────────
    all_drivers: dict[str, str] = {}
    for y in years:
        for d in load_season_drivers(y):
            all_drivers.setdefault(d["code"], d["full_name"])
    driver_codes = sorted(all_drivers.keys())

    if len(driver_codes) < 2:
        st.error("Not enough drivers to compare.")
        return

    def _label(code: str) -> str:
        return f"{code} — {all_drivers.get(code, code)}"

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        driver_a = st.selectbox(
            "Driver 1", options=driver_codes, index=0,
            key="dc_c_drv_a", format_func=_label,
        )
    with pcol2:
        b_options = [d for d in driver_codes if d != driver_a]
        if not b_options:
            st.error("Need at least 2 distinct drivers.")
            return
        driver_b = st.selectbox(
            "Driver 2", options=b_options, index=0,
            key="dc_c_drv_b", format_func=_label,
        )

    if driver_a == driver_b:
        st.warning("Select two different drivers.")
        return

    st.divider()

    # ── Build H2H ───────────────────────────────────────────────────────────
    with st.spinner("Building circuit head-to-head..."):
        circuit_df = build_circuit_h2h(selected_circuit, driver_a, driver_b, years)

    if circuit_df is None or len(circuit_df) == 0:
        st.info("No data for this circuit / driver combination.")
        return

    info = circuits.get(selected_circuit, {})
    gp_name = info.get("gp_name", selected_circuit)
    circuit_name = info.get("name", "")

    st.markdown(
        f"<div style='font-family:Barlow Condensed,sans-serif;font-size:15px;"
        f"color:#E8002D;font-weight:700;letter-spacing:0.05em;"
        f"text-transform:uppercase'>{gp_name}</div>"
        f"<div style='color:#666;font-size:12px;margin-bottom:8px'>{circuit_name}</div>",
        unsafe_allow_html=True,
    )

    # ── Career stats cards ──────────────────────────────────────────────────
    stats = build_circuit_career_stats(circuit_df, driver_a, driver_b)
    if stats:
        st.subheader(
            "Career record at this circuit",
            help=(
                "Agregat dari musim-musim yang tersedia. Best = posisi finish "
                "terbaik, Avg = rata-rata posisi finish (lebih kecil = lebih "
                "baik). Starts = jumlah kali start di sirkuit ini dalam rentang "
                "data."
            ),
        )
        scol1, scol2 = st.columns(2)
        for code, col in [(driver_a, scol1), (driver_b, scol2)]:
            s = stats.get(code, {})
            color = DRIVER_COLORS.get(code, "#888")
            best = s.get("best_finish")
            avg = s.get("avg_finish")
            best_str = f"P{best}" if best is not None else "—"
            avg_str = f"P{avg:.1f}" if avg is not None else "—"
            with col:
                st.markdown(
                    f"""
                    <div style='border-left:3px solid {color};
                               padding:12px 16px;margin:4px 0;
                               background:rgba(255,255,255,0.02);
                               border-radius:0 4px 4px 0'>
                        <div style='color:{color};font-size:20px;font-weight:800;
                                   font-family:Barlow Condensed,sans-serif;
                                   letter-spacing:0.04em;margin-bottom:8px'>{code}</div>
                        <table style='width:100%;font-family:Barlow,sans-serif;
                                     font-size:13px;color:#CCC'>
                            <tr><td style='color:#888'>Best finish</td>
                                <td style='text-align:right;font-weight:700'>{best_str}</td></tr>
                            <tr><td style='color:#888'>Avg finish</td>
                                <td style='text-align:right'>{avg_str}</td></tr>
                            <tr><td style='color:#888'>Wins</td>
                                <td style='text-align:right'>{s.get('wins', 0)}</td></tr>
                            <tr><td style='color:#888'>Podiums</td>
                                <td style='text-align:right'>{s.get('podiums', 0)}</td></tr>
                            <tr><td style='color:#888'>DNFs</td>
                                <td style='text-align:right'>{s.get('dnfs', 0)}</td></tr>
                            <tr><td style='color:#888'>Starts</td>
                                <td style='text-align:right'>{s.get('starts', 0)}</td></tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Finish position chart ───────────────────────────────────────────────
    st.subheader(
        "Finish position by year",
        help=(
            "Race finish position per tahun. Y-axis dibalik (P1 di atas). "
            "Tahun tanpa bar = driver tidak compete (DNC) atau tidak ada data "
            "posisi (DNS/DNF tanpa klasifikasi)."
        ),
    )
    fig = build_circuit_finish_chart(circuit_df, driver_a, driver_b)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("Not enough finish data to chart.")

    # ── Per-year detail table ───────────────────────────────────────────────
    st.subheader(
        "Year-by-year detail",
        help=(
            "Quali & race position + total points (race + sprint) per tahun. "
            "'DNC' = did not compete (driver belum/tidak balapan tahun itu)."
        ),
    )
    display_cols = [
        "Year",
        f"Quali {driver_a}", f"Quali {driver_b}",
        f"Race {driver_a}",  f"Race {driver_b}",
        f"Pts {driver_a}",   f"Pts {driver_b}",
        f"Status {driver_a}", f"Status {driver_b}",
    ]
    display_cols = [c for c in display_cols if c in circuit_df.columns]
    display_df = circuit_df[display_cols].copy()

    # Format points
    for pts_col in [f"Pts {driver_a}", f"Pts {driver_b}"]:
        if pts_col in display_df.columns:
            display_df[pts_col] = display_df[pts_col].apply(
                lambda v: f"{v:.0f}" if pd.notna(v) else "—"
            )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def render():
    st.title("Driver Comparison")

    _current_year = st.session_state.get("selected_year", 2024)
    st.markdown(
        f"<p style='color:#444;margin-top:-16px;font-size:13px;"
        f"letter-spacing:0.05em;text-transform:uppercase'>"
        f"Head-to-head · Quali & Race · {_current_year} Season</p>",
        unsafe_allow_html=True,
    )

    tab_a, tab_b, tab_c = st.tabs([
        "Single Race",
        "Season",
        "Circuit history",
    ])

    with tab_a:
        _render_mode_a()

    with tab_b:
        _render_mode_b()

    with tab_c:
        _render_mode_c()
