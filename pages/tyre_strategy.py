"""
Tyre / Race Strategy Page
==========================
Features:
- Stint breakdown chart (#1)
- Compound usage summary (#2)
- Pit stop timeline (#3)
- Tyre degradation scatter (#4)
- Stint pace evolution (#6)
- Undercut/Overcut detector (#8)
- Help tooltips (#19)
- ML Pit Optimizer (#12-#15):
  - Tyre degradation model (XGBoost regression)
  - Optimal pit window predictor
  - What-if simulator
  - SHAP / feature importance
"""

import pandas as pd
import streamlit as st

from src.pipeline.loader import (
    load_session,
    get_laps,
    get_available_rounds,
)
from src.viz.strategy_charts import (
    build_stint_breakdown,
    build_compound_usage,
    build_pit_timeline,
    build_tyre_degradation_scatter,
    build_stint_pace_evolution,
)
from src.viz.ml_charts import (
    build_predicted_vs_actual,
    build_pit_window_chart,
    build_feature_importance,
)
from src.ml.tyre_model import (
    prepare_training_data,
    train_pit_model,
    optimize_pit_window,
    compute_feature_importance,
    COMPOUND_LEVELS,
)
from src.utils.config import (
    COMPOUND_COLORS,
    DRIVER_COLORS,
    SESSION_LABELS,
    SUPPORTED_YEARS,
)


# ── ML training cache (module-level) ──────────────────────────────────────
# Defined di module level (bukan di dalam render) supaya Streamlit cache_resource
# stable identity → cache hit konsisten antar rerun.
@st.cache_resource(ttl=3600, show_spinner=False)
def _train_cached_model(session_id_key: str, _X, _y):
    """Trained model cached per session. _X/_y excluded from hash via _-prefix."""
    return train_pit_model(_X, _y)


# ── Helpers ────────────────────────────────────────────────────────────────

def _stints_df(session) -> pd.DataFrame:
    """
    Extract stint info per driver dari session.laps.
    Returns DataFrame: Driver, Stint, Compound, LapStart, LapEnd, Laps.
    """
    try:
        needed = ["Driver", "LapNumber", "Compound", "Stint"]
        if not all(c in session.laps.columns for c in needed):
            return pd.DataFrame()
        laps = session.laps[needed].copy()
    except Exception:
        return pd.DataFrame()

    laps = laps.dropna(subset=["Compound", "Stint", "LapNumber", "Driver"])
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


def _pit_laps_df(session) -> pd.DataFrame:
    """
    Pit-in laps DataFrame: Driver, LapNumber (pit-in lap), CompoundAfter.
    """
    try:
        needed = ["Driver", "LapNumber", "PitInTime", "Compound"]
        if not all(c in session.laps.columns for c in needed):
            return pd.DataFrame()
        all_laps = session.laps[needed].copy()
    except Exception:
        return pd.DataFrame()

    pit_rows = []
    for driver, drv_laps in all_laps.groupby("Driver"):
        drv = drv_laps.sort_values("LapNumber").reset_index(drop=True)
        for i in range(len(drv)):
            row = drv.iloc[i]
            if pd.isna(row.get("PitInTime")):
                continue
            compound_after = None
            if i + 1 < len(drv):
                comp = drv.iloc[i + 1].get("Compound")
                compound_after = comp if isinstance(comp, str) else None
            pit_rows.append({
                "Driver":        driver,
                "LapNumber":     int(row["LapNumber"]) if pd.notna(row["LapNumber"]) else None,
                "CompoundAfter": compound_after,
            })

    if not pit_rows:
        return pd.DataFrame()
    return pd.DataFrame(pit_rows)


def _driver_order_by_finish(session) -> list[str]:
    """Return list of driver abbreviations sorted by final classification."""
    try:
        res = session.results.dropna(subset=["Position"]).sort_values("Position")
        if "Abbreviation" not in res.columns:
            return []
        return [str(v) for v in res["Abbreviation"].tolist() if pd.notna(v)]
    except Exception:
        return []


def _pit_effect_table(session) -> pd.DataFrame:
    """
    Per pit stop: position before vs 3 laps after pit-out.
    Negative delta = gained positions (potential undercut/overcut success).
    """
    try:
        needed = ["Driver", "LapNumber", "PitInTime", "Position", "Compound"]
        if not all(c in session.laps.columns for c in needed):
            return pd.DataFrame()
        all_laps = session.laps[needed].copy()
    except Exception:
        return pd.DataFrame()

    all_laps = all_laps.dropna(subset=["LapNumber", "Driver"])
    if len(all_laps) == 0:
        return pd.DataFrame()

    rows = []
    for driver, drv_laps in all_laps.groupby("Driver"):
        drv = drv_laps.sort_values("LapNumber").reset_index(drop=True)
        for i in range(len(drv)):
            row = drv.iloc[i]
            if pd.isna(row.get("PitInTime")):
                continue
            # Need positions before & after
            pos_before_val = row.get("Position")
            if pd.isna(pos_before_val):
                continue
            try:
                pos_before = int(pos_before_val)
            except (TypeError, ValueError):
                continue
            pit_lap = int(row["LapNumber"]) if pd.notna(row["LapNumber"]) else None

            # Settled position = 3 laps after pit-out (= 3 laps after current row + out lap)
            settled_idx = min(i + 3, len(drv) - 1)
            settled_row = drv.iloc[settled_idx]
            pos_after_val = settled_row.get("Position")
            if pd.isna(pos_after_val):
                continue
            try:
                pos_after = int(pos_after_val)
            except (TypeError, ValueError):
                continue

            delta = pos_after - pos_before  # negative = gained positions

            # Compound transition
            comp_in = row.get("Compound")
            comp_out = drv.iloc[i + 1].get("Compound") if i + 1 < len(drv) else None
            comp_str = (
                f"{comp_in.title() if isinstance(comp_in, str) else '—'} → "
                f"{comp_out.title() if isinstance(comp_out, str) else '—'}"
            )

            # Effect label
            if delta < 0:
                effect = f"+{-delta}"   # gained
            elif delta > 0:
                effect = f"−{delta}"    # lost (en dash for clarity)
            else:
                effect = "0"

            rows.append({
                "Driver":     driver,
                "Pit Lap":    pit_lap,
                "Compound":   comp_str,
                "Pos Before": f"P{pos_before}",
                "Pos +3 laps": f"P{pos_after}",
                "Effect":     effect,
                "_delta":     delta,  # hidden, untuk styling
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["Pit Lap", "Driver"]).reset_index(drop=True)
    return df


# ── Nice-to-have analytics ────────────────────────────────────────────────

def _strategy_comparison(stints_df: pd.DataFrame, driver_order: list[str]) -> pd.DataFrame:
    """
    Untuk tiap driver: compound sequence (e.g., "M-H-H") + jumlah stop + detail stints.
    Driver_order untuk preserve finishing position order.
    """
    if stints_df is None or len(stints_df) == 0:
        return pd.DataFrame()

    rows = []
    drivers = [d for d in driver_order if d in stints_df["Driver"].unique()] \
              if driver_order else sorted(stints_df["Driver"].unique())

    for driver in drivers:
        drv = stints_df[stints_df["Driver"] == driver].sort_values("Stint")
        if len(drv) == 0:
            continue
        compounds = [
            (c[0].upper() if isinstance(c, str) and len(c) > 0 else "?")
            for c in drv["Compound"]
        ]
        sequence = "–".join(compounds)
        detail = " · ".join(
            f"{(c[0].upper() if isinstance(c, str) and len(c) > 0 else '?')}({int(l)})"
            for c, l in zip(drv["Compound"], drv["Laps"])
        )
        rows.append({
            "Driver":   driver,
            "Stops":    len(drv) - 1,
            "Strategy": sequence,
            "Stints":   detail,
        })

    return pd.DataFrame(rows)


def _driver_tyre_mgmt(clean_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Degradation rate (s/lap) per (Driver, Compound) via linear regression
    pada TyreLife vs LapTimeSeconds. Lower = better tyre management.
    """
    needed = ("Driver", "Compound", "TyreLife", "LapTimeSeconds")
    if any(c not in clean_laps.columns for c in needed):
        return pd.DataFrame()

    df = clean_laps.dropna(subset=list(needed)).copy()
    if len(df) == 0:
        return pd.DataFrame()

    import numpy as np
    rows = []
    for (driver, compound), grp in df.groupby(["Driver", "Compound"]):
        if not isinstance(compound, str) or len(grp) < 5:
            continue
        # Filter outliers per group (>2σ above median)
        median = grp["LapTimeSeconds"].median()
        std = grp["LapTimeSeconds"].std()
        if pd.notna(std):
            grp = grp[grp["LapTimeSeconds"] <= median + 2 * std]
        if len(grp) < 5:
            continue
        x = grp["TyreLife"].astype(float).values
        y = grp["LapTimeSeconds"].astype(float).values
        try:
            slope, intercept = np.polyfit(x, y, deg=1)
        except Exception:
            continue
        rows.append({
            "Driver":              driver,
            "Compound":            compound.title(),
            "Degradation (s/lap)": float(round(slope, 4)),
            "Avg pace (s)":        float(round(y.mean(), 3)),
            "Samples":             len(grp),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Degradation (s/lap)").reset_index(drop=True)


def _tyre_warmup(clean_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Per compound, rata-rata tyre age (laps) saat peak pace (fastest lap di stint
    itu). Mengindikasikan berapa lap warm-up tipikal sebelum compound capai
    optimal pace.
    """
    needed = ("Driver", "Stint", "Compound", "TyreLife", "LapTimeSeconds")
    if any(c not in clean_laps.columns for c in needed):
        return pd.DataFrame()

    df = clean_laps.dropna(subset=list(needed)).copy()
    if len(df) == 0:
        return pd.DataFrame()

    rows = []
    for compound, grp_c in df.groupby("Compound"):
        if not isinstance(compound, str):
            continue
        ages_at_best = []
        for (driver, stint), grp in grp_c.groupby(["Driver", "Stint"]):
            if len(grp) < 3:
                continue
            best_idx = grp["LapTimeSeconds"].idxmin()
            try:
                ages_at_best.append(int(grp.loc[best_idx, "TyreLife"]))
            except Exception:
                continue
        if not ages_at_best:
            continue
        rows.append({
            "Compound":             compound.title(),
            "Avg laps to peak":     float(round(sum(ages_at_best) / len(ages_at_best), 1)),
            "Min":                  int(min(ages_at_best)),
            "Max":                  int(max(ages_at_best)),
            "Stints analyzed":      len(ages_at_best),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Avg laps to peak").reset_index(drop=True)


def _pit_window_stats(stints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per compound: min/median/max stint length — defines typical "pit window".
    """
    if stints_df is None or len(stints_df) == 0:
        return pd.DataFrame()
    if "Compound" not in stints_df.columns or "Laps" not in stints_df.columns:
        return pd.DataFrame()

    import numpy as np
    rows = []
    for compound, grp in stints_df.groupby("Compound"):
        if not isinstance(compound, str):
            continue
        lengths = grp["Laps"].dropna().astype(int).values
        if len(lengths) == 0:
            continue
        rows.append({
            "Compound":     compound.title(),
            "Min stint":    int(lengths.min()),
            "Median stint": int(round(np.median(lengths))),
            "Max stint":    int(lengths.max()),
            "Avg":          float(round(lengths.mean(), 1)),
            "Stints":       len(lengths),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Median stint", ascending=False).reset_index(drop=True)


def _strategy_auto_summary(
    stints_df: pd.DataFrame,
    session,
) -> str | None:
    """
    Auto-generate 2-4 sentence narrative tentang strategy race ini.
    """
    if stints_df is None or len(stints_df) == 0:
        return None

    # Stops per driver
    stops_per_driver = (
        stints_df.groupby("Driver")["Stint"].max() - 1
    ).astype(int)
    if len(stops_per_driver) == 0:
        return None

    n_drivers = len(stops_per_driver)
    sentences: list[str] = []

    # 1. Most common stop strategy
    mode_stops = stops_per_driver.mode()
    if not mode_stops.empty:
        most_common = int(mode_stops.iloc[0])
        count_majority = int((stops_per_driver == most_common).sum())
        sentences.append(
            f"<b>{count_majority}/{n_drivers}</b> drivers ran a "
            f"<b>{most_common}-stop</b> strategy."
        )

    # 2. Compound sequences
    sequences: dict[str, list[str]] = {}
    for driver in stops_per_driver.index:
        drv = stints_df[stints_df["Driver"] == driver].sort_values("Stint")
        seq = "–".join(
            c[0].upper() if isinstance(c, str) and len(c) > 0 else "?"
            for c in drv["Compound"]
        )
        sequences.setdefault(seq, []).append(driver)

    if sequences:
        top_seq, top_drivers = max(sequences.items(), key=lambda kv: len(kv[1]))
        if len(top_drivers) >= 2:
            sentences.append(
                f"Dominant compound sequence: <b style='color:#FFD700'>{top_seq}</b> "
                f"({len(top_drivers)} drivers)."
            )

        # Outliers — unique strategies
        outliers = {seq: drvs for seq, drvs in sequences.items() if len(drvs) == 1}
        if outliers:
            outlier_strs = [f"{drvs[0]} ({seq})" for seq, drvs in list(outliers.items())[:3]]
            if outlier_strs:
                sentences.append(
                    f"Outlier strategies: {', '.join(outlier_strs)}."
                )

    if not sentences:
        return None
    return " ".join(sentences)


def _strategy_backtest(
    model,
    stints_df: pd.DataFrame,
    pit_laps_df: pd.DataFrame,
    total_laps: int,
    track_temp: float,
    air_temp: float,
    pit_loss: float = 22.0,
) -> pd.DataFrame:
    """
    Untuk tiap driver yang pit tepat 1×, predict optimal pit lap dari model
    (assume start state = lap 1, fresh tyres, starting compound). Compare ke
    actual pit lap → delta verdict (early/late/optimal).
    """
    if model is None or stints_df is None or pit_laps_df is None:
        return pd.DataFrame()
    if len(pit_laps_df) == 0 or len(stints_df) == 0:
        return pd.DataFrame()

    rows = []
    drivers = pit_laps_df["Driver"].unique()
    for driver in drivers:
        drv_pits = pit_laps_df[pit_laps_df["Driver"] == driver].sort_values("LapNumber")
        if len(drv_pits) != 1:    # skip 0-stop & 2+ stop drivers
            continue
        try:
            actual_pit_lap = int(drv_pits.iloc[0]["LapNumber"])
        except Exception:
            continue
        compound_after = drv_pits.iloc[0].get("CompoundAfter")

        drv_stints = stints_df[stints_df["Driver"] == driver].sort_values("Stint")
        if len(drv_stints) < 2:
            continue
        start_compound = drv_stints.iloc[0]["Compound"]
        if not isinstance(start_compound, str):
            continue
        fresh_compound = compound_after if isinstance(compound_after, str) else "HARD"

        opt = optimize_pit_window(
            model=model,
            current_lap=1, current_tyre_life=0,
            current_compound=start_compound, current_stint=1,
            fresh_compound=fresh_compound,
            total_laps=total_laps,
            track_temp=track_temp, air_temp=air_temp,
            pit_loss_seconds=pit_loss,
        )
        if opt is None:
            continue

        optimal_lap = int(opt["optimal_pit_lap"])
        delta = actual_pit_lap - optimal_lap
        if delta > 1:
            verdict = f"⟶ {delta} laps LATE"
        elif delta < -1:
            verdict = f"⟶ {-delta} laps EARLY"
        else:
            verdict = "≈ OPTIMAL"

        rows.append({
            "Driver":         driver,
            "Actual":         actual_pit_lap,
            "Predicted opt.": optimal_lap,
            "Δ (laps)":       delta,
            "Verdict":        verdict,
            "Saving":         f"{opt['saving']:.1f}s",
        })

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("Δ (laps)", key=abs)
        .reset_index(drop=True)
    )


# ── Render ─────────────────────────────────────────────────────────────────

def render():
    st.title("Tyre Strategy")

    _current_year = st.session_state.get("selected_year", 2024)
    st.markdown(
        f"<p style='color:#444;margin-top:-16px;font-size:13px;"
        f"letter-spacing:0.05em;text-transform:uppercase'>"
        f"Stints · Pit Windows · Tyre Degradation · {_current_year} Season</p>",
        unsafe_allow_html=True,
    )

    # ── Session selectors ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        selected_year = st.selectbox(
            "Season",
            options=sorted(SUPPORTED_YEARS, reverse=True),
            index=0,
            key="selected_year",
        )

    with col2:
        rounds = get_available_rounds(selected_year)
        gp_label = st.selectbox(
            "Grand Prix",
            options=[f"R{k} · {v}" for k, v in rounds.items()],
            index=0,
            key="ts_gp",
        )
        gp_round = int(gp_label.split("·")[0].strip()[1:])
        gp_name  = rounds[gp_round]

    with col3:
        # Tyre strategy hanya make sense untuk Race
        session_type = st.selectbox(
            "Session",
            options=["R"],
            format_func=lambda x: SESSION_LABELS[x],
            key="ts_session",
            help="Tyre strategy analysis hanya untuk Race session.",
        )

    st.divider()
    load_btn = st.button("Load Session", type="primary", key="ts_load")

    session_id  = f"{selected_year}_{gp_name}_{session_type}"
    session_key = f"ts_session_{session_id}"

    if not (load_btn or session_key in st.session_state):
        st.markdown(
            """
            <div style='text-align:center;padding:80px 0;color:#333'>
                <div style='font-size:18px;margin-top:16px;color:#444'>
                    Select a Grand Prix and load to see tyre strategy
                </div>
                <div style='font-size:13px;margin-top:8px;color:#2a2a2a'>
                    Stint breakdown · Compound usage · Pit timeline · Degradation curves
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner(f"Loading {gp_name} {SESSION_LABELS[session_type]} {selected_year}..."):
        session = load_session(selected_year, gp_name, session_type)

    if session is None:
        st.error("Session data unavailable. Try a different round.")
        return

    st.session_state[session_key] = True

    # Sticky session header (mirror Race Overview pattern)
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
            <span style='color:#CCC;font-weight:600'>{SESSION_LABELS[session_type]}</span>
            <span style='color:#666;margin:0 6px'>·</span>
            <span style='color:#888'>{selected_year}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Data prep ────────────────────────────────────────────────────────────
    stints = _stints_df(session)
    pit_laps = _pit_laps_df(session)
    finish_order = _driver_order_by_finish(session)

    if len(stints) == 0:
        st.error("No stint data available for this session.")
        return

    # ── Driver filter ────────────────────────────────────────────────────────
    all_drivers = sorted(stints["Driver"].unique().tolist())
    st.subheader(
        "Driver filter",
        help=(
            "Filter driver untuk chart degradation, stint pace evolution, "
            "dan pit effect table. Stint breakdown & compound summary tetap "
            "tampil semua driver."
        ),
    )
    st.caption("Default: all drivers. Deselect to focus on specific drivers.")
    selected_drivers = st.multiselect(
        "Drivers",
        options=all_drivers,
        default=all_drivers,
        label_visibility="collapsed",
        key="ts_drivers",
    )

    st.divider()

    # ── #1 Stint breakdown chart ─────────────────────────────────────────────
    st.subheader(
        "Stint breakdown",
        help=(
            "Tiap baris = 1 driver, tiap segmen warna = 1 stint dengan compound "
            "tertentu (S=Soft merah, M=Medium kuning, H=Hard putih, I=Inter hijau, "
            "W=Wet biru). Panjang bar = jumlah lap di stint itu. Driver disort "
            "by posisi finish (terbaik di atas)."
        ),
    )
    fig_stints = build_stint_breakdown(
        stints, drivers_order=finish_order or all_drivers
    )
    if fig_stints is not None:
        st.plotly_chart(fig_stints, use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("Stint breakdown unavailable for this session.")

    # ── #2 Compound usage summary ────────────────────────────────────────────
    st.subheader(
        "Compound usage",
        help=(
            "Total lap yang ditempuh di tiap compound, dijumlah dari semua "
            "driver. Bagus buat lihat compound mana yang dominant di race ini "
            "(strategy preference + track characteristic)."
        ),
    )
    fig_usage = build_compound_usage(stints)
    if fig_usage is not None:
        st.plotly_chart(fig_usage, use_container_width=True,
                        config={"displayModeBar": False})

    # ── #3 Pit stop timeline ─────────────────────────────────────────────────
    if len(pit_laps) > 0:
        try:
            total_laps_val = int(session.laps["LapNumber"].max())
        except Exception:
            total_laps_val = None

        st.subheader(
            "Pit stop timeline",
            help=(
                "Tiap diamond = 1 pit stop. Posisi X = lap saat pit. Warna "
                "diamond = compound baru yang dipasang (S=merah, M=kuning, dst). "
                "Driver disort by posisi finish."
            ),
        )
        fig_pits = build_pit_timeline(
            pit_laps, drivers_order=finish_order or all_drivers,
            total_laps=total_laps_val,
        )
        if fig_pits is not None:
            st.plotly_chart(fig_pits, use_container_width=True,
                            config={"displayModeBar": False})

    # ── #18 Auto strategy summary ────────────────────────────────────────────
    narrative = _strategy_auto_summary(stints, session)
    if narrative:
        st.markdown(
            f"""
            <div style='padding:12px 16px;margin:12px 0 8px;
                       background:linear-gradient(90deg, rgba(232,0,45,0.05) 0%, rgba(255,255,255,0.01) 100%);
                       border-left:3px solid #E8002D;border-radius:0 4px 4px 0;
                       font-family:Barlow,sans-serif;font-size:13.5px;
                       line-height:1.5;color:#DDD'>
                <div style='color:#888;font-size:10px;font-weight:700;
                           letter-spacing:0.15em;text-transform:uppercase;
                           font-family:Barlow Condensed,sans-serif;margin-bottom:6px'>
                    Strategy Summary
                </div>
                {narrative}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── #10 Strategy comparison ──────────────────────────────────────────────
    strat_df = _strategy_comparison(stints, finish_order or all_drivers)
    if len(strat_df) > 0:
        st.subheader(
            "Strategy comparison",
            help=(
                "Per driver: compound sequence (S=Soft, M=Medium, H=Hard, "
                "I=Inter, W=Wet) + jumlah pit stops. Detail kolom 'Stints' "
                "format C(N) = compound dengan N laps. Disort by finish position."
            ),
        )
        st.dataframe(strat_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Get clean laps untuk degradation analysis ────────────────────────────
    with st.spinner("Processing lap data for degradation analysis..."):
        clean_laps = get_laps(session, session_id)

    # ── #4 Tyre degradation scatter ──────────────────────────────────────────
    if len(clean_laps) > 0:
        deg_laps = clean_laps[clean_laps["Driver"].isin(selected_drivers)] \
                    if selected_drivers else clean_laps
        st.subheader(
            "Tyre degradation",
            help=(
                "Scatter lap time vs tyre age (umur ban dalam laps), color per "
                "compound. Dashed line = polynomial trend (degree 2). "
                "Carikan 'cliff point' — di mana curve mulai naik tajam = "
                "compound performance drop. Hard biasanya curve paling flat, "
                "Soft paling steep."
            ),
        )
        fig_deg = build_tyre_degradation_scatter(deg_laps)
        if fig_deg is not None:
            st.plotly_chart(fig_deg, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("Not enough data for degradation analysis.")

    # ── #6 Stint pace evolution ──────────────────────────────────────────────
    if len(clean_laps) > 0:
        st.subheader(
            "Stint pace evolution",
            help=(
                "Tiap line = 1 stint (1 driver × 1 stint). X = umur ban, Y = lap "
                "time. Color by compound. Banyak stint overlay membantu lihat "
                "konsistensi degradation per compound dan compare stint strategy "
                "antar driver."
            ),
        )
        fig_pace = build_stint_pace_evolution(clean_laps, selected_drivers)
        if fig_pace is not None:
            st.plotly_chart(fig_pace, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("Not enough stint data for pace evolution.")

    # ── #5 Driver tyre management ranking ────────────────────────────────────
    if len(clean_laps) > 0:
        mgmt_df = _driver_tyre_mgmt(clean_laps)
        if selected_drivers:
            mgmt_df = mgmt_df[mgmt_df["Driver"].isin(selected_drivers)]
        if len(mgmt_df) > 0:
            st.subheader(
                "Driver tyre management ranking",
                help=(
                    "Degradation rate (detik/lap) per driver per compound, "
                    "computed via linear regression TyreLife vs LapTimeSeconds. "
                    "Lower = better tyre management. Filter outliers >2σ per group."
                ),
            )
            # Highlight smallest degradation per compound (best manager)
            def _highlight_best_mgmt(col):
                if col.name == "Degradation (s/lap)":
                    return [
                        "background-color: rgba(57,181,74,0.18); color: #39B54A; font-weight: 700"
                        if v == col.min() else ""
                        for v in col
                    ]
                return [""] * len(col)
            try:
                styled = mgmt_df.style.apply(_highlight_best_mgmt, axis=0)
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(mgmt_df, use_container_width=True, hide_index=True)

    # ── #7 Tyre warm-up analysis ─────────────────────────────────────────────
    if len(clean_laps) > 0:
        warmup_df = _tyre_warmup(clean_laps)
        if len(warmup_df) > 0:
            st.subheader(
                "Tyre warm-up analysis",
                help=(
                    "Berapa lap rata-rata tyre butuh untuk capai peak pace (lap "
                    "tercepat di stint). Lower = warm-up cepat (e.g., Soft biasa "
                    "warm-up 1-3 lap). Bisa beda per circuit & weather."
                ),
            )
            st.dataframe(warmup_df, use_container_width=True, hide_index=True)

    # ── #9 Pit window analysis (stint length stats per compound) ─────────────
    pw_df = _pit_window_stats(stints)
    if len(pw_df) > 0:
        st.subheader(
            "Pit window — stint length per compound",
            help=(
                "Statistik panjang stint per compound dari data session ini. "
                "Median = typical stint length. Min/Max = range strategy. "
                "Berguna untuk reference 'kapan biasanya pit out of this compound'."
            ),
        )
        st.dataframe(pw_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── #8 Undercut/Overcut detector (pit effect table) ──────────────────────
    pit_effect = _pit_effect_table(session)
    if len(pit_effect) > 0:
        pit_effect_f = pit_effect[pit_effect["Driver"].isin(selected_drivers)] \
                       if selected_drivers else pit_effect

        if len(pit_effect_f) > 0:
            st.subheader(
                "Pit effect (undercut / overcut detector)",
                help=(
                    "Untuk tiap pit stop: posisi 1 lap SEBELUM pit-in vs "
                    "posisi 3 lap SETELAH pit-out (= waktu settle pasca pit). "
                    "Effect positif (+N) = gained positions → undercut atau "
                    "overcut sukses. Negatif (−N) = lost positions → typical pit "
                    "drop atau strategy gagal. Highlight hijau = sukses, "
                    "merah = gagal."
                ),
            )

            # Hide internal _delta, keep for styling
            display_cols = ["Driver", "Pit Lap", "Compound",
                            "Pos Before", "Pos +3 laps", "Effect"]

            def _highlight_effect(row):
                styles = [""] * len(row)
                if "Effect" in row.index:
                    idx_effect = list(row.index).index("Effect")
                    delta_val = pit_effect_f.loc[row.name, "_delta"]
                    if pd.notna(delta_val):
                        if delta_val < 0:
                            styles[idx_effect] = (
                                "background-color: rgba(57,181,74,0.22); "
                                "color: #39B54A; font-weight: 700"
                            )
                        elif delta_val > 0:
                            styles[idx_effect] = (
                                "background-color: rgba(220,50,50,0.18); "
                                "color: #DC3545; font-weight: 700"
                            )
                return styles

            try:
                styled = pit_effect_f[display_cols].style.apply(
                    _highlight_effect, axis=1
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                # Fallback ke unstyled kalau styler error
                st.dataframe(
                    pit_effect_f[display_cols],
                    use_container_width=True, hide_index=True,
                )

    st.divider()

    # ───────────────────────────────────────────────────────────────────────
    # ML SECTION — Tyre degradation model + Pit optimizer (#12-#15)
    # ───────────────────────────────────────────────────────────────────────
    st.subheader(
        "ML Pit Stop Optimizer",
        help=(
            "Trained XGBoost regressor pada lap data session ini untuk predict "
            "lap time sebagai fungsi (TyreLife, Compound, FuelLoad, "
            "TrackTemp, AirTemp, Stint). Lalu pakai model untuk cari "
            "pit lap optimal yang minimize total remaining race time. "
            "Slider what-if biar bisa explore alternative strategy."
        ),
    )

    # Get weather data (untuk fitur TrackTemp, AirTemp)
    try:
        weather_data = session.weather_data
    except Exception:
        weather_data = None

    try:
        total_laps_val = int(session.laps["LapNumber"].max())
    except Exception:
        total_laps_val = None

    if len(clean_laps) < 20:
        st.info(
            "Not enough clean laps (need ≥ 20) untuk train ML model. "
            "Coba session race full-length."
        )
        return

    # Prepare training data
    with st.spinner("Preparing training data..."):
        X, y, feature_cols = prepare_training_data(
            clean_laps, weather_data, total_laps_val
        )

    if X is None or y is None or len(X) < 20:
        st.warning("Insufficient feature data for ML training.")
        return

    # Train model — pakai module-level cache untuk stable identity
    with st.spinner(f"Training XGBoost model on {len(X)} laps..."):
        result = _train_cached_model(session_id, X, y)

    if result is None:
        st.error(
            "ML training failed — XGBoost/sklearn might not be installed, "
            "atau training data corrupted."
        )
        return

    model, metrics = result

    # Training metrics
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric("R² (test)", f"{metrics['r2_test']:.3f}",
                  delta=f"R² train: {metrics['r2_train']:.3f}", delta_color="off")
    with mcol2:
        st.metric("MAE (test)", f"{metrics['mae_test']:.3f}s")
    with mcol3:
        st.metric("Training rows", f"{metrics['n_train']} / {metrics['n_test']} test")

    # Predicted vs actual
    fig_pa = build_predicted_vs_actual(
        metrics["y_test"], metrics["y_pred"], r2=metrics["r2_test"]
    )
    if fig_pa is not None:
        st.plotly_chart(fig_pa, use_container_width=True,
                        config={"displayModeBar": False})

    st.divider()

    # ── Pit window optimizer (#13) ────────────────────────────────────────
    st.subheader(
        "Optimal Pit Window Predictor",
        help=(
            "Set current state di kanan → app simulate semua kandidat pit lap "
            "dari current+1 sampai akhir race, predict total remaining time "
            "untuk masing-masing, pilih yang minimum. Garis dotted = baseline "
            "kalau tidak pit sama sekali."
        ),
    )

    # Pre-compute track/air temp dari weather (untuk display + prediction)
    track_temp_val = 30.0
    air_temp_val = 25.0
    if weather_data is not None and len(weather_data) > 0:
        try:
            if "TrackTemp" in weather_data.columns and weather_data["TrackTemp"].notna().any():
                track_temp_val = float(weather_data["TrackTemp"].mean())
            if "AirTemp" in weather_data.columns and weather_data["AirTemp"].notna().any():
                air_temp_val = float(weather_data["AirTemp"].mean())
        except Exception:
            pass

    # Display weather context yang dipakai model
    st.caption(
        f"Model menggunakan rata-rata cuaca session ini: "
        f"<b style='color:#FF6B35'>Track {track_temp_val:.0f}°C</b> · "
        f"<b style='color:#3DA9FF'>Air {air_temp_val:.0f}°C</b>",
        unsafe_allow_html=True,
    )

    # Safe max_value: leave at least 3 laps remaining (1 pit candidate + 2 fresh laps)
    sim_max_lap = max(2, (total_laps_val or 100) - 3)

    pcol1, pcol2 = st.columns([1, 2])
    with pcol1:
        st.markdown("**Current state**")

        # Default values dari data
        default_lap = max(1, (total_laps_val or 30) // 3)
        default_lap = min(default_lap, sim_max_lap)

        sim_current_lap = st.number_input(
            "Current lap",
            min_value=1, max_value=sim_max_lap,
            value=default_lap,
            step=1, key="sim_lap",
            help=f"Saat ini balapan di lap berapa? Max {sim_max_lap} (sisa ≥3 lap untuk pit calculation).",
        )
        sim_tyre_age = st.number_input(
            "Current tyre age (laps)",
            min_value=0, max_value=60, value=15, step=1, key="sim_age",
            help="Sudah berapa lap pakai ban ini? Reset ke 0 berarti baru ganti.",
        )
        sim_stint = st.number_input(
            "Current stint #",
            min_value=1, max_value=5, value=1, step=1, key="sim_stint",
            help="Sudah berapa kali ganti ban + 1. Stint 1 = belum pernah pit, Stint 2 = sudah 1× pit, dst.",
        )
        sim_current_compound = st.selectbox(
            "Current compound",
            options=COMPOUND_LEVELS,
            index=1,  # Medium default
            key="sim_curr_comp",
        )
        sim_fresh_compound = st.selectbox(
            "Fresh compound (after pit)",
            options=COMPOUND_LEVELS,
            index=2,  # Hard default
            key="sim_fresh_comp",
        )
        sim_pit_loss = st.slider(
            "Pit stop loss (sec)",
            min_value=15.0, max_value=35.0, value=22.0, step=0.5,
            key="sim_pit_loss",
            help=(
                "Total waktu hilang di pit lane (drive in + stop + drive out). "
                "Modern F1: 20-25s typical (fastest ~18s, slow ~30s)."
            ),
        )

    with pcol2:
        opt_result = optimize_pit_window(
            model=model,
            current_lap=int(sim_current_lap),
            current_tyre_life=int(sim_tyre_age),
            current_compound=sim_current_compound,
            current_stint=int(sim_stint),
            fresh_compound=sim_fresh_compound,
            total_laps=total_laps_val or 50,
            track_temp=track_temp_val,
            air_temp=air_temp_val,
            pit_loss_seconds=float(sim_pit_loss),
        )

        if opt_result is None:
            st.warning("Could not compute pit optimization (check inputs).")
        else:
            # Summary metrics
            ocol1, ocol2 = st.columns(2)
            with ocol1:
                st.metric(
                    "Optimal pit lap",
                    f"Lap {opt_result['optimal_pit_lap']}",
                )
            with ocol2:
                saving = opt_result["saving"]
                if saving > 0:
                    delta_str = f"Saves {saving:.1f}s vs no-pit"
                    delta_color = "normal"
                else:
                    delta_str = f"No-pit is {-saving:.1f}s faster"
                    delta_color = "inverse"
                st.metric(
                    "Best total time",
                    f"{opt_result['best_total']:.1f}s",
                    delta=delta_str,
                    delta_color=delta_color,
                )

            # Pit window chart
            fig_pw = build_pit_window_chart(
                opt_result["candidates"],
                opt_result["optimal_pit_lap"],
                opt_result["baseline_no_pit"],
            )
            if fig_pw is not None:
                st.plotly_chart(fig_pw, use_container_width=True,
                                config={"displayModeBar": False})

    st.divider()

    # ── Feature importance / SHAP (#15) ───────────────────────────────────
    st.subheader(
        "Feature importance",
        help=(
            "Apa yang paling memengaruhi prediksi lap time? Mean absolute "
            "impact per feature. Pakai SHAP kalau library tersedia (lebih "
            "robust), fallback ke XGBoost built-in feature importance (gain)."
        ),
    )

    importance_df = compute_feature_importance(model, X, feature_cols)
    if len(importance_df) > 0:
        fig_imp = build_feature_importance(importance_df)
        if fig_imp is not None:
            st.plotly_chart(fig_imp, use_container_width=True,
                            config={"displayModeBar": False})

        # Show method used
        method = importance_df["Method"].iloc[0] if "Method" in importance_df.columns else "?"
        st.caption(
            f"Computed via **{method}**. "
            f"{'SHAP = Shapley value, robust untuk tree models.' if method == 'SHAP' else 'Fallback: XGBoost split gain. Install shap untuk SHAP-based importance.'}"
        )
    else:
        st.info("Feature importance unavailable.")

    st.divider()

    # ── #16 Strategy backtest ────────────────────────────────────────────────
    st.subheader(
        "Strategy backtest",
        help=(
            "Untuk tiap driver yang pit 1× di race ini, predict optimal pit lap "
            "menggunakan trained model (assume start state = lap 1, fresh tyres, "
            "starting compound dari first stint). Compare ke actual pit lap. "
            "Δ positif = pit terlalu telat, negatif = terlalu awal, ≈0 = optimal."
        ),
    )

    backtest_df = _strategy_backtest(
        model=model,
        stints_df=stints,
        pit_laps_df=pit_laps,
        total_laps=total_laps_val or 50,
        track_temp=track_temp_val,
        air_temp=air_temp_val,
        pit_loss=22.0,
    )
    if len(backtest_df) > 0:
        # Filter ke selected drivers
        if selected_drivers:
            backtest_df = backtest_df[backtest_df["Driver"].isin(selected_drivers)]

        if len(backtest_df) > 0:
            def _highlight_verdict(row):
                styles = [""] * len(row)
                if "Δ (laps)" in row.index:
                    delta_val = row["Δ (laps)"]
                    idx_v = list(row.index).index("Verdict")
                    if abs(delta_val) <= 1:
                        styles[idx_v] = (
                            "background-color: rgba(57,181,74,0.22); "
                            "color: #39B54A; font-weight: 700"
                        )
                    elif abs(delta_val) <= 3:
                        styles[idx_v] = (
                            "background-color: rgba(255,200,0,0.18); "
                            "color: #FFC800; font-weight: 700"
                        )
                    else:
                        styles[idx_v] = (
                            "background-color: rgba(220,50,50,0.18); "
                            "color: #DC3545; font-weight: 700"
                        )
                return styles

            try:
                styled = backtest_df.style.apply(_highlight_verdict, axis=1)
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(backtest_df, use_container_width=True, hide_index=True)

            st.caption(
                "Disclaimer: model trained pada data session ini sendiri "
                "(in-sample), jadi backtest mengukur 'apakah pit timing match "
                "dengan optimum dari distribution data, BUKAN out-of-sample "
                "prediction'. Untuk realistic backtest, perlu train pada race "
                "berbeda lalu test di race ini."
            )
        else:
            st.info("No 1-stop drivers in selected filter for backtest.")
    else:
        st.info(
            "Strategy backtest unavailable — perlu minimal beberapa driver "
            "dengan 1-stop strategy + trained model."
        )

    st.divider()
