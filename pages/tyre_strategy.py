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
from src.utils.config import (
    COMPOUND_COLORS,
    DRIVER_COLORS,
    SESSION_LABELS,
    SUPPORTED_YEARS,
)


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
