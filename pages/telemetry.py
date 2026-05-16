"""
Telemetry Analyzer Page
========================
Feature: Lap-by-lap telemetry comparison for up to 3 drivers (fastest atau lap pilih sendiri).
Channels: Speed, Throttle, Brake, Gear, RPM
Extras:
  - Lap summary cards (lap time, top speed, %throttle, %brake, gear shifts,
    tyre compound chip, weather snapshot)
  - Sector times table (fastest highlighted ungu, F1 broadcast convention)
  - Multichannel chart dengan corner labels (T1, T2, ...), sector splits (S2, S3),
    dan DRS zone overlay
  - Time delta chart (driver 2 vs driver 1)
  - Per-driver track speed heatmap
  - Track dominance map (mini-sector winner per driver)
  - Lap time distribution violin (semua clean laps di session)
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline.loader import (
    load_session,
    get_laps,
    get_session_drivers,
    get_available_rounds,
)
from src.viz.telemetry_charts import (
    build_telemetry_multichannel,
    build_delta_time,
    build_track_speed_map,
    build_track_dominance_map,
    build_lap_consistency_chart,
    build_lap_time_dist,
)
from src.utils.config import (
    COMPOUND_COLORS,
    DRIVER_COLORS,
    SESSION_LABELS,
    SUPPORTED_YEARS,
)


def _fmt_laptime(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:06.3f}"


def _fmt_sector(seconds) -> str:
    """Format sector time. Sektor biasanya <60s, total bisa >60s."""
    if pd.isna(seconds):
        return "—"
    s = float(seconds)
    if s >= 60:
        m = int(s // 60)
        return f"{m}:{s % 60:06.3f}"
    return f"{s:.3f}"


def _get_valid_lap_numbers(session, driver: str) -> list[int]:
    """Return ascending list of lap numbers with valid LapTime for the driver."""
    try:
        laps = session.laps.pick_drivers(driver)
        valid = laps[laps["LapTime"].notna()]
        return sorted(valid["LapNumber"].dropna().astype(int).unique().tolist())
    except Exception:
        return []


def _fetch_lap_data(session, driver: str, lap_choice: str) -> dict | None:
    """
    Resolve a driver's lap (fastest or specific) → fetch telemetry + position.
    Returns dict {lap, tel, pos_tel} or None kalau data unavailable.
    """
    try:
        driver_laps = session.laps.pick_drivers(driver)
    except Exception:
        return None

    if lap_choice == "Fastest":
        try:
            fl = driver_laps.pick_fastest()
        except Exception:
            return None
    else:
        try:
            lap_num = int(lap_choice.split()[-1])
        except (ValueError, IndexError):
            return None
        match = driver_laps[driver_laps["LapNumber"] == lap_num]
        if len(match) == 0:
            return None
        fl = match.iloc[0]

    if fl is None or pd.isna(fl.get("LapTime")):
        return None

    try:
        tel = fl.get_car_data().add_distance()
        if "nGear" in tel.columns and "Gear" not in tel.columns:
            tel = tel.rename(columns={"nGear": "Gear"})
        max_dist = tel["Distance"].max()
        if max_dist > 0:
            tel["DistanceNorm"] = tel["Distance"] / max_dist
    except Exception:
        return None

    pos_tel = None
    try:
        pos = fl.get_pos_data()
        if pos is not None and "Speed" in tel.columns:
            merged = pd.merge_asof(
                pos.sort_values("Time"),
                tel[["Time", "Speed"]].sort_values("Time"),
                on="Time",
                direction="nearest",
            ).dropna(subset=["Speed", "X", "Y"])
            if len(merged) > 0:
                pos_tel = merged
    except Exception:
        pass

    return {"lap": fl, "tel": tel, "pos_tel": pos_tel}


def _compute_lap_stats(tel: pd.DataFrame) -> dict:
    """Return per-lap summary stats dari telemetry samples."""
    stats: dict = {}
    if len(tel) == 0:
        return stats
    if "Speed" in tel.columns:
        stats["top_speed"] = float(tel["Speed"].max())
        stats["avg_speed"] = float(tel["Speed"].mean())
    if "Throttle" in tel.columns:
        # Threshold 99 — FastF1 kadang record 99.x sebagai "full throttle"
        stats["full_throttle_pct"] = float((tel["Throttle"] >= 99).mean() * 100)
    if "Brake" in tel.columns:
        stats["brake_pct"] = float(tel["Brake"].astype(float).mean() * 100)
    if "Gear" in tel.columns and len(tel) > 1:
        # Count gear shift events: di mana gear berubah dari sample sebelumnya
        diffs = tel["Gear"].astype(float).diff().fillna(0)
        stats["gear_changes"] = int((diffs != 0).sum())
    return stats


def _get_weather_at(session, lap_end_time) -> dict | None:
    """Snapshot cuaca di waktu paling dekat dengan akhir lap."""
    try:
        weather = session.weather_data
    except Exception:
        return None
    if weather is None or len(weather) == 0 or pd.isna(lap_end_time):
        return None
    diffs = (weather["Time"] - lap_end_time).abs()
    row = weather.loc[diffs.idxmin()]
    return {
        "track_temp": row.get("TrackTemp"),
        "air_temp":   row.get("AirTemp"),
        "humidity":   row.get("Humidity"),
        "wind_speed": row.get("WindSpeed"),
        "rainfall":   row.get("Rainfall"),
    }


def _sector_end_distances(lap, tel: pd.DataFrame) -> list[float] | None:
    """
    Distance (m) di akhir S1 dan akhir S2 sepanjang lap ini.
    Cara hitung: cari sample telemetry pada timestamp t0 + S1Time dan
    t0 + S1Time + S2Time, lalu interpolasi Distance-nya.
    Return None kalau data sektor tidak tersedia.
    """
    s1 = lap.get("Sector1Time")
    s2 = lap.get("Sector2Time")
    if pd.isna(s1) or pd.isna(s2) or len(tel) < 2:
        return None

    times = tel["Time"].dt.total_seconds().values
    dists = tel["Distance"].values
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    dists = dists[sort_idx]

    t0 = times[0]
    s1_end = t0 + s1.total_seconds()
    s2_end = t0 + s1.total_seconds() + s2.total_seconds()
    return [
        float(np.interp(s1_end, times, dists)),
        float(np.interp(s2_end, times, dists)),
    ]


def _find_drs_zones(tel: pd.DataFrame) -> list[tuple[float, float]]:
    """
    Return list of (start_dist, end_dist) di mana DRS aktif sepanjang lap ini.
    Konvensi FastF1: DRS values 10/12/14 = active, lainnya = off/eligible.
    """
    if "DRS" not in tel.columns or len(tel) == 0:
        return []
    is_active = tel["DRS"].isin([10, 12, 14]).values
    if not is_active.any():
        return []
    distances = tel["Distance"].values
    zones: list[tuple[float, float]] = []
    in_zone = False
    start = 0.0
    for i, active in enumerate(is_active):
        if active and not in_zone:
            in_zone = True
            start = float(distances[i])
        elif not active and in_zone:
            in_zone = False
            zones.append((start, float(distances[i - 1])))
    if in_zone:
        zones.append((start, float(distances[-1])))
    return zones


def _brake_points_per_corner(
    corners_df: pd.DataFrame | None,
    tel: pd.DataFrame,
    lookback_m: float = 250.0,
) -> dict[int, float]:
    """
    Untuk tiap corner, cari titik di mana driver mulai brake (transisi
    Brake False→True) yang paling dekat sebelum corner itu, max
    `lookback_m` meter ke belakang.

    Return {corner_number: distance_before_corner_in_meters}.
    Smaller value = brake lebih telat (better — late braking advantage).
    """
    if (corners_df is None or len(corners_df) == 0
            or "Brake" not in tel.columns or len(tel) < 2
            or "Distance" not in tel.columns):
        return {}

    tel_s = tel.sort_values("Distance").reset_index(drop=True)
    distances = tel_s["Distance"].values
    is_brake = tel_s["Brake"].astype(bool).values

    # Brake-on transitions (False → True)
    brake_starts: list[float] = []
    prev = False
    for i, b in enumerate(is_brake):
        if b and not prev:
            brake_starts.append(float(distances[i]))
        prev = b
    if not brake_starts:
        return {}
    brake_arr = np.array(brake_starts)

    result: dict[int, float] = {}
    for _, c in corners_df.iterrows():
        num = c.get("Number")
        c_dist = c.get("Distance")
        if pd.isna(num) or pd.isna(c_dist):
            continue
        cd = float(c_dist)
        valid = brake_arr[(brake_arr < cd) & (cd - brake_arr <= lookback_m)]
        if len(valid) > 0:
            closest = float(valid.max())  # latest brake-on before corner
            result[int(num)] = cd - closest
    return result


def _tyre_chip_html(compound: str | None, tyre_life: int | None) -> str:
    """Render compound sebagai colored dot + nama (warna Pirelli dari config)."""
    if not compound:
        return ""
    color = COMPOUND_COLORS.get(compound.upper(), "#888")
    chip = (
        f"<span style='display:inline-flex;align-items:center;gap:5px;vertical-align:middle'>"
        f"<span style='width:8px;height:8px;border-radius:50%;background:{color};"
        f"display:inline-block;box-shadow:0 0 4px {color}80'></span>"
        f"<span style='color:{color};font-weight:600'>{compound.title()}</span>"
        f"</span>"
    )
    if tyre_life is not None:
        chip += f" <span style='color:#777'>· {tyre_life} laps</span>"
    return chip


def render():
    st.title("Telemetry Analyzer")

    # Subtitle: baca tahun dari state — selectbox di bawah pakai key yang sama,
    # jadi nilainya konsisten dalam render-pass ini.
    _current_year = st.session_state.get("selected_year", 2024)
    st.markdown(
        f"<p style='color:#444;margin-top:-16px;font-family:Barlow,sans-serif;font-size:13px;letter-spacing:0.05em;text-transform:uppercase'>Lap-by-Lap · Car Data · {_current_year} Season</p>",
        unsafe_allow_html=True,
    )

    # ── Session selectors ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        selected_year = st.selectbox(
            "Season",
            options=sorted(SUPPORTED_YEARS, reverse=True),
            index=0,
            key="selected_year",  # shared key — sidebar & page lain ikut sinkron
        )

    with col2:
        rounds = get_available_rounds(selected_year)
        default_idx = 3 if len(rounds) > 3 else 0
        gp_label = st.selectbox(
            "Grand Prix",
            options=[f"R{k} · {v}" for k, v in rounds.items()],
            index=default_idx,
        )
        gp_round = int(gp_label.split("·")[0].strip()[1:])
        gp_name  = rounds[gp_round]

    with col3:
        session_type = st.selectbox(
            "Session",
            options=["Q", "R"],
            format_func=lambda x: SESSION_LABELS[x],
        )

    st.divider()
    load_btn = st.button("Load Session", type="primary")

    st.divider()

    # ── Session load ──────────────────────────────────────────────────────────
    session_id  = f"{selected_year}_{gp_name}_{session_type}"
    session_key = f"session_{session_id}"

    if load_btn or session_key in st.session_state:
        with st.spinner(f"Loading {gp_name} {SESSION_LABELS[session_type]} {selected_year}..."):
            session = load_session(selected_year, gp_name, session_type)

        if session is None:
            st.error("Session data unavailable. Try a different round.")
            return

        st.session_state[session_key] = True

        # ── Compare-with-other-session toggle (Q vs R) ────────────────────────
        # Swap lap source dari sesi yang user pilih ke sesi pasangannya
        # (Q ↔ R). Useful buat compare race pace vs qualifying pace driver
        # yang sama di GP yang sama.
        if session_type in ("R", "Q"):
            _other_type = "Q" if session_type == "R" else "R"
            _other_session_id = f"{selected_year}_{gp_name}_{_other_type}"

            use_other_session = st.checkbox(
                f"Compare with {SESSION_LABELS[_other_type]} session "
                f"— replace lap source",
                value=False, key="use_other_session",
                help=(
                    f"Kalau dicentang, lap source dipindah dari "
                    f"{SESSION_LABELS[session_type]} ke "
                    f"{SESSION_LABELS[_other_type]}. Berguna buat compare "
                    f"race-pace vs qualifying-pace driver yang sama."
                ),
            )
            if use_other_session:
                with st.spinner(f"Loading {SESSION_LABELS[_other_type]}..."):
                    _other_session = load_session(
                        selected_year, gp_name, _other_type
                    )
                if _other_session is not None:
                    # Swap binding — semua downstream pakai sesi baru ini
                    session      = _other_session
                    session_id   = _other_session_id
                    session_type = _other_type
                    st.info(
                        f"Compare mode aktif — lap & data dari "
                        f"**{SESSION_LABELS[_other_type]}** session."
                    )
                else:
                    st.warning(
                        f"{SESSION_LABELS[_other_type]} session unavailable "
                        f"— comparison disabled."
                    )

        # ── Driver selectors ──────────────────────────────────────────────────
        drivers = get_session_drivers(session, session_id)

        if not drivers:
            st.error("No driver data found in this session.")
            return

        st.markdown("### Driver & lap selection")
        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            d1 = st.selectbox("Driver 1 (reference)", drivers,
                              index=drivers.index("VER") if "VER" in drivers else 0,
                              key="d1")
            l1_opts = ["Fastest"] + [f"Lap {n}" for n in _get_valid_lap_numbers(session, d1)]
            l1 = st.selectbox("Lap", l1_opts, key="l1")

        with dcol2:
            d2_opts = [d for d in drivers if d != d1]
            d2 = st.selectbox("Driver 2", d2_opts,
                              index=d2_opts.index("LEC") if "LEC" in d2_opts else 0,
                              key="d2")
            l2_opts = ["Fastest"] + [f"Lap {n}" for n in _get_valid_lap_numbers(session, d2)]
            l2 = st.selectbox("Lap", l2_opts, key="l2")

        with dcol3:
            d3_opts = ["None"] + [d for d in drivers if d not in (d1, d2)]
            d3_raw  = st.selectbox("Driver 3 (optional)", d3_opts, key="d3")
            d3 = None if d3_raw == "None" else d3_raw
            if d3:
                l3_opts = ["Fastest"] + [f"Lap {n}" for n in _get_valid_lap_numbers(session, d3)]
                l3 = st.selectbox("Lap", l3_opts, key="l3")
            else:
                l3 = None

        selected_drivers = [d for d in [d1, d2, d3] if d]
        lap_choices = {d1: l1, d2: l2}
        if d3:
            lap_choices[d3] = l3

        # ── Channel selector ──────────────────────────────────────────────────
        st.markdown("### Telemetry channels")
        all_channels = ["Speed", "Throttle", "Brake", "Gear", "RPM"]

        ch_cols = st.columns(len(all_channels))
        selected_channels = []
        defaults = {"Speed": True, "Throttle": True, "Brake": True,
                    "Gear": False, "RPM": False}

        for i, ch in enumerate(all_channels):
            with ch_cols[i]:
                if st.checkbox(ch, value=defaults.get(ch, False)):
                    selected_channels.append(ch)

        if not selected_channels:
            st.warning("Select at least one telemetry channel.")
            return

        st.divider()

        # ── Fetch laps + telemetry per driver ─────────────────────────────────
        # driver_data[driver] = {lap, tel, pos_tel, stats, weather}
        driver_data: dict[str, dict] = {}

        fetch_progress = st.progress(0, text="Fetching telemetry...")
        n = len(selected_drivers)

        for idx, driver in enumerate(selected_drivers):
            choice = lap_choices.get(driver, "Fastest")
            fetch_progress.progress((idx + 1) / n, text=f"Loading {driver} ({choice})...")

            data = _fetch_lap_data(session, driver, choice)
            if data is None:
                st.warning(f"No telemetry for {driver} ({choice}). Skipping.")
                continue

            data["stats"]   = _compute_lap_stats(data["tel"])
            data["weather"] = _get_weather_at(session, data["lap"].get("Time"))
            driver_data[driver] = data

        fetch_progress.empty()

        # Backward-compat aliases — sisa kode existing masih pakai nama-nama ini
        tel_data     = {d: dd["tel"] for d, dd in driver_data.items()}
        tel_pos_data = {d: dd["pos_tel"] for d, dd in driver_data.items() if dd["pos_tel"] is not None}
        lap_times    = {d: dd["lap"]["LapTime"].total_seconds() for d, dd in driver_data.items()}

        if not tel_data:
            st.error("No telemetry data available for selected drivers.")
            return

        # ── Lap summary cards ─────────────────────────────────────────────────
        st.markdown("### Lap summary")
        active_drivers_for_cards = [d for d in selected_drivers if d in driver_data]
        metric_cols = st.columns(len(active_drivers_for_cards))

        try:
            total_session_laps = int(session.laps["LapNumber"].max())
        except Exception:
            total_session_laps = None

        for i, driver in enumerate(active_drivers_for_cards):
            data  = driver_data[driver]
            lap   = data["lap"]
            stats = data["stats"]
            wx    = data["weather"]
            color = DRIVER_COLORS.get(driver, "#AAAAAA")

            # Konteks: Lap N/Total · Compound (TyreLife) · Track/Air temp
            lap_n   = int(lap["LapNumber"]) if pd.notna(lap.get("LapNumber")) else None
            lap_str = (
                f"Lap {lap_n}/{total_session_laps}" if lap_n and total_session_laps
                else (f"Lap {lap_n}" if lap_n else "Lap —")
            )
            ctx_parts = [lap_str]

            compound = lap.get("Compound") if pd.notna(lap.get("Compound")) else None
            tyre_life = (
                int(lap["TyreLife"]) if pd.notna(lap.get("TyreLife")) else None
            )
            tyre_html = _tyre_chip_html(compound, tyre_life)
            if tyre_html:
                ctx_parts.append(tyre_html)

            # Weather: track/air temp + humidity + wind + rainfall flag
            if wx:
                if pd.notna(wx.get("track_temp")):
                    ctx_parts.append(f"Track {wx['track_temp']:.0f}°C")
                if pd.notna(wx.get("air_temp")):
                    ctx_parts.append(f"Air {wx['air_temp']:.0f}°C")
                if pd.notna(wx.get("humidity")):
                    ctx_parts.append(f"Hum {wx['humidity']:.0f}%")
                if pd.notna(wx.get("wind_speed")):
                    ctx_parts.append(f"Wind {wx['wind_speed']:.0f}m/s")
                # Rainfall flag — boolean True kalau hujan
                if wx.get("rainfall"):
                    ctx_parts.append(
                        "<span style='color:#3DA9FF;font-weight:700'>RAIN</span>"
                    )

            ctx_str = " &middot; ".join(ctx_parts)

            # Stats: Top speed · % full throttle · % brake · gear shifts
            stats_parts = []
            if "top_speed" in stats:
                stats_parts.append(
                    f"<span style='color:#666'>Top</span> "
                    f"<span style='color:#FFF;font-weight:600'>{stats['top_speed']:.0f}</span> "
                    f"<span style='color:#666;font-size:10px'>km/h</span>"
                )
            if "full_throttle_pct" in stats:
                stats_parts.append(
                    f"<span style='color:#666'>Throttle</span> "
                    f"<span style='color:#FFF;font-weight:600'>{stats['full_throttle_pct']:.0f}%</span>"
                )
            if "brake_pct" in stats:
                stats_parts.append(
                    f"<span style='color:#666'>Brake</span> "
                    f"<span style='color:#FFF;font-weight:600'>{stats['brake_pct']:.0f}%</span>"
                )
            if "gear_changes" in stats:
                stats_parts.append(
                    f"<span style='color:#666'>Shifts</span> "
                    f"<span style='color:#FFF;font-weight:600'>{stats['gear_changes']}</span>"
                )
            stats_str = " &nbsp;&middot;&nbsp; ".join(stats_parts)

            with metric_cols[i]:
                st.markdown(
                    f"""
                    <div style='border-left:3px solid {color};padding:6px 0 6px 14px;background:rgba(255,255,255,0.015);border-radius:0 4px 4px 0'>
                        <div style='color:{color};font-size:11px;font-weight:700;letter-spacing:0.12em;margin-bottom:4px'>{driver}</div>
                        <div style='font-size:28px;font-weight:600;font-family:monospace;line-height:1.05;margin-bottom:8px'>
                            {_fmt_laptime(lap_times.get(driver))}
                        </div>
                        <div style='font-size:10.5px;color:#777;margin-bottom:8px;letter-spacing:0.02em;text-transform:uppercase;font-family:Barlow Condensed,sans-serif'>
                            {ctx_str}
                        </div>
                        <div style='font-size:11.5px;line-height:1.7'>{stats_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Sector times ──────────────────────────────────────────────────────
        st.subheader(
            "Sector times",
            help=(
                "S1 + S2 + S3 = LapTime. Fastest tiap sector di-highlight ungu "
                "(konvensi F1 broadcast). Theoretical Best = jumlah best sector "
                "lifetime di session itu — mirror lap teoretis tercepat."
            ),
        )

        sector_rows: dict[str, dict] = {}
        for sec_label, sec_col in [
            ("S1", "Sector1Time"),
            ("S2", "Sector2Time"),
            ("S3", "Sector3Time"),
            ("Total", "LapTime"),
        ]:
            sector_rows[sec_label] = {}
            for driver in active_drivers_for_cards:
                t = driver_data[driver]["lap"].get(sec_col)
                sector_rows[sec_label][driver] = (
                    t.total_seconds() if pd.notna(t) else float("nan")
                )

        # Theoretical Best — sum of best sectors lifetime di session ini per driver
        theo_best: dict[str, float] = {}
        for driver in active_drivers_for_cards:
            try:
                all_d_laps = session.laps.pick_drivers(driver)
                s1_min = all_d_laps["Sector1Time"].dropna().min()
                s2_min = all_d_laps["Sector2Time"].dropna().min()
                s3_min = all_d_laps["Sector3Time"].dropna().min()
                if pd.notna(s1_min) and pd.notna(s2_min) and pd.notna(s3_min):
                    theo_best[driver] = (
                        s1_min.total_seconds()
                        + s2_min.total_seconds()
                        + s3_min.total_seconds()
                    )
                else:
                    theo_best[driver] = float("nan")
            except Exception:
                theo_best[driver] = float("nan")
        sector_rows["Theo. Best"] = theo_best

        sector_df = pd.DataFrame(sector_rows).T  # rows = sectors, cols = drivers

        def _highlight_fastest(row):
            valid = row.dropna()
            if len(valid) == 0:
                return [""] * len(row)
            fastest_val = valid.min()
            # Purple = konvensi F1 broadcast untuk fastest sector
            return [
                "background-color: rgba(176,38,255,0.32); font-weight: 700; color: #FFF;"
                if pd.notna(v) and v == fastest_val else ""
                for v in row
            ]

        styled = (
            sector_df.style
            .apply(_highlight_fastest, axis=1)
            .format(_fmt_sector)
        )
        st.dataframe(styled, use_container_width=True)

        st.divider()

        # ── Multi-channel telemetry chart ─────────────────────────────────────
        st.subheader(
            "Multichannel telemetry",
            help=(
                "Trace tiap channel (Speed, Throttle, dll) terhadap distance "
                "sepanjang lap. Overlay corner labels (T1, T2, ...), sector "
                "splits (S2, S3 dashed gold), dan zona DRS aktif (shaded biru)."
            ),
        )

        # Circuit info untuk corner labels (best-effort — beberapa session lama
        # mungkin tidak ada data corner-nya)
        try:
            circuit_info = session.get_circuit_info()
            corners_df = circuit_info.corners if circuit_info is not None else None
        except Exception:
            corners_df = None

        # Sector split + DRS zones — hitung dari driver pertama yang punya data
        ref_driver = active_drivers_for_cards[0] if active_drivers_for_cards else None
        sector_dists = None
        drs_zones = None
        if ref_driver is not None:
            ref_lap = driver_data[ref_driver]["lap"]
            ref_tel = driver_data[ref_driver]["tel"]
            sector_dists = _sector_end_distances(ref_lap, ref_tel)
            drs_zones    = _find_drs_zones(ref_tel)

        fig_multi = build_telemetry_multichannel(
            tel_data, selected_channels,
            corners=corners_df,
            sector_distances=sector_dists,
            drs_zones=drs_zones,
        )
        st.plotly_chart(fig_multi, use_container_width=True, config={"displayModeBar": False})

        # ── Delta time (only when 2+ drivers selected) ────────────────────────
        active_drivers = [d for d in selected_drivers if d in tel_data]
        if len(active_drivers) >= 2:
            st.subheader(
                "Time delta",
                help=(
                    "Selisih waktu kumulatif driver lain vs driver 1 (reference) "
                    "di setiap titik trek. Positif = lebih lambat dari reference. "
                    "Saat 1 cmp: region hijau/merah menunjukkan faster/slower. "
                    "Saat 2 cmp: tiap driver punya line warna sendiri."
                ),
            )
            ref_d = active_drivers[0]
            cmp_drivers = active_drivers[1:]
            if len(cmp_drivers) == 1:
                caption_text = (
                    f"Positive = {cmp_drivers[0]} is slower than {ref_d} "
                    f"at that track position. Green regions = {cmp_drivers[0]} faster."
                )
            else:
                cmp_list_str = ", ".join(cmp_drivers)
                caption_text = (
                    f"Positive = slower than {ref_d} (reference). "
                    f"Comparing: {cmp_list_str}."
                )
            st.caption(caption_text)

            tel_cmps = {d: tel_data[d] for d in cmp_drivers}
            fig_delta = build_delta_time(
                tel_data[ref_d],
                tel_cmps,
                ref_d,
            )
            st.plotly_chart(fig_delta, use_container_width=True, config={"displayModeBar": False})

        # ── Track speed maps ──────────────────────────────────────────────────
        if tel_pos_data:
            st.subheader(
                "Track map — speed heatmap",
                help=(
                    "Bentuk trek lap ini, dengan tiap titik di-color by kecepatan "
                    "(merah = pelan, biru = kencang). Berguna untuk lihat speed "
                    "profile per corner."
                ),
            )
            map_cols = st.columns(len(tel_pos_data))
            for i, (driver, pos_tel) in enumerate(tel_pos_data.items()):
                with map_cols[i]:
                    if "X" in pos_tel.columns and "Speed" in pos_tel.columns:
                        fig_map = build_track_speed_map(pos_tel, driver)
                        if fig_map:
                            st.plotly_chart(
                                fig_map, use_container_width=True,
                                config={"displayModeBar": False}
                            )

        # ── Track dominance map (mini-sector winner per driver) ──────────────
        if len(tel_pos_data) >= 2 and len(tel_data) >= 2:
            st.subheader(
                "Track dominance",
                help=(
                    "Trek dibagi 24 mini-sector. Tiap mini-sector di-color "
                    "dengan warna driver tercepat di sector itu. Cara cepat lihat "
                    "di mana tiap driver punya keunggulan kecepatan riil."
                ),
            )
            st.caption(
                "Setiap mini-sector di-color dengan warna driver tercepat di sector itu."
            )
            fig_dom = build_track_dominance_map(tel_pos_data, tel_data, n_sectors=24)
            if fig_dom is not None:
                st.plotly_chart(fig_dom, use_container_width=True, config={"displayModeBar": False})

        # ── Brake points comparison ───────────────────────────────────────────
        if corners_df is not None and len(active_drivers_for_cards) >= 2:
            st.subheader(
                "Brake points",
                help=(
                    "Jarak antara titik brake-on dengan corner di depan. Semakin "
                    "kecil = brake makin telat (late-braking advantage). Lookback "
                    "window 250m — kalau no brake event ditemukan dalam window itu, "
                    "ditampilkan '—'. Latest braker per corner di-highlight ungu."
                ),
            )

            brake_table: dict[str, dict[int, float]] = {}
            any_data = False
            for driver in active_drivers_for_cards:
                bp = _brake_points_per_corner(
                    corners_df, driver_data[driver]["tel"]
                )
                brake_table[driver] = bp
                if bp:
                    any_data = True

            if not any_data:
                st.info("No brake events detected near corners — check session data quality.")
            else:
                all_corner_nums = sorted(
                    corners_df["Number"].dropna().astype(int).unique().tolist()
                )
                bp_rows: dict[str, dict] = {}
                for cn in all_corner_nums:
                    bp_rows[f"T{cn}"] = {
                        d: brake_table.get(d, {}).get(cn, float("nan"))
                        for d in active_drivers_for_cards
                    }

                bp_df = pd.DataFrame(bp_rows).T

                def _fmt_brake_dist(v) -> str:
                    if pd.isna(v):
                        return "—"
                    return f"{float(v):.0f}m"

                def _highlight_latest_braker(row):
                    valid = row.dropna()
                    if len(valid) == 0:
                        return [""] * len(row)
                    # Smallest distance = latest brake = paling jago
                    latest = valid.min()
                    return [
                        "background-color: rgba(176,38,255,0.32); font-weight: 700; color: #FFF;"
                        if pd.notna(v) and v == latest else ""
                        for v in row
                    ]

                styled_bp = (
                    bp_df.style
                    .apply(_highlight_latest_braker, axis=1)
                    .format(_fmt_brake_dist)
                )
                st.dataframe(styled_bp, use_container_width=True)

        st.divider()

        # ── Lap consistency (multi-lap overlay) ───────────────────────────────
        st.subheader(
            "Lap consistency",
            help=(
                "Overlay speed trace dari N lap tercepat 1 driver. Semakin overlap "
                "semua trace = pace makin konsisten. Lap tercepat solid penuh; "
                "lap-lap berikutnya semakin ghosted (transparant)."
            ),
        )
        if active_drivers_for_cards:
            ccol1, ccol2 = st.columns([2, 1])
            with ccol1:
                consist_driver = st.selectbox(
                    "Driver",
                    active_drivers_for_cards,
                    key="consist_driver",
                )
            with ccol2:
                n_laps_overlay = st.slider(
                    "Lap count", min_value=2, max_value=10, value=5,
                    key="consist_n",
                )

            try:
                all_d_laps = (
                    session.laps.pick_drivers(consist_driver)
                    .dropna(subset=["LapTime"])
                )
                top_n = all_d_laps.nsmallest(int(n_laps_overlay), "LapTime")
            except Exception:
                top_n = pd.DataFrame()

            consist_laps: list[dict] = []
            if len(top_n) > 0:
                for _, lap in top_n.iterrows():
                    try:
                        tel = lap.get_car_data().add_distance()
                        if "nGear" in tel.columns and "Gear" not in tel.columns:
                            tel = tel.rename(columns={"nGear": "Gear"})
                        consist_laps.append({
                            "lap_num": int(lap["LapNumber"]),
                            "lap_time": lap["LapTime"].total_seconds(),
                            "tel": tel,
                        })
                    except Exception:
                        continue

            if consist_laps:
                fig_consist = build_lap_consistency_chart(consist_driver, consist_laps)
                if fig_consist is not None:
                    st.plotly_chart(
                        fig_consist, use_container_width=True,
                        config={"displayModeBar": False},
                    )
            else:
                st.info(f"Tidak cukup data lap untuk {consist_driver}.")

        st.divider()

        # ── Lap time distribution ─────────────────────────────────────────────
        st.subheader(
            "Lap time distribution (all clean laps)",
            help=(
                "Box plot tiap driver dari semua clean laps di session ini, "
                "disort by median pace (tercepat di atas). Box = IQR 25-75%, "
                "garis tebal = median, dashed = mean, dot di luar whisker = "
                "outlier lap. Pit-laps & outlier statistical sudah difilter."
            ),
        )
        with st.spinner("Computing lap statistics..."):
            laps_df = get_laps(session, session_id)
            laps_filtered = laps_df[laps_df["Driver"].isin(selected_drivers)]

        if len(laps_filtered) > 0:
            fig_dist = build_lap_time_dist(laps_filtered, selected_drivers)
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Not enough clean lap data for distribution chart (qualifying sessions have fewer laps).")

        # ── Raw data expander ─────────────────────────────────────────────────
        st.markdown("### Raw Telemetry Data")
        preferred_cols = ["Distance", "Speed", "Throttle", "Brake", "Gear", "RPM"]
        for driver, tel in tel_data.items():
            st.markdown(f"**{driver}** — {len(tel)} samples")
            cols = [c for c in preferred_cols if c in tel.columns]
            st.dataframe(
                tel[cols].head(100),
                use_container_width=True,
                hide_index=True,
            )

    else:
        # Landing state — nothing loaded yet
        st.markdown("""
        <div style='text-align:center;padding:80px 0;color:#444'>
            <div style='font-size:48px'>🏎️</div>
            <div style='font-size:18px;margin-top:16px'>Select a Grand Prix and session above</div>
            <div style='font-size:13px;margin-top:8px;color:#333'>
                Then hit <strong>Load Session</strong> to fetch telemetry data
            </div>
        </div>
        """, unsafe_allow_html=True)
