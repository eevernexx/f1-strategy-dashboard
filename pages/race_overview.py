"""
Race Overview Page
==================
Features:
- Race summary cards (total laps, race time, fastest lap, pit stops, weather)
- Driver of the Day banner (biggest position gain)
- Final classification (with Grid → Finish delta)
- Lap leader summary
- Race pace ranking, top speed leaderboard, sector pace ranking
- Pit stop summary table
- Lap-by-lap positions chart (with SC/VSC overlay)
- Race pace heatmap
- Gap to leader chart
"""

import pandas as pd
import streamlit as st

from src.pipeline.loader import (
    load_session,
    get_laps,
    get_available_rounds,
)
from src.viz.race_charts import (
    build_position_chart,
    build_gap_to_leader,
    build_fastest_laps_table,
    build_session_results,
    build_race_pace_ranking,
    build_top_speed_table,
    build_sector_pace_table,
    build_pit_stops_summary,
    build_pace_heatmap,
    build_weather_timeline,
)
from src.utils.config import (
    DRIVER_COLORS,
    SESSION_LABELS,
    SUPPORTED_YEARS,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _fmt_laptime(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    s = float(seconds)
    if s >= 60:
        m = int(s // 60)
        return f"{m}:{s % 60:06.3f}"
    return f"{s:.3f}"


def _fmt_sector(seconds) -> str:
    if pd.isna(seconds):
        return "—"
    s = float(seconds)
    if s >= 60:
        m = int(s // 60)
        return f"{m}:{s % 60:06.3f}"
    return f"{s:.3f}"


def _fmt_duration_hms(seconds: float | None) -> str:
    """Format detik jadi h:mm:ss.SSS untuk durasi race."""
    if seconds is None or pd.isna(seconds):
        return "—"
    s = float(seconds)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}:{m:02d}:{s % 60:06.3f}"


def _race_summary_stats(session) -> dict:
    """Session-wide stats untuk header summary cards."""
    stats: dict = {}

    try:
        stats["total_laps"] = int(session.laps["LapNumber"].max())
    except Exception:
        stats["total_laps"] = None

    # Fastest lap of race
    try:
        fl = session.laps.pick_fastest()
        if fl is not None and pd.notna(fl.get("LapTime")):
            stats["fastest_driver"] = str(fl.get("Driver", "—"))
            stats["fastest_time"] = fl["LapTime"].total_seconds()
            stats["fastest_lap_num"] = (
                int(fl["LapNumber"]) if pd.notna(fl.get("LapNumber")) else None
            )
    except Exception:
        pass

    # Total pit stops session-wide
    try:
        stats["total_pit_stops"] = int(session.laps["PitInTime"].notna().sum())
    except Exception:
        stats["total_pit_stops"] = None

    # Race duration — winner's race time (P1 dari results)
    try:
        results = session.results.dropna(subset=["Position"]).sort_values("Position")
        if len(results) > 0:
            winner_time = results.iloc[0].get("Time")
            if pd.notna(winner_time):
                stats["race_duration"] = winner_time.total_seconds()
    except Exception:
        pass

    # Weather avg
    try:
        wx = session.weather_data
        if wx is not None and len(wx) > 0:
            if "TrackTemp" in wx.columns and wx["TrackTemp"].notna().any():
                stats["avg_track_temp"] = float(wx["TrackTemp"].mean())
            if "AirTemp" in wx.columns and wx["AirTemp"].notna().any():
                stats["avg_air_temp"] = float(wx["AirTemp"].mean())
            if "Rainfall" in wx.columns:
                stats["any_rain"] = bool(wx["Rainfall"].any())
    except Exception:
        pass

    return stats


def _driver_of_the_day(session) -> dict | None:
    """
    Driver dengan position gain terbesar dari grid → finish.
    Return None kalau tidak ada gain valid (e.g., quali atau data tidak cukup).
    """
    try:
        res = session.results
        if res is None or len(res) == 0:
            return None
        df = res.dropna(subset=["Position", "GridPosition"]).copy()
    except Exception:
        return None
    if len(df) == 0:
        return None

    # GridPosition 0 = pit lane start, treat sebagai posisi 21 supaya jadi candidate kuat
    df["_grid"] = df["GridPosition"].astype(float).replace(0, 21)
    df["_finish"] = df["Position"].astype(int)
    df["_delta"] = df["_grid"] - df["_finish"]

    if df["_delta"].max() <= 0:
        return None

    row = df.loc[df["_delta"].idxmax()]
    grid_orig = row["GridPosition"]
    grid_label = "PL" if grid_orig == 0 else f"P{int(grid_orig)}"

    # Guard NaN untuk Abbreviation & TeamName
    abbr_val = row.get("Abbreviation")
    team_val = row.get("TeamName")
    return {
        "driver": str(abbr_val) if pd.notna(abbr_val) else "?",
        "team":   str(team_val) if pd.notna(team_val) else "?",
        "grid":   grid_label,
        "finish": f"P{int(row['Position'])}",
        "delta":  int(row["_delta"]),
    }


def _lap_leader_summary(session) -> dict:
    """Laps led per driver + total lead changes."""
    out = {"laps_led": {}, "lead_changes": 0, "total_laps_with_p1": 0}
    try:
        laps = session.laps[["Driver", "LapNumber", "Position"]].dropna(
            subset=["Position", "LapNumber"]
        )
    except Exception:
        return out

    p1 = laps[laps["Position"] == 1].sort_values("LapNumber")
    if len(p1) == 0:
        return out

    out["laps_led"] = (
        p1.groupby("Driver").size().sort_values(ascending=False).to_dict()
    )
    out["total_laps_with_p1"] = len(p1)

    # Lead changes
    prev = None
    changes = 0
    for d in p1["Driver"].tolist():
        if prev is not None and d != prev:
            changes += 1
        prev = d
    out["lead_changes"] = changes
    return out


def _race_events(session) -> list[tuple[str, int, int]]:
    """
    Parse SC / VSC / Red flag periods dari session.race_control_messages.
    Returns list of (event_type, start_lap, end_lap).
    """
    try:
        msgs = session.race_control_messages
    except Exception:
        return []
    if msgs is None or len(msgs) == 0:
        return []

    events: list[tuple[str, int, int]] = []
    sc_active = False;  sc_start: int | None = None
    vsc_active = False; vsc_start: int | None = None
    red_active = False; red_start: int | None = None
    last_lap = 0

    # Cari kolom Lap (FastF1 biasanya "Lap")
    lap_col = "Lap" if "Lap" in msgs.columns else None
    msg_col = "Message" if "Message" in msgs.columns else None
    if not lap_col or not msg_col:
        return []

    for _, row in msgs.iterrows():
        lap_val = row.get(lap_col)
        if pd.isna(lap_val):
            continue
        try:
            lap = int(lap_val)
        except (TypeError, ValueError):
            continue
        last_lap = max(last_lap, lap)
        text = str(row.get(msg_col, "")).upper()

        # VSC harus dicek SEBELUM SC karena string "SAFETY CAR" muncul di keduanya
        if "VIRTUAL SAFETY CAR DEPLOYED" in text:
            if not vsc_active:
                vsc_active = True; vsc_start = lap
        elif "VIRTUAL SAFETY CAR ENDING" in text:
            if vsc_active and vsc_start is not None:
                events.append(("VSC", vsc_start, lap))
                vsc_active = False
        elif "SAFETY CAR DEPLOYED" in text:
            if not sc_active:
                sc_active = True; sc_start = lap
        elif "SAFETY CAR ENDING" in text or "SAFETY CAR IN THIS LAP" in text:
            if sc_active and sc_start is not None:
                events.append(("SC", sc_start, lap))
                sc_active = False
        elif "RED FLAG" in text:
            if not red_active:
                red_active = True; red_start = lap
        elif "GREEN" in text and red_active:
            if red_start is not None:
                events.append(("RED", red_start, lap))
                red_active = False

    # Close any still-open events at last seen lap
    if sc_active and sc_start is not None:
        events.append(("SC", sc_start, last_lap))
    if vsc_active and vsc_start is not None:
        events.append(("VSC", vsc_start, last_lap))
    if red_active and red_start is not None:
        events.append(("RED", red_start, last_lap))

    return events


def _pit_stop_durations(session, drivers_filter: list[str] | None = None) -> list[dict]:
    """
    Numeric pit stop durations untuk leaderboard & team avg.
    Returns list of {driver, team, lap, duration_seconds}.
    """
    try:
        cols = ["Driver", "Team", "LapNumber", "PitInTime", "PitOutTime"]
        avail = [c for c in cols if c in session.laps.columns]
        if "PitInTime" not in avail or "PitOutTime" not in avail or "Driver" not in avail:
            return []
        all_laps = session.laps[avail].copy()
    except Exception:
        return []

    if drivers_filter:
        all_laps = all_laps[all_laps["Driver"].isin(drivers_filter)]

    out: list[dict] = []
    for driver, drv_laps in all_laps.groupby("Driver"):
        drv = drv_laps.sort_values("LapNumber").reset_index(drop=True)
        for i in range(len(drv)):
            row = drv.iloc[i]
            pit_in = row.get("PitInTime")
            if pd.isna(pit_in):
                continue
            pit_out = drv.iloc[i + 1].get("PitOutTime") if i + 1 < len(drv) else None
            if pd.isna(pit_out):
                continue
            try:
                duration = float((pit_out - pit_in).total_seconds())
            except Exception:
                continue
            if duration <= 0 or duration > 120:
                # Sanity check: pit stops > 2 menit = data anomaly atau retired
                continue
            # Guard NaN — kalau Team value NaN, fallback ke "—" (jangan jadi "nan")
            team_val = row.get("Team") if "Team" in drv.columns else None
            team_str = str(team_val) if pd.notna(team_val) else "—"
            out.append({
                "Driver": driver,
                "Team": team_str,
                "Lap": int(row["LapNumber"]) if pd.notna(row["LapNumber"]) else None,
                "DurationSeconds": duration,
            })
    return out


def _position_changes(session) -> pd.DataFrame:
    """
    Count on-track position changes per driver (exclude pit-related changes).
    Returns DataFrame: Driver, Gained, Lost, Net.
    """
    try:
        cols = ["Driver", "LapNumber", "Position", "PitInTime", "PitOutTime"]
        avail = [c for c in cols if c in session.laps.columns]
        if "Driver" not in avail or "Position" not in avail:
            return pd.DataFrame()
        laps = session.laps[avail].copy()
    except Exception:
        return pd.DataFrame()

    laps = laps.dropna(subset=["Position", "LapNumber"])
    if len(laps) == 0:
        return pd.DataFrame()

    rows = []
    for driver, drv_laps in laps.groupby("Driver"):
        drv = drv_laps.sort_values("LapNumber").reset_index(drop=True)
        gained = 0
        lost = 0
        for i in range(1, len(drv)):
            cur = drv.iloc[i]
            prev = drv.iloc[i - 1]
            # Skip changes yang melibatkan pit lap (in or out)
            if "PitInTime" in drv.columns and pd.notna(cur.get("PitInTime")):
                continue
            if "PitOutTime" in drv.columns and pd.notna(cur.get("PitOutTime")):
                continue
            if "PitInTime" in drv.columns and pd.notna(prev.get("PitInTime")):
                continue
            try:
                delta = int(prev["Position"]) - int(cur["Position"])
            except (TypeError, ValueError):
                continue
            if delta > 0:
                gained += delta
            elif delta < 0:
                lost += -delta
        rows.append({"Driver": driver, "Gained": gained, "Lost": lost, "Net": gained - lost})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Net", ascending=False).reset_index(drop=True)


def _battles(session, drs_range: float = 1.5, min_duration: int = 2) -> list[dict]:
    """
    Find lap-lap windows di mana 2 driver dalam <drs_range detik (DRS range).
    Return: list of {driver_a, driver_b, start_lap, end_lap, duration_laps}.
    """
    try:
        cols = ["Driver", "LapNumber", "Time"]
        if not all(c in session.laps.columns for c in cols):
            return []
        laps = session.laps[cols].copy()
    except Exception:
        return []

    laps = laps.dropna(subset=["Time", "LapNumber"])
    if len(laps) == 0:
        return []
    laps["ElapsedSeconds"] = laps["Time"].dt.total_seconds()

    try:
        pivot = laps.pivot_table(
            index="LapNumber", columns="Driver",
            values="ElapsedSeconds", aggfunc="first",
        )
    except Exception:
        return []
    if len(pivot) == 0 or len(pivot.columns) < 2:
        return []

    drivers = pivot.columns.tolist()
    all_laps_idx = pivot.index.tolist()
    battles: list[dict] = []

    for i, da in enumerate(drivers):
        for db in drivers[i + 1:]:
            try:
                gaps = (pivot[da] - pivot[db]).abs()
            except Exception:
                continue
            in_range = (gaps < drs_range).values

            # Cari continuous periods di mana keduanya in range
            in_p = False
            start_lap: int | None = None
            prev_lap: int | None = None
            for lap_idx, active in zip(all_laps_idx, in_range):
                if pd.isna(pivot.loc[lap_idx, da]) or pd.isna(pivot.loc[lap_idx, db]):
                    # Salah satu missing → end period kalau ada
                    if in_p and start_lap is not None and prev_lap is not None:
                        if (prev_lap - start_lap + 1) >= min_duration:
                            battles.append({
                                "Driver A": da, "Driver B": db,
                                "Start lap": int(start_lap),
                                "End lap": int(prev_lap),
                                "Duration (laps)": int(prev_lap - start_lap + 1),
                            })
                        in_p = False
                        start_lap = None
                    continue
                if active and not in_p:
                    in_p = True
                    start_lap = int(lap_idx)
                elif not active and in_p:
                    in_p = False
                    if start_lap is not None and prev_lap is not None:
                        if (prev_lap - start_lap + 1) >= min_duration:
                            battles.append({
                                "Driver A": da, "Driver B": db,
                                "Start lap": int(start_lap),
                                "End lap": int(prev_lap),
                                "Duration (laps)": int(prev_lap - start_lap + 1),
                            })
                    start_lap = None
                prev_lap = int(lap_idx)
            # Close trailing
            if in_p and start_lap is not None and prev_lap is not None:
                if (prev_lap - start_lap + 1) >= min_duration:
                    battles.append({
                        "Driver A": da, "Driver B": db,
                        "Start lap": int(start_lap),
                        "End lap": int(prev_lap),
                        "Duration (laps)": int(prev_lap - start_lap + 1),
                    })

    # Sort by duration descending
    battles.sort(key=lambda x: -x["Duration (laps)"])
    return battles


def _rain_laps(session) -> list[tuple[int, int]]:
    """Periods (start_lap, end_lap) di mana Rainfall terdeteksi."""
    try:
        wx = session.weather_data
        if wx is None or len(wx) == 0 or "Rainfall" not in wx.columns:
            return []
        laps = session.laps[["LapNumber", "LapStartTime", "Time"]].copy()
    except Exception:
        return []

    laps = laps.dropna(subset=["LapStartTime", "Time", "LapNumber"])
    if len(laps) == 0:
        return []

    rain_lap_nums: set[int] = set()
    wx_time = wx["Time"]
    for _, lap in laps.iterrows():
        try:
            mask = (wx_time >= lap["LapStartTime"]) & (wx_time <= lap["Time"])
            if wx.loc[mask, "Rainfall"].any():
                rain_lap_nums.add(int(lap["LapNumber"]))
        except Exception:
            continue

    if not rain_lap_nums:
        return []

    # Group consecutive laps jadi periods
    rl = sorted(rain_lap_nums)
    periods: list[tuple[int, int]] = []
    start = rl[0]
    prev = start
    for lap in rl[1:]:
        if lap == prev + 1:
            prev = lap
        else:
            periods.append((start, prev))
            start = lap
            prev = lap
    periods.append((start, prev))
    return periods


def _dnf_list(session) -> list[dict]:
    """Drivers yang tidak finish (status ≠ Finished/Lapped)."""
    try:
        res = session.results
        if res is None or len(res) == 0:
            return []
    except Exception:
        return []

    out = []
    for _, r in res.iterrows():
        status = str(r.get("Status", "")) if pd.notna(r.get("Status")) else ""
        # "Finished" or "+X Lap(s)" = finished. Else = DNF/DSQ/DNS/etc.
        if status == "Finished" or "Lap" in status:
            continue
        # Guard NaN untuk Abbreviation & TeamName
        abbr_val = r.get("Abbreviation")
        team_val = r.get("TeamName")
        out.append({
            "Driver":   str(abbr_val) if pd.notna(abbr_val) else "?",
            "Team":     str(team_val) if pd.notna(team_val) else "?",
            "Position": int(r["Position"]) if pd.notna(r.get("Position")) else None,
            "Status":   status or "—",
        })
    return out


def _penalty_list(session) -> list[dict]:
    """Parse penalty mentions dari race_control_messages."""
    try:
        msgs = session.race_control_messages
        if msgs is None or len(msgs) == 0:
            return []
    except Exception:
        return []

    lap_col = "Lap" if "Lap" in msgs.columns else None
    msg_col = "Message" if "Message" in msgs.columns else None
    if not lap_col or not msg_col:
        return []

    out = []
    for _, row in msgs.iterrows():
        text = str(row.get(msg_col, ""))
        upper = text.upper()
        # Filter: cari pesan dengan kata "PENALTY" yang impose, bukan info umum
        if "PENALTY" not in upper:
            continue
        # Skip pesan yang cuma confirm/review (no penalty given)
        if any(skip in upper for skip in ("NO FURTHER ACTION", "UNDER INVESTIGATION",
                                          "BEING INVESTIGATED")):
            continue
        lap_val = row.get(lap_col)
        out.append({
            "Lap":     int(lap_val) if pd.notna(lap_val) else None,
            "Message": text,
        })
    return out


def _race_summary_narrative(session) -> str | None:
    """
    Auto-generate 2-4 sentence narrative dari data race.
    Format: winner + gap, race interruptions, DNFs, fastest lap.
    """
    sentences: list[str] = []

    # 1. Winner + gap to P2
    try:
        res = session.results.dropna(subset=["Position"]).sort_values("Position")
        if len(res) >= 1:
            winner = res.iloc[0]
            winner_name = winner.get("Abbreviation")
            winner_name = str(winner_name) if pd.notna(winner_name) else "?"
            team_val = winner.get("TeamName")
            team_str = str(team_val) if pd.notna(team_val) else None

            gap_str = ""
            if len(res) >= 2:
                second = res.iloc[1]
                second_name = second.get("Abbreviation")
                second_name = str(second_name) if pd.notna(second_name) else "?"
                gap_time = second.get("Time")
                if pd.notna(gap_time):
                    try:
                        gap_seconds = gap_time.total_seconds()
                        gap_str = f" by {gap_seconds:.3f}s over {second_name}"
                    except Exception:
                        pass

            team_suffix = f" ({team_str})" if team_str else ""
            sentences.append(f"<b>{winner_name}</b>{team_suffix} won{gap_str}.")
    except Exception:
        pass

    # 2. Race events (SC / VSC / Red flag)
    events = _race_events(session)
    if events:
        counts: dict[str, list[tuple[int, int]]] = {}
        for ev_type, s, e in events:
            counts.setdefault(ev_type, []).append((s, e))

        parts = []
        for ev_type, periods in counts.items():
            label = {"SC": "Safety Car", "VSC": "VSC", "RED": "red flag"}.get(ev_type, ev_type)
            n = len(periods)
            plural = "s" if n > 1 else ""
            # Format laps: "L18-22, L42-44"
            lap_strs = ", ".join(f"L{s}-{e}" for s, e in periods[:2])
            if len(periods) > 2:
                lap_strs += "..."
            parts.append(f"{n} {label}{plural} ({lap_strs})")

        if parts:
            sentences.append(f"Race interrupted by {', '.join(parts)}.")

    # 3. DNFs
    dnfs = _dnf_list(session)
    if dnfs:
        n = len(dnfs)
        plural = "s" if n > 1 else ""
        # Top DNF causes
        cause_counts: dict[str, int] = {}
        for d in dnfs:
            cause_counts[d["Status"]] = cause_counts.get(d["Status"], 0) + 1
        top_causes = sorted(cause_counts.items(), key=lambda kv: -kv[1])[:3]
        causes_str = ", ".join(f"{c} ({n})" if n > 1 else c for c, n in top_causes)
        sentences.append(f"{n} retirement{plural}: {causes_str}.")

    # 4. Fastest lap
    flap = _fastest_lap_info(session)
    if flap is not None:
        fl_t = _fmt_laptime(flap["lap_time"])
        if flap["eligible"]:
            elig_str = "earning the <b>+1 pt</b> bonus"
        elif flap.get("position"):
            elig_str = f"(finished P{flap['position']}, no bonus point)"
        else:
            elig_str = ""
        base = (
            f"Fastest lap: <b>{flap['driver']}</b> {fl_t} "
            f"on lap {flap['lap_number']}"
        )
        sentences.append(f"{base} {elig_str}.".strip() if elig_str else f"{base}.")

    if not sentences:
        return None
    return " ".join(sentences)


def _teammate_h2h(session) -> list[dict]:
    """
    Per-team H2H comparison for this single race.
    Returns list of {team, drivers, race_winner, grid_winner, points, ...}.
    """
    try:
        res = session.results
        if res is None or len(res) == 0:
            return []
        df = res.dropna(subset=["Position"]).copy()
    except Exception:
        return []
    if len(df) == 0 or "TeamName" not in df.columns:
        return []

    pairs = []
    for team, grp in df.groupby("TeamName"):
        if len(grp) < 2:
            continue
        # Sort by Position (best first)
        gs = grp.sort_values("Position").reset_index(drop=True)
        d1 = gs.iloc[0]
        d2 = gs.iloc[1]

        def _abbr(row):
            v = row.get("Abbreviation")
            return str(v) if pd.notna(v) else "?"

        def _grid(row):
            g = row.get("GridPosition")
            if pd.isna(g):
                return None
            return "PL" if g == 0 else f"P{int(g)}"

        def _pts(row):
            p = row.get("Points")
            try:
                return float(p) if pd.notna(p) else 0.0
            except Exception:
                return 0.0

        # Grid winner: lower grid number = better
        d1_grid_n = d1.get("GridPosition")
        d2_grid_n = d2.get("GridPosition")
        if pd.notna(d1_grid_n) and pd.notna(d2_grid_n):
            d1_g = float(d1_grid_n) if d1_grid_n != 0 else 21
            d2_g = float(d2_grid_n) if d2_grid_n != 0 else 21
            grid_winner_abbr = _abbr(d1) if d1_g <= d2_g else _abbr(d2)
        else:
            grid_winner_abbr = None

        pairs.append({
            "team":         str(team),
            "d1":           _abbr(d1),
            "d1_finish":    f"P{int(d1['Position'])}",
            "d1_grid":      _grid(d1) or "—",
            "d1_pts":       _pts(d1),
            "d2":           _abbr(d2),
            "d2_finish":    f"P{int(d2['Position'])}",
            "d2_grid":      _grid(d2) or "—",
            "d2_pts":       _pts(d2),
            "race_winner":  _abbr(d1),   # d1 already sorted as race winner
            "grid_winner":  grid_winner_abbr,
        })

    # Sort teams by best finish position (constructor standings di GP itu)
    pairs.sort(key=lambda p: int(p["d1_finish"][1:]))
    return pairs


def _fastest_lap_info(session) -> dict | None:
    """Fastest lap holder + eligibility for +1 point (P10 or higher)."""
    try:
        fl = session.laps.pick_fastest()
        if fl is None or pd.isna(fl.get("LapTime")):
            return None
    except Exception:
        return None

    driver = str(fl.get("Driver", "?"))
    pos = None
    eligible = False
    try:
        res = session.results.dropna(subset=["Position"])
        match = res[res["Abbreviation"] == driver]
        if len(match) > 0:
            pos = int(match.iloc[0]["Position"])
            eligible = pos <= 10
    except Exception:
        pass

    # Guard NaN untuk Team
    team_val = fl.get("Team") if "Team" in fl.index else None
    team_str = str(team_val) if pd.notna(team_val) else "—"

    return {
        "driver":     driver,
        "team":       team_str,
        "lap_time":   fl["LapTime"].total_seconds(),
        "lap_number": int(fl["LapNumber"]) if pd.notna(fl.get("LapNumber")) else None,
        "compound":   str(fl.get("Compound", "")) if pd.notna(fl.get("Compound")) else None,
        "position":   pos,
        "eligible":   eligible,
    }


def render():
    st.title("Race Overview")

    _current_year = st.session_state.get("selected_year", 2024)
    st.markdown(
        f"<p style='color:#444;margin-top:-16px;font-size:13px;letter-spacing:0.05em;text-transform:uppercase'>Lap Data · Positions · Gaps · {_current_year} Season</p>",
        unsafe_allow_html=True,
    )

    # ── Session selectors ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        selected_year = st.selectbox(
            "Season",
            options=sorted(SUPPORTED_YEARS, reverse=True),
            index=0,
            key="selected_year",  # shared dengan telemetry & sidebar
        )

    with col2:
        rounds = get_available_rounds(selected_year)
        gp_label = st.selectbox(
            "Grand Prix",
            options=[f"R{k} · {v}" for k, v in rounds.items()],
            index=0,
            key="ro_gp",
        )
        gp_round = int(gp_label.split("·")[0].strip()[1:])
        gp_name  = rounds[gp_round]

    with col3:
        session_type = st.selectbox(
            "Session",
            options=["R", "Q"],
            format_func=lambda x: SESSION_LABELS[x],
            key="ro_session",
        )

    st.divider()
    load_btn = st.button("Load Session", type="primary", key="ro_load")

    session_id  = f"{selected_year}_{gp_name}_{session_type}"
    session_key = f"ro_session_{session_id}"

    if load_btn or session_key in st.session_state:
        with st.spinner(f"Loading {gp_name} {SESSION_LABELS[session_type]} {selected_year}..."):
            session = load_session(selected_year, gp_name, session_type)

        if session is None:
            st.error("Session data unavailable. Try a different round.")
            return

        st.session_state[session_key] = True

        # ── Sticky session header (#22) ──────────────────────────────────────
        # Position: sticky di-render Streamlit di main content area, akan
        # menempel di top saat scroll. Backdrop-filter blur biar background
        # tidak overlap dengan section di bawah.
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

        # ── Get clean laps ────────────────────────────────────────────────────
        with st.spinner("Processing lap data..."):
            laps = get_laps(session, session_id)

        if len(laps) == 0:
            st.error("No lap data found for this session.")
            return

        # ── Race summary cards (#5) ──────────────────────────────────────────
        if session_type == "R":
            stats = _race_summary_stats(session)
            scol1, scol2, scol3, scol4 = st.columns(4)

            with scol1:
                st.metric(
                    "Total laps",
                    str(stats.get("total_laps", "—")),
                )
            with scol2:
                st.metric(
                    "Race duration",
                    _fmt_duration_hms(stats.get("race_duration")),
                )
            with scol3:
                fl_d = stats.get("fastest_driver", "—")
                fl_t = _fmt_laptime(stats.get("fastest_time"))
                fl_n = stats.get("fastest_lap_num")
                st.metric(
                    "Fastest lap",
                    f"{fl_t}",
                    delta=f"{fl_d}" + (f" · L{fl_n}" if fl_n else ""),
                    delta_color="off",
                )
            with scol4:
                ps_count = stats.get("total_pit_stops")
                st.metric(
                    "Pit stops",
                    str(ps_count) if ps_count is not None else "—",
                )

            # Weather strip — terpisah dari cards supaya tidak confusing
            weather_parts = []
            if stats.get("avg_track_temp") is not None:
                weather_parts.append(f"Track {stats['avg_track_temp']:.0f}°C avg")
            if stats.get("avg_air_temp") is not None:
                weather_parts.append(f"Air {stats['avg_air_temp']:.0f}°C avg")
            if stats.get("any_rain"):
                weather_parts.append(
                    "<span style='color:#3DA9FF;font-weight:700'>RAIN detected</span>"
                )
            if weather_parts:
                st.markdown(
                    f"<div style='color:#888;font-size:11px;margin:-8px 0 12px;"
                    f"font-family:Barlow,sans-serif;letter-spacing:0.02em'>"
                    f"<span style='color:#666;text-transform:uppercase;"
                    f"font-family:Barlow Condensed,sans-serif;font-weight:700;"
                    f"font-size:10px;letter-spacing:0.15em'>Weather</span> · "
                    f"{' · '.join(weather_parts)}</div>",
                    unsafe_allow_html=True,
                )

        # ── Race summary narrative (#21) — auto-generated 2-4 kalimat ───────
        if session_type == "R":
            narrative = _race_summary_narrative(session)
            if narrative:
                st.markdown(
                    f"""
                    <div style='
                        padding: 12px 16px;
                        margin: 4px 0 16px;
                        background: linear-gradient(90deg, rgba(232,0,45,0.06) 0%, rgba(255,255,255,0.01) 100%);
                        border-left: 3px solid #E8002D;
                        border-radius: 0 4px 4px 0;
                        font-family: Barlow, sans-serif;
                        font-size: 13.5px;
                        line-height: 1.55;
                        color: #DDD;
                    '>
                        <div style='color:#888;font-size:10px;font-weight:700;
                                   letter-spacing:0.15em;text-transform:uppercase;
                                   font-family:Barlow Condensed,sans-serif;
                                   margin-bottom:6px'>
                            Race Summary
                        </div>
                        {narrative}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Driver of the Day + Fastest Lap banners (#4, #19) ─────────────────
        if session_type == "R":
            dod = _driver_of_the_day(session)
            flap = _fastest_lap_info(session)

            # Render in two columns kalau dua-duanya ada
            if dod is not None and flap is not None:
                bcol1, bcol2 = st.columns(2)
            else:
                bcol1 = st.container()
                bcol2 = None

            if dod is not None:
                team_color = DRIVER_COLORS.get(dod["driver"], "#E8002D")
                with bcol1:
                    st.markdown(
                        f"""
                        <div style='border-left:3px solid {team_color};
                                   padding:10px 16px;margin:8px 0 16px;
                                   background:rgba(255,255,255,0.02);
                                   border-radius:0 4px 4px 0'>
                            <div style='color:#888;font-size:10px;font-weight:700;
                                       letter-spacing:0.15em;text-transform:uppercase;
                                       font-family:Barlow Condensed,sans-serif'>
                                Driver of the Day
                            </div>
                            <div style='display:flex;align-items:baseline;gap:12px;
                                        margin-top:4px;flex-wrap:wrap'>
                                <span style='color:{team_color};font-size:22px;
                                            font-weight:700;letter-spacing:0.05em'>{dod["driver"]}</span>
                                <span style='color:#AAA;font-size:13px'>
                                    {dod["grid"]} → {dod["finish"]}
                                    <span style='color:#39B54A;font-weight:700;margin-left:6px'>
                                        +{dod["delta"]} positions
                                    </span>
                                </span>
                                <span style='color:#666;font-size:12px'>{dod["team"]}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if flap is not None:
                fl_color = DRIVER_COLORS.get(flap["driver"], "#B026FF")
                fl_t_str = _fmt_laptime(flap["lap_time"])
                pos_str = f"P{flap['position']}" if flap["position"] else "—"
                # Purple = F1 broadcast convention untuk fastest lap
                point_str = (
                    "<span style='color:#B026FF;font-weight:700'>+1 pt</span>"
                    if flap["eligible"]
                    else f"<span style='color:#666'>{pos_str}, no point</span>"
                )
                container = bcol2 if bcol2 is not None else bcol1
                with container:
                    st.markdown(
                        f"""
                        <div style='border-left:3px solid #B026FF;
                                   padding:10px 16px;margin:8px 0 16px;
                                   background:rgba(176,38,255,0.06);
                                   border-radius:0 4px 4px 0'>
                            <div style='color:#888;font-size:10px;font-weight:700;
                                       letter-spacing:0.15em;text-transform:uppercase;
                                       font-family:Barlow Condensed,sans-serif'>
                                Fastest Lap of the Race
                            </div>
                            <div style='display:flex;align-items:baseline;gap:12px;
                                        margin-top:4px;flex-wrap:wrap'>
                                <span style='color:{fl_color};font-size:22px;
                                            font-weight:700;letter-spacing:0.05em'>{flap["driver"]}</span>
                                <span style='color:#FFF;font-size:18px;
                                            font-family:monospace;font-weight:600'>{fl_t_str}</span>
                                <span style='color:#888;font-size:12px'>L{flap['lap_number']}</span>
                                <span style='font-size:12px'>{point_str}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Final classification (with Grid → Finish delta, #1) ──────────────
        st.subheader(
            "Final classification",
            help=(
                "Hasil resmi sesi. Grid = posisi start, Δ = perubahan posisi "
                "ke finish (+ = naik posisi, − = turun). 'PL' = pit lane start."
            ),
        )
        results_df = build_session_results(session)
        if len(results_df) > 0:
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            st.info("Result data not available for this session.")

        # ── DNF list (#17) + Penalty list (#18) — side by side ───────────────
        if session_type == "R":
            dnf_rows = _dnf_list(session)
            penalty_rows = _penalty_list(session)

            if dnf_rows or penalty_rows:
                ecol1, ecol2 = st.columns(2)

                with ecol1:
                    if dnf_rows:
                        st.subheader(
                            "DNF list",
                            help=(
                                "Drivers yang tidak menyelesaikan race "
                                "(retired / accident / mechanical / DSQ). "
                                "Lapped finishers tidak masuk sini."
                            ),
                        )
                        st.dataframe(
                            pd.DataFrame(dnf_rows),
                            use_container_width=True, hide_index=True,
                        )

                with ecol2:
                    if penalty_rows:
                        st.subheader(
                            "Penalties",
                            help=(
                                "Penalty yang dikenakan saat race "
                                "(parsed dari race control messages). "
                                "Pesan 'under investigation' / 'no further action' "
                                "tidak ditampilkan."
                            ),
                        )
                        st.dataframe(
                            pd.DataFrame(penalty_rows),
                            use_container_width=True, hide_index=True,
                        )

        # ── Teammate H2H (#23) ───────────────────────────────────────────────
        if session_type == "R":
            h2h_pairs = _teammate_h2h(session)
            if h2h_pairs:
                st.subheader(
                    "Teammate head-to-head",
                    help=(
                        "Per-team comparison untuk race ini: race finish, "
                        "grid position, dan points. Driver tercepat di tim "
                        "dilingkari di kiri. Berguna untuk lihat siapa "
                        "outperform teammate di GP itu."
                    ),
                )

                # Render 2 cards per row
                n_pairs = len(h2h_pairs)
                rows = (n_pairs + 1) // 2

                for r in range(rows):
                    cols = st.columns(2)
                    for c in range(2):
                        idx = r * 2 + c
                        if idx >= n_pairs:
                            continue
                        p = h2h_pairs[idx]
                        d1, d2 = p["d1"], p["d2"]
                        c1 = DRIVER_COLORS.get(d1, "#888")
                        c2 = DRIVER_COLORS.get(d2, "#888")

                        # Highlight winner per category
                        race_w = p["race_winner"]
                        grid_w = p["grid_winner"]
                        d1_race = "<b style='color:#FFD700'>●</b>" if race_w == d1 else "<span style='color:#444'>○</span>"
                        d2_race = "<b style='color:#FFD700'>●</b>" if race_w == d2 else "<span style='color:#444'>○</span>"
                        d1_grid = "<b style='color:#FFD700'>●</b>" if grid_w == d1 else "<span style='color:#444'>○</span>"
                        d2_grid = "<b style='color:#FFD700'>●</b>" if grid_w == d2 else "<span style='color:#444'>○</span>"

                        with cols[c]:
                            st.markdown(
                                f"""
                                <div style='border:1px solid #1F1F1F;border-radius:4px;
                                           padding:10px 14px;margin-bottom:8px;
                                           background:rgba(255,255,255,0.012)'>
                                    <div style='color:#666;font-size:10px;font-weight:700;
                                               letter-spacing:0.12em;text-transform:uppercase;
                                               font-family:Barlow Condensed,sans-serif;
                                               margin-bottom:8px'>
                                        {p["team"]}
                                    </div>
                                    <table style='width:100%;font-family:Barlow,sans-serif;font-size:13px;color:#CCC'>
                                        <tr>
                                            <td style='padding:2px 0'>
                                                <b style='color:{c1};font-size:15px'>{d1}</b>
                                            </td>
                                            <td style='padding:2px 0;text-align:center;color:#666'>vs</td>
                                            <td style='padding:2px 0;text-align:right'>
                                                <b style='color:{c2};font-size:15px'>{d2}</b>
                                            </td>
                                        </tr>
                                        <tr>
                                            <td style='padding:1px 0'>{d1_race} <span style='color:#888'>{p["d1_finish"]}</span></td>
                                            <td style='padding:1px 0;text-align:center;color:#555;font-size:10px;text-transform:uppercase;letter-spacing:0.1em'>Race</td>
                                            <td style='padding:1px 0;text-align:right'><span style='color:#888'>{p["d2_finish"]}</span> {d2_race}</td>
                                        </tr>
                                        <tr>
                                            <td style='padding:1px 0'>{d1_grid} <span style='color:#888'>{p["d1_grid"]}</span></td>
                                            <td style='padding:1px 0;text-align:center;color:#555;font-size:10px;text-transform:uppercase;letter-spacing:0.1em'>Grid</td>
                                            <td style='padding:1px 0;text-align:right'><span style='color:#888'>{p["d2_grid"]}</span> {d2_grid}</td>
                                        </tr>
                                        <tr>
                                            <td style='padding:1px 0;color:#FFF;font-weight:600'>{p["d1_pts"]:.0f}</td>
                                            <td style='padding:1px 0;text-align:center;color:#555;font-size:10px;text-transform:uppercase;letter-spacing:0.1em'>Points</td>
                                            <td style='padding:1px 0;text-align:right;color:#FFF;font-weight:600'>{p["d2_pts"]:.0f}</td>
                                        </tr>
                                    </table>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # ── Lap leader summary (#3) ──────────────────────────────────────────
        if session_type == "R":
            lead_info = _lap_leader_summary(session)
            if lead_info["total_laps_with_p1"] > 0:
                top_leader = max(lead_info["laps_led"].items(), key=lambda kv: kv[1])
                top_d, top_n = top_leader
                top_pct = (top_n / lead_info["total_laps_with_p1"]) * 100 if lead_info["total_laps_with_p1"] else 0
                top_color = DRIVER_COLORS.get(top_d, "#E8002D")
                # Format laps led list (top 3) — tiap driver pakai warna timnya
                sorted_leaders = sorted(
                    lead_info["laps_led"].items(), key=lambda kv: -kv[1]
                )[:3]
                leaders_str = " · ".join(
                    f"<b style='color:{DRIVER_COLORS.get(d, '#CCC')}'>{d}</b> {n}"
                    for d, n in sorted_leaders
                )
                st.markdown(
                    f"""
                    <div style='padding:8px 14px;margin:8px 0 12px;
                               background:rgba(255,255,255,0.015);
                               border-left:2px solid {top_color};
                               font-family:Barlow,sans-serif;font-size:13px;color:#BBB'>
                        <b style='color:{top_color}'>{top_d}</b> led
                        <b style='color:#FFF'>{top_n}</b> of
                        <b>{lead_info['total_laps_with_p1']}</b> laps
                        ({top_pct:.0f}%) ·
                        <span style='color:#888'>{lead_info['lead_changes']} lead changes</span>
                        <div style='font-size:11px;color:#666;margin-top:3px'>
                            Top: {leaders_str}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Driver filter ─────────────────────────────────────────────────────
        all_drivers = sorted(laps["Driver"].unique().tolist())

        st.subheader(
            "Driver filter",
            help=(
                "Filter driver untuk tabel & chart di bawah (race pace, top speed, "
                "sector pace, pit stops, position chart, heatmap, gap to leader). "
                "Section di atas filter (race summary, classification, DNF, "
                "penalty, lap leader) selalu tampil all drivers."
            ),
        )
        st.caption("Default: all drivers. Deselect to focus on specific drivers.")

        selected_drivers = st.multiselect(
            "Drivers",
            options=all_drivers,
            default=all_drivers,
            label_visibility="collapsed",
        )

        if not selected_drivers:
            st.warning("Select at least one driver.")
            return

        laps_filtered = laps[laps["Driver"].isin(selected_drivers)]

        st.divider()

        # ── Race pace ranking (#6) ───────────────────────────────────────────
        st.subheader(
            "Race pace ranking",
            help=(
                "Average + median lap time per driver dari clean laps "
                "(pit & outliers sudah difilter). Disort by median (tercepat duluan)."
            ),
        )
        pace_df = build_race_pace_ranking(laps_filtered, selected_drivers)
        if len(pace_df) > 0:
            st.dataframe(pace_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough clean laps for pace ranking.")

        st.divider()

        # ── Fastest laps table (existing) ─────────────────────────────────────
        st.subheader(
            "Fastest laps",
            help="Lap tercepat per driver dari clean laps di session ini.",
        )
        fastest_df = build_fastest_laps_table(laps_filtered)
        if len(fastest_df) == 0:
            st.info("No valid lap times for selected drivers.")
        else:
            st.dataframe(fastest_df, use_container_width=True, hide_index=False)

        # ── Top speed leaderboard (#8) ───────────────────────────────────────
        topspd_df = build_top_speed_table(session, selected_drivers)
        if len(topspd_df) > 0:
            st.subheader(
                "Top speed leaderboard",
                help=(
                    "Top speed per driver di speed trap (kolom SpeedST = "
                    "end-of-main-straight). Sorted descending."
                ),
            )
            st.dataframe(topspd_df, use_container_width=True, hide_index=True)

        # ── Sector pace ranking (#9) ─────────────────────────────────────────
        sector_df = build_sector_pace_table(session, selected_drivers)
        if len(sector_df) > 0:
            st.subheader(
                "Sector pace ranking",
                help=(
                    "Best sector time per driver dari semua lap. Total = "
                    "jumlah best sectors (theoretical best lap). Fastest "
                    "tiap sector di-highlight ungu."
                ),
            )

            def _fmt_sec_cell(v):
                return _fmt_sector(v)

            def _highlight_best_sector(col):
                """Highlight smallest value di kolom S1/S2/S3/Total."""
                if col.name in ("S1", "S2", "S3", "Total"):
                    valid = col.dropna()
                    if len(valid) == 0:
                        return [""] * len(col)
                    fastest = valid.min()
                    return [
                        "background-color: rgba(176,38,255,0.32); font-weight: 700; color: #FFF;"
                        if pd.notna(v) and v == fastest else ""
                        for v in col
                    ]
                return [""] * len(col)

            styled_sec = (
                sector_df.style
                .apply(_highlight_best_sector, axis=0)
                .format({"S1": _fmt_sec_cell, "S2": _fmt_sec_cell,
                         "S3": _fmt_sec_cell, "Total": _fmt_sec_cell})
            )
            st.dataframe(styled_sec, use_container_width=True, hide_index=True)

        # ── Pit stop summary (#10) ───────────────────────────────────────────
        if session_type == "R":
            pit_df = build_pit_stops_summary(session, selected_drivers)
            if len(pit_df) > 0:
                st.subheader(
                    "Pit stop summary",
                    help=(
                        "Tiap row = 1 pit stop. 'Tyre Off' = compound saat "
                        "masuk pit, 'Tyre On' = compound setelah keluar pit. "
                        "Age = umur ban (laps) saat dilepas. "
                        "Duration = waktu total di pit lane."
                    ),
                )
                st.dataframe(pit_df, use_container_width=True, hide_index=True)

        # ── Pit stop leaderboards (#11, #12) ─────────────────────────────────
        if session_type == "R":
            pit_durations = _pit_stop_durations(session, selected_drivers)
            if pit_durations:
                pcol1, pcol2 = st.columns(2)

                # #11 Fastest pit stops
                with pcol1:
                    st.subheader(
                        "Fastest pit stops",
                        help=(
                            "Top 5 pit stops dengan durasi terkecil (total time "
                            "di pit lane). Pit stops > 2 menit difilter (anomali "
                            "atau driver retired)."
                        ),
                    )
                    sorted_p = sorted(pit_durations, key=lambda x: x["DurationSeconds"])
                    top5 = sorted_p[:5]
                    fastest_df = pd.DataFrame([{
                        "Rank":     i + 1,
                        "Driver":   r["Driver"],
                        "Team":     r["Team"],
                        "Lap":      r["Lap"],
                        "Duration": f"{r['DurationSeconds']:.2f}s",
                    } for i, r in enumerate(top5)])
                    st.dataframe(fastest_df, use_container_width=True, hide_index=True)

                # #12 Team average pit time
                with pcol2:
                    st.subheader(
                        "Team avg pit time",
                        help=(
                            "Rata-rata durasi pit stop per tim (semua stops). "
                            "Indicator kualitas pit crew."
                        ),
                    )
                    teams: dict[str, list[float]] = {}
                    for r in pit_durations:
                        teams.setdefault(r["Team"], []).append(r["DurationSeconds"])
                    team_rows = []
                    for team, durs in teams.items():
                        team_rows.append({
                            "Team":     team,
                            "Avg":      sum(durs) / len(durs),
                            "Stops":    len(durs),
                            "Fastest":  min(durs),
                        })
                    if team_rows:
                        team_df = pd.DataFrame(team_rows).sort_values("Avg").reset_index(drop=True)
                        team_df.insert(0, "Rank", range(1, len(team_df) + 1))
                        team_df["Avg"]     = team_df["Avg"].apply(lambda x: f"{x:.2f}s")
                        team_df["Fastest"] = team_df["Fastest"].apply(lambda x: f"{x:.2f}s")
                        st.dataframe(team_df, use_container_width=True, hide_index=True)

        # ── Position changes / Overtake counter (#13) ────────────────────────
        if session_type == "R":
            pos_changes_df = _position_changes(session)
            if len(pos_changes_df) > 0:
                # Filter ke selected_drivers
                pos_changes_df = pos_changes_df[
                    pos_changes_df["Driver"].isin(selected_drivers)
                ].reset_index(drop=True)
                if len(pos_changes_df) > 0:
                    st.subheader(
                        "Position changes (on-track)",
                        help=(
                            "Net position changes per driver — gained = jumlah "
                            "posisi yang naik on-track, lost = jumlah turun. "
                            "Net = gained - lost. Pit-related changes "
                            "(masuk/keluar pit) di-exclude."
                        ),
                    )
                    st.dataframe(
                        pos_changes_df, use_container_width=True, hide_index=True,
                    )

        # ── Battles / DRS-range pairs (#14) ──────────────────────────────────
        if session_type == "R":
            battles = _battles(session, drs_range=1.5, min_duration=3)
            if battles:
                # Filter battles yang melibatkan selected_drivers
                battles_f = [
                    b for b in battles
                    if b["Driver A"] in selected_drivers and b["Driver B"] in selected_drivers
                ]
                # Top 10 longest battles
                top_battles = battles_f[:10]
                if top_battles:
                    st.subheader(
                        "Closest battles",
                        help=(
                            "Pair of drivers yang berada dalam <1.5s "
                            "(approx DRS range) selama minimal 3 lap berturut-turut. "
                            "Diurutkan by durasi battle (lama). Cuma top 10 "
                            "yang ditampilkan."
                        ),
                    )
                    st.dataframe(
                        pd.DataFrame(top_battles),
                        use_container_width=True, hide_index=True,
                    )

        st.divider()

        # ── Position chart (Race only, with race events + rain overlay #2,#16) ─
        if session_type == "R":
            st.subheader(
                "Lap-by-lap positions",
                help=(
                    "Trajectory posisi per driver tiap lap. Shaded yellow = "
                    "Safety Car / VSC, shaded red = red flag, shaded biru = "
                    "rain detected. Hover untuk klasemen lengkap di lap itu "
                    "(sorted by posisi). Toggle 'Show previous year' untuk "
                    "overlay GP yang sama tahun sebelumnya."
                ),
            )

            # #24 Year-on-year overlay toggle
            prev_year = selected_year - 1
            prev_available = prev_year in SUPPORTED_YEARS
            prev_laps_df = None
            show_prev = False
            if prev_available:
                # Cek apakah GP yang sama ada di tahun sebelumnya
                prev_rounds = get_available_rounds(prev_year)
                gp_in_prev = gp_name in prev_rounds.values()
                if gp_in_prev:
                    show_prev = st.checkbox(
                        f"Show {prev_year} {gp_name} (ghost overlay)",
                        value=False, key="ro_yoy",
                        help=(
                            f"Overlay position chart dari race {prev_year} di "
                            f"sirkuit yang sama. Driver yang sama di kedua tahun "
                            f"di-render sebagai dashed line dengan warna sama. "
                            f"Berguna buat compare race outcome year-on-year."
                        ),
                    )
                    if show_prev:
                        with st.spinner(f"Loading {prev_year} {gp_name}..."):
                            prev_session = load_session(prev_year, gp_name, session_type)
                        if prev_session is not None:
                            try:
                                prev_laps_df = prev_session.laps[
                                    ["Driver", "LapNumber", "Position"]
                                ].dropna(subset=["Position"]).copy()
                                prev_laps_df["Position"] = prev_laps_df["Position"].astype(int)
                            except Exception:
                                prev_laps_df = None

            pos_laps = session.laps[
                ["Driver", "LapNumber", "Position"]
            ].dropna(subset=["Position"]).copy()
            pos_laps["Position"] = pos_laps["Position"].astype(int)
            pos_laps_filtered = pos_laps[pos_laps["Driver"].isin(selected_drivers)]

            # Combine SC/VSC/Red events + rain laps overlay
            race_events = _race_events(session)
            rain_periods = _rain_laps(session)
            race_events += [("RAIN", s, e) for s, e in rain_periods]

            if len(pos_laps_filtered) > 0:
                fig_pos = build_position_chart(
                    pos_laps_filtered, selected_drivers,
                    race_events=race_events,
                    prev_year_laps=prev_laps_df,
                    prev_year_label=str(prev_year) if prev_laps_df is not None else "",
                )
                st.plotly_chart(
                    fig_pos, use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info("Position data not available for this session.")

            st.divider()

        # ── Weather timeline (#15) ───────────────────────────────────────────
        fig_weather = build_weather_timeline(session)
        if fig_weather is not None:
            st.subheader(
                "Weather timeline",
                help=(
                    "Track temp, air temp, wind speed, dan rainfall over "
                    "session time. Rainfall sebagai shaded area di subplot bawah."
                ),
            )
            st.plotly_chart(
                fig_weather, use_container_width=True,
                config={"displayModeBar": False},
            )

            st.divider()

        # ── Race pace heatmap (#7) ───────────────────────────────────────────
        st.subheader(
            "Race pace heatmap",
            help=(
                "Driver × lap, warna = deviation lap time vs median lap itu. "
                "Biru = lebih cepat dari median, merah = lebih lambat. "
                "Bagus buat lihat siapa konsisten + identifikasi anomali "
                "(SC, undercut, slow lap)."
            ),
        )
        fig_heatmap = build_pace_heatmap(laps, selected_drivers)
        if fig_heatmap is not None:
            st.plotly_chart(
                fig_heatmap, use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Not enough lap data for pace heatmap.")

        st.divider()

        # ── Gap to leader ─────────────────────────────────────────────────────
        st.subheader(
            "Gap to leader",
            help=(
                "Selisih waktu kumulatif tiap driver vs leader race di setiap "
                "lap. Pakai session.laps['Time'] (timestamp end-of-lap) langsung, "
                "jadi pit stop & SC otomatis ikut terhitung. Hover lap untuk "
                "ranking lengkap sorted leader → backmarker."
            ),
        )
        if session_type == "Q":
            st.caption("In qualifying, gap is calculated from cumulative lap time.")

        # Pakai session.laps mentah (TIDAK difilter pit/outlier) — kolom `Time`
        # adalah timestamp riil end-of-lap dari awal session, sehingga pit stop
        # & lap kuning otomatis ikut terhitung di gap kumulatif.
        gap_laps = session.laps[["Driver", "LapNumber", "Time"]].copy()
        gap_laps["ElapsedSeconds"] = gap_laps["Time"].dt.total_seconds()
        gap_laps = gap_laps[gap_laps["Driver"].isin(selected_drivers)]

        fig_gap = build_gap_to_leader(gap_laps, selected_drivers)
        st.plotly_chart(fig_gap, use_container_width=True, config={"displayModeBar": False})

    else:
        st.markdown("""
        <div style='text-align:center;padding:80px 0;color:#333'>
            <div style='font-size:48px'>🏁</div>
            <div style='font-size:18px;margin-top:16px;color:#444'>Select a Grand Prix and session above</div>
            <div style='font-size:13px;margin-top:8px;color:#2a2a2a'>Then hit Load Session</div>
        </div>
        """, unsafe_allow_html=True)
