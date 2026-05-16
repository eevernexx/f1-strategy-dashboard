"""
F1 Data Pipeline
================
Handles all FastF1 data fetching with:
- Persistent disk cache (avoid re-downloading)
- Streamlit cache layer (@st.cache_data) for in-session speed
- Graceful error handling — never crash the UI
"""

import os
import fastf1
import pandas as pd
import streamlit as st

from src.utils.config import CACHE_DIR, F1_ROUNDS, YEAR


def _setup_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)


_setup_cache()


# ── Session loader ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_session(year: int, gp: str, session_type: str) -> fastf1.core.Session | None:
    """
    Load a FastF1 session. Returns None on failure (never raises).
    Cached for 1 hour — fast enough for a dashboard.
    """
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=True, laps=True, weather=True)
        return session
    except Exception as e:
        st.error(f"Failed to load session: {e}")
        return None


# ── Lap data ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_laps(_session: fastf1.core.Session, session_id: str) -> pd.DataFrame:
    """
    Extract clean lap DataFrame from session.
    Adds human-readable lap time (seconds) column.

    `session_id` is a hashable key (e.g. f"{year}_{gp}_{session_type}") so
    Streamlit's cache can distinguish between sessions — `_session` is
    skipped from the cache key by Streamlit.
    """
    laps = _session.laps.copy()

    # Convert LapTime timedelta → float seconds for plotting
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    # Filter out obvious outliers: pit laps, safety car laps, and
    # statistical outliers (> 2 std dev above median per driver)
    valid = laps[
        laps["LapTimeSeconds"].notna()
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
    ].copy()

    # Remove statistical outliers per driver
    def remove_outliers(group):
        median = group["LapTimeSeconds"].median()
        std    = group["LapTimeSeconds"].std()
        if pd.isna(std):
            # Single-lap driver (qualifying) — std is NaN, keep as-is
            return group
        return group[group["LapTimeSeconds"] < median + 2 * std]

    valid = valid.groupby("Driver", group_keys=False).apply(remove_outliers)
    return valid.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_driver_fastest_lap(_session: fastf1.core.Session, driver: str, session_id: str):
    """Return the single fastest lap object for a driver."""
    try:
        laps = _session.laps.pick_drivers(driver)
        return laps.pick_fastest()
    except Exception:
        return None


# ── Telemetry ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_telemetry_for_lap(_lap, lap_id: str) -> pd.DataFrame | None:
    """
    Get car telemetry for a specific lap.
    Adds Distance column normalized to 0..1 for multi-driver comparison.
    Returns None if telemetry unavailable.
    """
    try:
        tel = _lap.get_car_data().add_distance()

        # FastF1 menamai kolom gear sebagai "nGear" — alias jadi "Gear"
        # supaya seluruh kode aplikasi konsisten pakai nama "Gear".
        if "nGear" in tel.columns and "Gear" not in tel.columns:
            tel = tel.rename(columns={"nGear": "Gear"})

        # Normalize distance to [0, 1] — makes overlay comparison valid
        max_dist = tel["Distance"].max()
        if max_dist > 0:
            tel["DistanceNorm"] = tel["Distance"] / max_dist
        else:
            tel["DistanceNorm"] = tel["Distance"]

        return tel
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_pos_data_for_lap(_lap, lap_id: str) -> pd.DataFrame | None:
    """Get positional (X/Y) data for track map rendering."""
    try:
        return _lap.get_pos_data()
    except Exception:
        return None


# ── Driver roster ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_session_drivers(_session: fastf1.core.Session, session_id: str) -> list[str]:
    """Return sorted list of driver abbreviations present in the session."""
    try:
        return sorted(_session.laps["Driver"].unique().tolist())
    except Exception:
        return []


# ── Race-level helpers ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_race_position_history(_session: fastf1.core.Session, session_id: str) -> pd.DataFrame:
    """
    Build lap-by-lap position DataFrame for all drivers.
    Columns: Driver, LapNumber, Position
    """
    laps = _session.laps[["Driver", "LapNumber", "Position"]].copy()
    laps = laps.dropna(subset=["Position"])
    laps["Position"] = laps["Position"].astype(int)
    return laps


@st.cache_data(ttl=3600, show_spinner=False)
def get_tyre_strategy(_session: fastf1.core.Session, session_id: str) -> pd.DataFrame:
    """
    Build tyre stint summary: driver, stint, compound, lap_start, lap_end.
    """
    laps = _session.laps[["Driver", "LapNumber", "Compound", "Stint"]].copy()
    laps = laps.dropna(subset=["Compound"])

    stints = (
        laps.groupby(["Driver", "Stint", "Compound"])
        .agg(LapStart=("LapNumber", "min"), LapEnd=("LapNumber", "max"))
        .reset_index()
    )
    return stints


# ── Round helpers ─────────────────────────────────────────────────────────────

def get_available_rounds(year: int = YEAR) -> dict[int, str]:
    """Return calendar for a given year."""
    return F1_ROUNDS.get(year, F1_ROUNDS[2024])
