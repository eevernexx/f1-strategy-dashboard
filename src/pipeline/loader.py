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


def _resolve_cache_dir() -> str:
    """
    Pick writable cache dir:
    - Streamlit Cloud / cloud env → /tmp/fastf1_cache (always writable, ephemeral)
    - Local dev → data/cache (relative to project, persistent)

    Detection: kalau HOME=/home/adminuser (Streamlit Cloud convention) atau
    /mount/src ada (mount path Streamlit Cloud).
    """
    is_cloud = (
        os.environ.get("HOME") == "/home/adminuser"
        or os.path.exists("/mount/src")
    )
    return "/tmp/fastf1_cache" if is_cloud else CACHE_DIR


def _setup_cache():
    cache_dir = _resolve_cache_dir()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
    except Exception:
        # Worst case: FastF1 jalan tanpa disk cache (slower tapi tetap work)
        pass


_setup_cache()


# ── Session loader ──────────────────────────────────────────────────────────

_IS_CLOUD = (
    os.environ.get("HOME") == "/home/adminuser"
    or os.path.exists("/mount/src")
)


# CRITICAL: gunakan cache_resource (BUKAN cache_data) untuk FastF1 Session.
# cache_data pickles return value → FastF1 Session object kehilangan state
# "data loaded" setelah unpickle → DataNotLoadedError saat akses .laps/.telemetry.
# cache_resource menyimpan object as-is di memory (no pickling) → state preserved.
@st.cache_resource(ttl=3600, show_spinner=False)
def load_session(year: int, gp: str, session_type: str) -> fastf1.core.Session | None:
    """
    Load a FastF1 session. Returns None on failure (never raises).
    Cached as RESOURCE (not data) — preserves loaded state across cache hits.

    Loading strategy:
    - Local: load semua (laps + telemetry + weather + messages)
    - Cloud: laps WAJIB sukses; telemetry/weather/messages best-effort.
    """
    try:
        session = fastf1.get_session(year, gp, session_type)
    except Exception as e:
        st.error(f"Failed to get session: {e}")
        return None

    # Phase 1: WAJIB — laps + messages (race events parsing)
    try:
        session.load(laps=True, telemetry=False, weather=False, messages=True)
    except Exception as e:
        st.error(
            f"Failed to load lap data: {e}\n\n"
            "FastF1 might be rate-limited atau F1 API unreachable. "
            "Coba session lain atau tunggu beberapa menit."
        )
        return None

    # Phase 2: OPSIONAL — telemetry (heaviest, paling sering fail di cloud)
    try:
        session.load(laps=False, telemetry=True, weather=False, messages=False)
    except Exception:
        if not _IS_CLOUD:
            st.warning("Telemetry data unavailable — some features may be limited.")

    # Phase 3: OPSIONAL — weather
    try:
        session.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception:
        pass

    return session


# ── Lap data ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_laps(_session: fastf1.core.Session, session_id: str) -> pd.DataFrame:
    """
    Extract clean lap DataFrame from session.
    Adds human-readable lap time (seconds) column.

    `session_id` is a hashable key (e.g. f"{year}_{gp}_{session_type}") so
    Streamlit's cache can distinguish between sessions — `_session` is
    skipped from the cache key by Streamlit.

    Returns empty DataFrame kalau FastF1 raise DataNotLoadedError atau
    session.laps unavailable (caller pages udah handle empty).
    """
    try:
        if _session.laps is None or len(_session.laps) == 0:
            return pd.DataFrame()
        laps = _session.laps.copy()
    except Exception:
        return pd.DataFrame()

    # Convert LapTime timedelta → float seconds for plotting
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    # Filter out obvious outliers: pit laps, safety car laps, and
    # statistical outliers (> 2 std dev above median per driver)
    valid = laps[
        laps["LapTimeSeconds"].notna()
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
    ].copy()

    # Remove statistical outliers per driver. Pakai boolean mask per-group
    # (bukan groupby.apply) supaya bebas dari DeprecationWarning pandas 2.2
    # soal apply yang menyentuh grouping column.
    keep_mask = pd.Series(True, index=valid.index)
    for _driver, group in valid.groupby("Driver"):
        median = group["LapTimeSeconds"].median()
        std    = group["LapTimeSeconds"].std()
        if pd.isna(std):
            # Single-lap driver (qualifying) — std is NaN, keep as-is
            continue
        keep_mask.loc[group.index] = group["LapTimeSeconds"] < median + 2 * std

    valid = valid[keep_mask]
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
    """
    Return sorted list of driver abbreviations present in the session.
    Coba dari laps dulu, fallback ke results (untuk kasus laps fail load
    di cloud environment).
    """
    # 1. Try session.laps["Driver"]
    try:
        if _session.laps is not None and len(_session.laps) > 0:
            if "Driver" in _session.laps.columns:
                drivers = _session.laps["Driver"].dropna().unique().tolist()
                if drivers:
                    return sorted([str(d) for d in drivers])
    except Exception:
        pass

    # 2. Fallback: session.results["Abbreviation"]
    try:
        if hasattr(_session, "results") and _session.results is not None:
            if len(_session.results) > 0 and "Abbreviation" in _session.results.columns:
                drivers = _session.results["Abbreviation"].dropna().unique().tolist()
                if drivers:
                    return sorted([str(d) for d in drivers])
    except Exception:
        pass

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
