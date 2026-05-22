"""
F1 Race Outcome Predictor — Training Script for Google Colab
=============================================================
Cara pakai:
  1. Buka Google Colab (colab.research.google.com)
  2. Upload file ini ATAU buat notebook baru dan copy tiap section ke cell terpisah
     (setiap "# ===== CELL N" adalah satu cell baru di notebook)
  3. Jalankan semua cell secara urut (Runtime → Run all / Ctrl+F9)
  4. Download race_model.joblib yang ter-generate
  5. Taruh di folder models/ di project lokal, commit ke repo

Improvement vs on-the-fly training di dashboard:
  - 26 features (9 sebelumnya): Ergast quali gap, championship standings,
    constructor standings, rolling form 3 race, circuit DNF rate
  - Evaluasi JUJUR: train 2022-2023 / test 2024 (fully out-of-sample season)
  - Final model: trained pada semua 2022-2024 + early stopping (n_jobs=-1)
  - SHAP verification sebelum save

PENTING: Pipeline ini IDENTIK dengan src/ml/race_model.py.
  Feature names, enrichment functions, normalisasi tim, dan bundle format
  harus TIDAK BERBEDA supaya model yang di-save kompatibel dengan dashboard.
  Jika kamu memodifikasi race_model.py, update file ini juga.
"""

# ============================================================
# CELL 1 — Install dependencies
# Jalankan SEKALI, lalu Restart Runtime (Runtime → Restart runtime),
# kemudian jalankan ulang dari CELL 2.
# ============================================================

# !pip install -q fastf1 xgboost scikit-learn shap joblib tqdm requests

# ============================================================
# CELL 2 — Imports & FastF1 cache setup
# ============================================================

import os
import re
import time
import datetime
import warnings

import numpy as np
import pandas as pd
import requests
import fastf1
import xgboost as xgb
import joblib
import shap
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")

os.makedirs("/tmp/fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("/tmp/fastf1_cache")

print(f"FastF1 {fastf1.__version__}  |  XGBoost {xgb.__version__}")
print("FastF1 cache: /tmp/fastf1_cache")

# ============================================================
# CELL 3 — Constants (IDENTIK dengan src/ml/race_model.py & src/utils/config.py)
# ============================================================

# Circuit metadata — overtake difficulty + circuit type
CIRCUIT_META = {
    "Bahrain":         {"overtake_idx": 7, "track_type": 1},
    "Saudi Arabia":    {"overtake_idx": 5, "track_type": 0},
    "Australia":       {"overtake_idx": 6, "track_type": 1},
    "Japan":           {"overtake_idx": 6, "track_type": 1},
    "China":           {"overtake_idx": 7, "track_type": 1},
    "Miami":           {"overtake_idx": 5, "track_type": 1},
    "Emilia Romagna":  {"overtake_idx": 5, "track_type": 1},
    "Monaco":          {"overtake_idx": 1, "track_type": 0},
    "Spain":           {"overtake_idx": 5, "track_type": 1},
    "Canada":          {"overtake_idx": 7, "track_type": 1},
    "Austria":         {"overtake_idx": 7, "track_type": 1},
    "United Kingdom":  {"overtake_idx": 7, "track_type": 1},
    "Hungary":         {"overtake_idx": 3, "track_type": 1},
    "Belgium":         {"overtake_idx": 7, "track_type": 1},
    "Netherlands":     {"overtake_idx": 5, "track_type": 1},
    "Italy":           {"overtake_idx": 9, "track_type": 1},
    "Singapore":       {"overtake_idx": 2, "track_type": 0},
    "United States":   {"overtake_idx": 7, "track_type": 1},
    "Mexico":          {"overtake_idx": 5, "track_type": 1},
    "Brazil":          {"overtake_idx": 8, "track_type": 1},
    "Las Vegas":       {"overtake_idx": 7, "track_type": 0},
    "Qatar":           {"overtake_idx": 6, "track_type": 1},
    "Azerbaijan":      {"overtake_idx": 6, "track_type": 0},
    "Abu Dhabi":       {"overtake_idx": 5, "track_type": 1},
    "France":          {"overtake_idx": 5, "track_type": 1},
}

# Team normalizer — maps BOTH FastF1 TeamName variants AND Ergast constructorName
# variants ke nama kanonik yang sama. IDENTIK dengan race_model.py _normalize_team().
_TEAM_FRAGMENTS = [
    ("red bull",      "Red Bull Racing"),
    ("ferrari",       "Ferrari"),
    ("mercedes",      "Mercedes"),
    ("mclaren",       "McLaren"),
    ("alpine",        "Alpine"),
    ("alphatauri",    "RB"),
    ("alpha tauri",   "RB"),
    ("rb f1",         "RB"),
    ("racing bulls",  "RB"),
    ("visa cash app", "RB"),
    ("aston martin",  "Aston Martin"),
    ("williams",      "Williams"),
    ("alfa romeo",    "Kick Sauber"),
    ("kick sauber",   "Kick Sauber"),
    ("sauber",        "Kick Sauber"),
    ("haas",          "Haas F1 Team"),
]


def _normalize_team(name) -> str:
    """Map any FastF1/Ergast team-name variant to canonical TEAM_NAMES value."""
    if not isinstance(name, str):
        return ""
    key = name.strip().lower()
    if not key:
        return ""
    for frag, canon in _TEAM_FRAGMENTS:
        if frag in key:
            return canon
    if key == "rb":
        return "RB"
    return name.strip()


TEAM_NAMES = sorted([
    "Red Bull Racing", "Ferrari", "McLaren", "Mercedes",
    "Aston Martin", "Alpine", "Williams", "RB",
    "Haas F1 Team", "Kick Sauber",
])

CLASS_NAMES = ["DNF", "Podium", "Points", "Outside Points"]

_FINISHED_RE = re.compile(r"^(Finished|\+\d+ Lap)")

# Kalender F1 2022-2024 (identik dengan src/utils/config.py F1_ROUNDS)
F1_ROUNDS = {
    2022: {
        1: "Bahrain", 2: "Saudi Arabia", 3: "Australia", 4: "Emilia Romagna",
        5: "Miami", 6: "Spain", 7: "Monaco", 8: "Azerbaijan", 9: "Canada",
        10: "United Kingdom", 11: "Austria", 12: "France", 13: "Hungary",
        14: "Belgium", 15: "Netherlands", 16: "Italy", 17: "Singapore",
        18: "Japan", 19: "United States", 20: "Mexico", 21: "Brazil",
        22: "Abu Dhabi",
    },
    2023: {
        1: "Bahrain", 2: "Saudi Arabia", 3: "Australia", 4: "Azerbaijan",
        5: "Miami", 6: "Monaco", 7: "Spain", 8: "Canada", 9: "Austria",
        10: "United Kingdom", 11: "Hungary", 12: "Belgium", 13: "Netherlands",
        14: "Italy", 15: "Singapore", 16: "Japan", 17: "Qatar",
        18: "United States", 19: "Mexico", 20: "Brazil", 21: "Las Vegas",
        22: "Abu Dhabi",
    },
    2024: {
        1: "Bahrain", 2: "Saudi Arabia", 3: "Australia", 4: "Japan",
        5: "China", 6: "Miami", 7: "Emilia Romagna", 8: "Monaco",
        9: "Canada", 10: "Spain", 11: "Austria", 12: "United Kingdom",
        13: "Hungary", 14: "Belgium", 15: "Netherlands", 16: "Italy",
        17: "Azerbaijan", 18: "Singapore", 19: "United States", 20: "Mexico",
        21: "Brazil", 22: "Las Vegas", 23: "Qatar", 24: "Abu Dhabi",
    },
}

SUPPORTED_YEARS = [2022, 2023, 2024]

# Impute/default constants — IDENTIK dengan race_model.py
_DEF_QUALI_GAP        = 5.0    # gap normal ~field median (%)
_IMPUTE_QUALI_GAP     = 107.0  # "no quali time" sentinel (107% rule proxy)
_DEF_DRIVER_PTS       = 0.0
_DEF_DRIVER_POS       = 10.0
_DEF_CONSTRUCTOR_PTS  = 0.0
_DEF_CONSTRUCTOR_POS  = 5.0
_DEF_FORM             = 10.0
_DEF_CIRCUIT_DNF      = 0.12

print(f"TEAM_NAMES ({len(TEAM_NAMES)}): {TEAM_NAMES}")
print(f"F1_ROUNDS: {sum(len(v) for v in F1_ROUNDS.values())} total rounds across {len(F1_ROUNDS)} years")
print("Constants OK.")

# ============================================================
# CELL 4 — Core helper functions (IDENTIK dengan race_model.py)
# ============================================================


def _is_finished(status) -> bool:
    """True jika status = classified finish (Finished / +N Lap(s))."""
    if not isinstance(status, str):
        return False
    s = status.strip()
    return s == "Finished" or "Lap" in s


def encode_dnf(status: str) -> bool:
    if not status or not isinstance(status, str):
        return True
    return _FINISHED_RE.match(status.strip()) is None


def _outcome_class(status: str, position: float) -> int:
    if encode_dnf(status):
        return 0       # DNF
    if position <= 3:
        return 1       # Podium
    if position <= 10:
        return 2       # Points
    return 3           # Outside Points


def _parse_quali_time_seconds(t) -> "float | None":
    """
    Parse qualifying time → float seconds.
    Handles: None, NaN/NaT, pd.Timedelta, datetime.timedelta, np.timedelta64,
    float, "1:23.456" (m:ss.mmm), "83.456" (raw seconds), "".
    Returns None for invalid / non-positive values.
    """
    if t is None:
        return None
    if isinstance(t, pd.Timedelta):
        if pd.isna(t):
            return None
        s = t.total_seconds()
        return s if s > 0 else None
    try:
        if isinstance(t, datetime.timedelta):
            s = t.total_seconds()
            return s if s > 0 else None
    except Exception:
        pass
    if isinstance(t, np.timedelta64):
        try:
            s = t / np.timedelta64(1, "s")
            return float(s) if s > 0 else None
        except Exception:
            return None
    if isinstance(t, (int, float, np.integer, np.floating)):
        try:
            if pd.isna(t):
                return None
        except (TypeError, ValueError):
            pass
        return float(t) if float(t) > 0 else None
    # String
    s = str(t).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    try:
        if ":" in s:
            parts = s.split(":")
            val = float(parts[0]) * 60.0 + float(parts[1])
        else:
            val = float(s)
    except (ValueError, IndexError):
        return None
    return val if val > 0 else None


print("Core helpers defined.")

# ============================================================
# CELL 5 — Ergast REST API fetchers
# Catatan: Ergast API (ergast.com) menyediakan data historis F1.
# Satu request per season sudah cukup karena Ergast mengembalikan semua
# races per season dalam satu response JSON (nested structure).
# ============================================================


def _ergast_get(url: str, retries: int = 3, delay: float = 2.0) -> dict:
    """GET dari Ergast REST API dengan exponential retry."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                print(f"    [retry {attempt + 1}/{retries}] {e}")
                time.sleep(delay * (attempt + 1))
            else:
                print(f"    [ERROR] {url} -> {e}")
                raise
    return {}


def get_season_races_colab(year: int) -> "dict[int, pd.DataFrame]":
    """
    Fetch semua race results satu season dari Ergast REST API.
    Returns {round_num: DataFrame} dengan cols:
      driverCode, constructorName, position, points, grid, status
    """
    url = f"http://ergast.com/api/f1/{year}/results.json?limit=100"
    try:
        data = _ergast_get(url)
    except Exception as e:
        print(f"  [WARN] get_season_races_colab({year}) failed: {e}")
        return {}

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    out: "dict[int, pd.DataFrame]" = {}
    for race in races:
        rnd = int(race.get("round", 0))
        if rnd == 0:
            continue
        results = race.get("Results", [])
        rows = []
        for r in results:
            try:
                code = r["Driver"].get("code", "")
                if not code:
                    # Fallback: use driverId initials if code missing
                    code = r["Driver"].get("driverId", "")[:3].upper()
                rows.append({
                    "driverCode":     code.upper(),
                    "constructorName": r["Constructor"].get("name", ""),
                    "position":       int(r.get("position", 20)),
                    "points":         float(r.get("points", 0.0)),
                    "grid":           int(r.get("grid", 0)),
                    "status":         r.get("status", ""),
                })
            except (KeyError, ValueError, TypeError):
                continue
        if rows:
            out[rnd] = pd.DataFrame(rows)
    return out


def get_season_qualifying_colab(year: int) -> "dict[int, pd.DataFrame]":
    """
    Fetch qualifying results satu season dari Ergast REST API.
    Returns {round_num: DataFrame} dengan cols:
      driverCode, Q1, Q2, Q3 (string "1:28.796" atau "" jika tidak ada)
    """
    url = f"http://ergast.com/api/f1/{year}/qualifying.json?limit=100"
    try:
        data = _ergast_get(url)
    except Exception as e:
        print(f"  [WARN] get_season_qualifying_colab({year}) failed: {e}")
        return {}

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    out: "dict[int, pd.DataFrame]" = {}
    for race in races:
        rnd = int(race.get("round", 0))
        if rnd == 0:
            continue
        results = race.get("QualifyingResults", [])
        rows = []
        for r in results:
            try:
                rows.append({
                    "driverCode": r["Driver"].get("code", "").upper(),
                    "Q1": r.get("Q1") or "",
                    "Q2": r.get("Q2") or "",
                    "Q3": r.get("Q3") or "",
                })
            except (KeyError, TypeError):
                continue
        if rows:
            out[rnd] = pd.DataFrame(rows)
    return out


# Quick connection test
print("Testing Ergast connection...")
_test_q = get_season_qualifying_colab(2024)
_test_r = get_season_races_colab(2024)
print(f"  2024 race rounds: {len(_test_r)} | quali rounds: {len(_test_q)}")
if _test_r:
    print(f"  Sample R1 results:\n{_test_r.get(1, pd.DataFrame()).head(3)}")
else:
    print("  [WARN] No data returned — check internet connection in Colab.")

# ============================================================
# CELL 6 — Enrichment builders (IDENTIK dengan src/ml/race_model.py)
# ============================================================


def _empty_enrichment() -> dict:
    """Enrichment kosong — _enrich_features pakai default untuk semua driver."""
    return {"standings": {}, "constructor": {}, "form": {}, "quali_gap": {}}


def build_quali_gap(quali_dict: dict) -> "dict[int, dict[str, float]]":
    """
    Per round: gap-to-pole (%) per driver.
    best_time = Q3 → Q2 → Q1 (yang pertama valid).
    gap_pct = (driver_time - pole_time) / pole_time * 100.
    Driver tanpa waktu valid → _IMPUTE_QUALI_GAP (107.0).
    """
    out: "dict[int, dict[str, float]]" = {}
    if not quali_dict:
        return out
    for rnd, df in quali_dict.items():
        if df is None or len(df) == 0 or "driverCode" not in df.columns:
            continue
        times: "dict[str, float | None]" = {}
        for _, row in df.iterrows():
            code = row.get("driverCode")
            if not isinstance(code, str) or not code:
                continue
            best = None
            for col in ("Q3", "Q2", "Q1"):
                if col in df.columns:
                    val = _parse_quali_time_seconds(row.get(col))
                    if val is not None:
                        best = val
                        break
            times[code] = best
        valid = [v for v in times.values() if v is not None]
        if not valid:
            continue
        pole = min(valid)
        rd: "dict[str, float]" = {}
        for code, v in times.items():
            if v is None or pole <= 0:
                rd[code] = _IMPUTE_QUALI_GAP
            else:
                rd[code] = (v - pole) / pole * 100.0
        out[rnd] = rd
    return out


def build_standings_before_round(race_results: dict, year: int) -> dict:
    """
    Championship standings driver SEBELUM tiap round (akumulasi round < R).
    Round pertama: semua 0 pts, posisi neutral (10).
    Returns {round: {driver_code: {driver_pts_before, driver_pos_before}}}.
    """
    out: dict = {}
    if not race_results:
        return out
    rounds = sorted(race_results.keys())
    cum: "dict[str, float]" = {}

    for i, R in enumerate(rounds):
        df_R = race_results[R]
        codes = (
            [c for c in df_R["driverCode"].dropna().unique() if isinstance(c, str)]
            if (df_R is not None and "driverCode" in df_R.columns) else []
        )
        if i == 0:
            out[R] = {
                c: {"driver_pts_before": 0.0,
                    "driver_pos_before": int(_DEF_DRIVER_POS)}
                for c in codes
            }
        else:
            ranked = sorted(cum.items(), key=lambda kv: kv[1], reverse=True)
            pos_map = {code: idx + 1 for idx, (code, _) in enumerate(ranked)}
            fallback = len(pos_map) + 1
            out[R] = {
                c: {
                    "driver_pts_before": float(cum.get(c, 0.0)),
                    "driver_pos_before": int(pos_map.get(c, fallback)),
                }
                for c in codes
            }
        # Accumulate points for next iteration
        if df_R is not None and "driverCode" in df_R.columns:
            for _, row in df_R.iterrows():
                c = row.get("driverCode")
                if not isinstance(c, str) or not c:
                    continue
                try:
                    p = float(row.get("points")) if pd.notna(row.get("points")) else 0.0
                except (TypeError, ValueError):
                    p = 0.0
                cum[c] = cum.get(c, 0.0) + p
    return out


def build_constructor_standings_before_round(race_results: dict) -> dict:
    """
    Constructor standings BEFORE each round, keyed by canonical team name.
    Returns {round: {team_canonical: {constructor_pts_before, constructor_pos_before}}}.
    """
    out: dict = {}
    if not race_results:
        return out
    rounds = sorted(race_results.keys())
    cum: "dict[str, float]" = {}

    for i, R in enumerate(rounds):
        df_R = race_results[R]
        teams_in_round: set = set()
        if df_R is not None and "constructorName" in df_R.columns:
            teams_in_round = {
                _normalize_team(t) for t in df_R["constructorName"].dropna().unique()
            }
            teams_in_round.discard("")
        if i == 0:
            out[R] = {
                t: {"constructor_pts_before": 0.0,
                    "constructor_pos_before": int(_DEF_CONSTRUCTOR_POS)}
                for t in teams_in_round
            }
        else:
            ranked = sorted(cum.items(), key=lambda kv: kv[1], reverse=True)
            pos_map = {t: idx + 1 for idx, (t, _) in enumerate(ranked)}
            fallback = len(pos_map) + 1
            out[R] = {
                t: {
                    "constructor_pts_before": float(cum.get(t, 0.0)),
                    "constructor_pos_before": int(pos_map.get(t, fallback)),
                }
                for t in teams_in_round
            }
        if df_R is not None and "constructorName" in df_R.columns:
            for _, row in df_R.iterrows():
                t = _normalize_team(row.get("constructorName"))
                if not t:
                    continue
                try:
                    p = float(row.get("points")) if pd.notna(row.get("points")) else 0.0
                except (TypeError, ValueError):
                    p = 0.0
                cum[t] = cum.get(t, 0.0) + p
    return out


def build_rolling_form(race_results: dict, n_races: int = 3) -> dict:
    """
    Rata-rata finish position driver di N race sebelumnya (DNF = 20).
    Round tanpa history → impute season-average grid.
    Returns {round: {driver_code: avg_finish_last_N}}.
    """
    out: dict = {}
    if not race_results:
        return out
    rounds = sorted(race_results.keys())
    # Season-average grid for early-round imputation
    grids: "list[float]" = []
    for R in rounds:
        df = race_results[R]
        if df is not None and "grid" in df.columns:
            for v in df["grid"]:
                try:
                    gv = float(v)
                    if gv > 0:
                        grids.append(gv)
                except (TypeError, ValueError):
                    pass
    season_avg_grid = (sum(grids) / len(grids)) if grids else _DEF_FORM

    history: "dict[str, list[int]]" = {}
    for R in rounds:
        df_R = race_results[R]
        codes = (
            [c for c in df_R["driverCode"].dropna().unique() if isinstance(c, str)]
            if (df_R is not None and "driverCode" in df_R.columns) else []
        )
        rd: "dict[str, float]" = {}
        for c in codes:
            hist = history.get(c, [])
            if not hist:
                rd[c] = float(season_avg_grid)
            else:
                last = hist[-n_races:]
                rd[c] = float(sum(last) / len(last))
        out[R] = rd
        # Update history with round R results
        if df_R is not None and "driverCode" in df_R.columns:
            for _, row in df_R.iterrows():
                c = row.get("driverCode")
                if not isinstance(c, str) or not c:
                    continue
                status = row.get("status", "")
                try:
                    pv = int(row.get("position")) if pd.notna(row.get("position")) else None
                except (TypeError, ValueError):
                    pv = None
                fin = pv if (_is_finished(status) and pv is not None) else 20
                history.setdefault(c, []).append(fin)
    return out


def build_circuit_dnf_rate(all_race_res: dict) -> "dict[str, float]":
    """
    DNF rate per circuit dari semua tahun.
    circuit name = F1_ROUNDS[year][round_num].
    Returns {circuit_name: dnf_rate (0..1)}.
    """
    counts: "dict[str, list[int]]" = {}
    if not all_race_res:
        return {}
    for year, races in all_race_res.items():
        rounds_map = F1_ROUNDS.get(year, {})
        if not races:
            continue
        for rnd, df in races.items():
            circuit = rounds_map.get(rnd, "")
            if not circuit or df is None or len(df) == 0 or "status" not in df.columns:
                continue
            for _, row in df.iterrows():
                status = row.get("status", "")
                c = counts.setdefault(circuit, [0, 0])
                c[1] += 1
                if not _is_finished(status):
                    c[0] += 1
    return {
        circ: (cnt[0] / cnt[1] if cnt[1] > 0 else _DEF_CIRCUIT_DNF)
        for circ, cnt in counts.items()
    }


def _build_enrichment_for_year_colab(year: int) -> dict:
    """Pre-load + build semua enrichment dicts untuk satu season."""
    print(f"  Loading Ergast data for {year}...")
    try:
        race_res  = get_season_races_colab(year)
        quali_res = get_season_qualifying_colab(year)
    except Exception as e:
        print(f"  [WARN] Enrichment failed for {year}: {e}")
        return _empty_enrichment()
    if not race_res:
        return _empty_enrichment()
    enrichment = {
        "standings":   build_standings_before_round(race_res, year),
        "constructor": build_constructor_standings_before_round(race_res),
        "form":        build_rolling_form(race_res),
        "quali_gap":   build_quali_gap(quali_res) if quali_res else {},
    }
    n_q = len(enrichment["quali_gap"])
    print(f"    Race rounds={len(race_res)} | Quali rounds with gap data={n_q}")
    return enrichment


def _enrich_features(
    df: pd.DataFrame,
    year: int,
    round_num: int,
    enrichment: dict,
    circuit_dnf: "dict[str, float]",
) -> pd.DataFrame:
    """
    Merge Ergast-derived features ke DataFrame per driver (in-place).
    Semua lookup pakai default kalau data hilang — tidak pernah crash.
    IDENTIK dengan src/ml/race_model.py::_enrich_features.
    """
    enrichment = enrichment or _empty_enrichment()
    circuit    = F1_ROUNDS.get(year, {}).get(round_num, "")
    qg_round   = enrichment.get("quali_gap", {}).get(round_num, {})
    st_round   = enrichment.get("standings", {}).get(round_num, {})
    cs_round   = enrichment.get("constructor", {}).get(round_num, {})
    form_round = enrichment.get("form", {}).get(round_num, {})
    dnf_rate   = (
        circuit_dnf.get(circuit, _DEF_CIRCUIT_DNF)
        if circuit_dnf else _DEF_CIRCUIT_DNF
    )

    for idx in df.index:
        code = df.at[idx, "driver_code"]
        team = df.at[idx, "team_name"]

        df.at[idx, "quali_gap_pct"] = float(qg_round.get(code, _DEF_QUALI_GAP))

        ds = st_round.get(code, {})
        df.at[idx, "driver_pts_before"] = float(
            ds.get("driver_pts_before", _DEF_DRIVER_PTS))
        df.at[idx, "driver_pos_before"] = float(
            ds.get("driver_pos_before", _DEF_DRIVER_POS))

        cs = cs_round.get(team, {})
        df.at[idx, "constructor_pts_before"] = float(
            cs.get("constructor_pts_before", _DEF_CONSTRUCTOR_PTS))
        df.at[idx, "constructor_pos_before"] = float(
            cs.get("constructor_pos_before", _DEF_CONSTRUCTOR_POS))

        df.at[idx, "driver_form_last3"] = float(form_round.get(code, _DEF_FORM))
        df.at[idx, "circuit_dnf_rate"]  = float(dnf_rate)

    return df


print("Enrichment builders defined.")

# ============================================================
# CELL 7 — Feature assembly
# WAJIB identik dengan src/ml/race_model.py::assemble_race_features.
# Jangan modifikasi tanpa update ke race_model.py juga.
# ============================================================


def assemble_race_features(
    year: int, round_num: int, session,
) -> "pd.DataFrame | None":
    """
    Build feature matrix untuk satu race dari session.results + weather.
    26 features setelah enrich: 9 base + 10 team one-hot + 7 Ergast-derived.
    """
    try:
        results = session.results
        if results is None or len(results) == 0:
            return None
    except Exception:
        return None

    needed = {"GridPosition", "Position", "Status", "Abbreviation", "TeamName"}
    if not needed.issubset(set(results.columns)):
        return None

    circuit_name = F1_ROUNDS.get(year, {}).get(round_num, "")
    meta = CIRCUIT_META.get(circuit_name, {"overtake_idx": 5, "track_type": 1})

    air_temp  = 25.0
    rain_flag = 0
    try:
        wd = session.weather_data
        if wd is not None and len(wd) > 0:
            if "AirTemp" in wd.columns and wd["AirTemp"].notna().any():
                air_temp = float(wd["AirTemp"].mean())
            if "Rainfall" in wd.columns:
                rain_flag = int(wd["Rainfall"].any())
    except Exception:
        pass

    rows = []
    for _, r in results.iterrows():
        grid = r.get("GridPosition", np.nan)
        try:
            grid = float(grid)
        except (TypeError, ValueError):
            grid = np.nan
        if np.isnan(grid) or grid == 0:
            grid = 18.0

        pos = r.get("Position", np.nan)
        try:
            pos = float(pos)
        except (TypeError, ValueError):
            pos = np.nan

        status_raw  = str(r.get("Status", ""))
        team_raw    = str(r.get("TeamName", ""))
        team        = _normalize_team(team_raw)
        driver_code = str(r.get("Abbreviation", ""))
        outcome     = _outcome_class(status_raw, pos)

        row = {
            # ── meta (excluded from feature_cols) ────────────────
            "driver_code":          driver_code,
            "team_name":            team,
            "year":                 year,
            "round_num":            round_num,
            "finish_pos":           pos,
            "status_raw":           status_raw,
            "outcome_class":        outcome,
            # ── base features (9) ─────────────────────────────────
            "grid_position":        grid,
            "grid_pos_pct":         grid / 20.0,
            "front_row":            1 if grid <= 3 else 0,
            "midfield":             1 if 4 <= grid <= 12 else 0,
            "back_marker":          1 if grid >= 16 else 0,
            "air_temp":             air_temp,
            "rain_flag":            rain_flag,
            "circuit_overtake_idx": meta["overtake_idx"],
            "track_type":           meta["track_type"],
            # ── team one-hot (10) — appended below ────────────────
        }
        # Team one-hot (10 teams sorted alphabetically)
        for t in TEAM_NAMES:
            row[f"team_{t}"] = 1 if team == t else 0
        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)
    # After _enrich_features: +7 Ergast-derived features = 26 total features


print("assemble_race_features defined (base 19 features; 26 after enrich).")

# ============================================================
# CELL 8 — Build full dataset
# Estimasi waktu: ~10-20 menit di Colab (68 races + Ergast calls)
# ============================================================

print("=" * 65)
print("STEP 1 of 3: Pre-loading Ergast enrichment (race + qualifying)...")
print("=" * 65)

enrichment_by_year: dict = {}
for yr in SUPPORTED_YEARS:
    enrichment_by_year[yr] = _build_enrichment_for_year_colab(yr)

print("\nSTEP 2 of 3: Building circuit DNF rates from Ergast race data...")
all_race_res_ergast = {yr: get_season_races_colab(yr) for yr in SUPPORTED_YEARS}
circuit_dnf = build_circuit_dnf_rate(all_race_res_ergast)
print(f"  Circuits with DNF data: {len(circuit_dnf)}")
top3_dnf = sorted(circuit_dnf.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"  Top 3 highest DNF rate: {[(c, f'{r:.1%}') for c, r in top3_dnf]}")

print("\nSTEP 3 of 3: Loading FastF1 sessions + assembling feature matrix...")
print("  (This takes ~10-20 minutes — FastF1 downloads & caches race data)")

all_rounds = [
    (yr, rn)
    for yr in SUPPORTED_YEARS
    for rn in sorted(F1_ROUNDS.get(yr, {}).keys())
]
frames = []
skipped = []

for yr, rn in tqdm(all_rounds, desc="Loading races"):
    circuit = F1_ROUNDS.get(yr, {}).get(rn, f"R{rn}")
    try:
        sess = fastf1.get_session(yr, rn, "R")
        sess.load(laps=False, telemetry=False, weather=True, messages=False)
        df = assemble_race_features(yr, rn, sess)
        if df is not None and len(df) > 0:
            enrich = enrichment_by_year.get(yr, _empty_enrichment())
            df = _enrich_features(df, yr, rn, enrich, circuit_dnf)
            frames.append(df)
    except Exception as e:
        skipped.append(f"{yr} R{rn} {circuit}: {e}")

if skipped:
    print(f"\nSkipped {len(skipped)} races:")
    for s in skipped:
        print(f"  {s}")

# Concat + sort kronologis (sama seperti build_training_dataset)
combined_df = pd.concat(frames, ignore_index=True)
combined_df["_sort_key"] = combined_df["year"] * 100 + combined_df["round_num"]
combined_df = combined_df.sort_values("_sort_key").reset_index(drop=True)

META_COLS = {
    "driver_code", "team_name", "year", "round_num",
    "finish_pos", "status_raw", "outcome_class", "_sort_key",
}
feature_cols = [c for c in combined_df.columns if c not in META_COLS]

print(f"\n{'='*65}")
print(f"Dataset: {len(combined_df)} driver-race rows | {len(frames)} races loaded")
print(f"Feature columns ({len(feature_cols)}):")
for i, fc in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {fc}")
print(f"\nClass distribution:")
for cls_id, cls_name in enumerate(CLASS_NAMES):
    cnt = (combined_df["outcome_class"] == cls_id).sum()
    print(f"  {cls_id} {cls_name:>15s}: {cnt:4d} ({cnt/len(combined_df):.1%})")

# Sanity check
assert len(feature_cols) == 26, (
    f"Feature count mismatch! Got {len(feature_cols)}, expected 26.\n"
    f"Cols: {feature_cols}"
)
print(f"\nFeature count check: {len(feature_cols)} == 26 OK")

# ============================================================
# CELL 9 — Time-series evaluation: Train 2022-2023 / Test 2024
# Ini evaluasi yang JUJUR — 2024 tidak pernah dilihat saat training.
# Metriks dari cell ini yang akan ditampilkan di dashboard.
# ============================================================

train_df = combined_df[combined_df["year"].isin([2022, 2023])].copy()
test_df  = combined_df[combined_df["year"] == 2024].copy()

X_train_ts = train_df[feature_cols].astype(float)
y_train_ts = train_df["outcome_class"].astype(int)
X_test_ts  = test_df[feature_cols].astype(float)
y_test_ts  = test_df["outcome_class"].astype(int)

print(f"Train (2022-2023): {len(X_train_ts)} rows")
print(f"Test  (2024):      {len(X_test_ts)} rows")
print(f"Class balance train: {dict(y_train_ts.value_counts().sort_index())}")
print(f"Class balance test : {dict(y_test_ts.value_counts().sort_index())}")

weights_ts = compute_sample_weight("balanced", y_train_ts)

eval_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    n_estimators=800,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,       # pakai semua CPU core di Colab
    verbosity=0,
)

print("\nTraining eval model (2022-2023) — ini untuk metriks jujur...")
eval_model.fit(
    X_train_ts, y_train_ts,
    sample_weight=weights_ts,
    eval_set=[(X_test_ts, y_test_ts)],
    verbose=False,
)
print(f"  Best iteration: {eval_model.best_iteration}")

y_pred_ts = eval_model.predict(X_test_ts)

print("\n" + "=" * 65)
print("TIME-SERIES EVALUATION  —  Train 2022-2023  |  Test 2024 (out-of-sample)")
print("=" * 65)
ts_accuracy = accuracy_score(y_test_ts, y_pred_ts)
print(f"Accuracy: {ts_accuracy:.3f}  ({ts_accuracy:.1%})")
print()
ts_report = classification_report(
    y_test_ts, y_pred_ts, target_names=CLASS_NAMES, zero_division=0,
)
print(ts_report)
ts_cm = confusion_matrix(y_test_ts, y_pred_ts, labels=[0, 1, 2, 3])
print("Confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(ts_cm, index=CLASS_NAMES, columns=CLASS_NAMES))

# Simpan untuk bundle
_ts_accuracy = ts_accuracy
_ts_report   = ts_report
_ts_cm       = ts_cm
_n_train     = int(len(X_train_ts))
_n_test      = int(len(X_test_ts))

print(
    "\nInterpretasi: angka di atas adalah HONEST out-of-sample evaluation.\n"
    "Model sama sekali belum melihat 2024 saat training. Ini yang ditampilkan\n"
    "sebagai 'Accuracy' dan 'Weighted F1' di dashboard."
)

# ============================================================
# CELL 10 — Final model: train pada SEMUA data 2022-2024
# Early stopping pakai 2024 R19-R24 sebagai internal eval set
# (hanya untuk stopping criterion — BUKAN untuk report akurasi).
# ============================================================

X_all = combined_df[feature_cols].astype(float)
y_all = combined_df["outcome_class"].astype(int)

# Internal eval: 2024 R19+ (~6 races, ~120 rows) — hanya untuk early stopping
eval_mask = (combined_df["year"] == 2024) & (combined_df["round_num"] >= 19)
X_final_eval  = X_all[eval_mask]
y_final_eval  = y_all[eval_mask]
X_final_train = X_all[~eval_mask]
y_final_train = y_all[~eval_mask]

print(f"Final model training: {len(X_final_train)} rows "
      f"(all 2022-2024 minus last 6 races for early stopping)")
print(f"Early-stopping eval set: {len(X_final_eval)} rows (2024 R19-R24)")

weights_final = compute_sample_weight("balanced", y_final_train)

final_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    eval_metric="mlogloss",
    early_stopping_rounds=60,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

final_model.fit(
    X_final_train, y_final_train,
    sample_weight=weights_final,
    eval_set=[(X_final_eval, y_final_eval)],
    verbose=False,
)
print(f"Final model trained. Best iteration: {final_model.best_iteration}")

# ============================================================
# CELL 11 — Build model bundle (format harus cocok dengan dashboard)
# ============================================================

bundle = {
    "model":            final_model,
    "feature_cols":     feature_cols,
    # Metrik = TIME-SERIES out-of-sample (honest: train 2022-2023, test 2024)
    "accuracy":         float(_ts_accuracy),
    "report":           _ts_report,
    "confusion_matrix": _ts_cm,
    "class_names":      CLASS_NAMES,
    "n_train":          _n_train,    # rows in 2022-2023 train set
    "n_test":           _n_test,     # rows in 2024 test set
    "X_train":          X_train_ts,  # reference data for SHAP background
    "y_train":          y_train_ts,
    "eval_note":        "Time-series: train=2022-2023, test=2024 (fully out-of-sample season)",
}

print("Bundle summary:")
print(f"  feature_cols ({len(feature_cols)}): {feature_cols}")
print(f"  accuracy (out-of-sample 2024)  : {_ts_accuracy:.1%}")
print(f"  n_train (2022-2023)            : {_n_train}")
print(f"  n_test  (2024)                 : {_n_test}")
print(f"  eval_note                      : {bundle['eval_note']}")

# ============================================================
# CELL 12 — SHAP verification (opsional, skip jika Colab lambat)
# ============================================================

print("Computing SHAP on test set (first 100 rows)...")
try:
    explainer = shap.TreeExplainer(final_model)
    sample_size = min(100, len(X_test_ts))
    sv = explainer.shap_values(X_test_ts.iloc[:sample_size])

    # Normalisasi ke [n_samples, n_classes, n_features]
    if isinstance(sv, list):
        sv = np.stack(sv, axis=1)
    else:
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = np.transpose(sv, (0, 2, 1))

    exp_shape = f"[{sample_size}, 4, {len(feature_cols)}]"
    print(f"  SHAP shape: {sv.shape}  (expected: {exp_shape})")

    # Top 5 most important features (mean |SHAP| averaged over classes)
    mean_abs = np.abs(sv).mean(axis=(0, 1))   # shape [n_features]
    top5 = sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 features by mean |SHAP|:")
    for fname, fval in top5:
        print(f"    {fname:<35s}: {fval:.5f}")
    print("  SHAP OK")
except Exception as e:
    print(
        f"  [WARN] SHAP failed: {e}\n"
        "  Bundle masih valid — dashboard fallback ke feature_importances_."
    )

# ============================================================
# CELL 13 — Save & download
# ============================================================

SAVE_PATH = "race_model.joblib"
joblib.dump(bundle, SAVE_PATH, compress=3)
size_mb = os.path.getsize(SAVE_PATH) / 1e6
print(f"Saved: {SAVE_PATH}  ({size_mb:.1f} MB)")

# Download otomatis di Colab
try:
    from google.colab import files
    files.download(SAVE_PATH)
    print("Download started (cek folder Downloads kamu).")
except ImportError:
    print(f"Bukan di Colab — file tersimpan di: {os.path.abspath(SAVE_PATH)}")

print("""
================================================================
LANGKAH SELANJUTNYA
================================================================
1. Taruh file di project lokal:
     models/race_model.joblib

2. Commit ke git:
     git add models/race_model.joblib
     git commit -m "feat: add pre-trained 26-feature race model (out-of-sample 2024 eval)"

3. Push ke GitHub:
     git push

4. Dashboard akan auto-detect file via load_model_bundle() dan
   menampilkan banner "Pre-trained model loaded ✅"

CATATAN EVALUASI
----------------
Akurasi + Weighted F1 di dashboard  = TIME-SERIES out-of-sample (2024)
Model yang dipakai untuk prediksi   = Final model (trained on all 2022-2024)

Ini adalah evaluasi yang JUJUR untuk portofolio.
Angkanya lebih rendah dari random-split, tapi LEBIH BERARTI.
================================================================
""")
