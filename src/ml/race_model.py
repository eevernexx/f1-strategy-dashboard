"""
Race outcome classifier — multi-class XGBoost.

Pipeline:
1. assemble_race_features(year, round_num, session) -> feature matrix for one race
2. build_training_dataset(years, rounds_per_year) -> aggregate features across seasons
3. train_race_model(X, y, feature_cols) -> fitted XGBClassifier + metrics
4. predict_race_outcome(model_bundle, session, year, round_num) -> per-driver probabilities
5. compute_race_shap(model_bundle, X) -> SHAP values (optional)

Outcome classes:
  0 = DNF  |  1 = Podium (P1-P3)  |  2 = Points (P4-P10)  |  3 = Outside Points (P11+)
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    _ML_OK = True
except ImportError:
    _ML_OK = False

try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

from src.utils.config import TEAM_COLORS, F1_ROUNDS, SUPPORTED_YEARS

# Ergast season data (race + qualifying results) untuk feature enrichment.
# Di-guard: kalau season_loader / Ergast tidak tersedia, fitur enrichment
# di-impute dengan nilai default (lihat _enrich_features) — tidak crash.
try:
    from src.pipeline.season_loader import get_season_races, get_season_qualifying
    _SEASON_OK = True
except Exception:
    _SEASON_OK = False

    def get_season_races(*_a, **_k):      # type: ignore[misc]
        return {}

    def get_season_qualifying(*_a, **_k):  # type: ignore[misc]
        return {}


# ── Circuit metadata ───────────────────────────────────────────────────────

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

TEAM_NAMES = sorted(TEAM_COLORS.keys())

# Canonical team normalization. Maps BOTH FastF1 TeamName (e.g. "Red Bull
# Racing", "Alpine") AND Ergast constructorName (e.g. "Red Bull", "Alpine F1
# Team", "Alfa Romeo", "AlphaTauri") ke nama kanonik yang sama persis dengan
# TEAM_NAMES. Wajib konsisten supaya lookup constructor-standings (key dari
# Ergast) cocok dengan kolom team_name (dari FastF1) di feature matrix.
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
    """
    Map any FastF1/Ergast team-name variant to a canonical TEAM_NAMES value.
    Unknown names pass through stripped (so one-hot just stays all-zero).
    """
    if not isinstance(name, str):
        return ""
    key = name.strip().lower()
    if not key:
        return ""
    for frag, canon in _TEAM_FRAGMENTS:
        if frag in key:
            return canon
    # Bare "rb" (Ergast 2024 short form) tidak punya fragment unik di atas
    if key == "rb":
        return "RB"
    return name.strip()


_FINISHED_RE = re.compile(r"^(Finished|\+\d+ Lap)")


# ── Helpers ────────────────────────────────────────────────────────────────

def encode_dnf(status: str) -> bool:
    """Return True if the driver DNF'd (status is not Finished or +N Lap(s))."""
    if not status or not isinstance(status, str):
        return True
    return _FINISHED_RE.match(status.strip()) is None


def _outcome_class(status: str, position: float) -> int:
    if encode_dnf(status):
        return 0
    if position <= 3:
        return 1
    if position <= 10:
        return 2
    return 3


# ── Feature assembly ───────────────────────────────────────────────────────

def assemble_race_features(
    year: int,
    round_num: int,
    session,
) -> pd.DataFrame | None:
    """
    Build feature matrix for a single race from session.results + weather.
    Returns None if session.results is empty.
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

    rounds = F1_ROUNDS.get(year, {})
    circuit_name = rounds.get(round_num, "")
    meta = CIRCUIT_META.get(circuit_name, {"overtake_idx": 5, "track_type": 1})

    air_temp = 25.0
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

        status_raw = str(r.get("Status", ""))
        team_raw = str(r.get("TeamName", ""))
        team = _normalize_team(team_raw)
        driver_code = str(r.get("Abbreviation", ""))

        outcome = _outcome_class(status_raw, pos)

        row = {
            "driver_code": driver_code,
            "team_name": team,
            "year": year,
            "round_num": round_num,
            "finish_pos": pos,
            "status_raw": status_raw,
            "outcome_class": outcome,
            "grid_position": grid,
            # Grid-position interactions (engineered, tidak butuh data eksternal).
            # Selalu ada bahkan tanpa enrichment → build_prediction_features konsisten.
            "grid_pos_pct": grid / 20.0,
            "front_row":   1 if grid <= 3 else 0,
            "midfield":    1 if 4 <= grid <= 12 else 0,
            "back_marker": 1 if grid >= 16 else 0,
            "air_temp": air_temp,
            "rain_flag": rain_flag,
            "circuit_overtake_idx": meta["overtake_idx"],
            "track_type": meta["track_type"],
        }

        for t in TEAM_NAMES:
            row[f"team_{t}"] = 1 if team == t else 0

        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)


# ── Feature enrichment (Ergast-derived) ─────────────────────────────────────
#
# Semua builder di bawah memetakan {round: {driver_code/team: value}} dari hasil
# season Ergast (race + qualifying). Dipakai SEKALI per season lalu di-merge ke
# tiap race via _enrich_features. Semua punya guard: input kosong → return {} /
# default, sehingga model tetap jalan walau Ergast unreachable.

# Nilai default/impute (dipakai juga di prediction path kalau data hilang)
_DEF_QUALI_GAP        = 5.0    # gap normal ~field median (%)
_IMPUTE_QUALI_GAP     = 107.0  # "no quali time" sentinel (107% rule proxy)
_DEF_DRIVER_PTS       = 0.0
_DEF_DRIVER_POS       = 10.0
_DEF_CONSTRUCTOR_PTS  = 0.0
_DEF_CONSTRUCTOR_POS  = 5.0
_DEF_FORM             = 10.0
_DEF_CIRCUIT_DNF      = 0.12


def _empty_enrichment() -> dict:
    """Enrichment kosong (semua sub-dict empty) — _enrich_features pakai default."""
    return {"standings": {}, "constructor": {}, "form": {}, "quali_gap": {}}


def _is_finished(status) -> bool:
    """True kalau status = classified finish (Finished / +N Lap(s))."""
    if not isinstance(status, str):
        return False
    s = status.strip()
    return s == "Finished" or "Lap" in s


def _parse_quali_time_seconds(t) -> float | None:
    """
    Parse a qualifying time → float seconds.
    Robust terhadap: None, NaN/NaT, "", pd.Timedelta, datetime.timedelta,
    angka, "1:23.456", "83.456". Return None kalau invalid / non-positive.
    """
    if t is None:
        return None
    # Timedelta (fastf1.ergast sering mengembalikan ini)
    if isinstance(t, pd.Timedelta):
        if pd.isna(t):
            return None
        s = t.total_seconds()
        return s if s > 0 else None
    try:
        import datetime as _dt
        if isinstance(t, _dt.timedelta):
            s = t.total_seconds()
            return s if s > 0 else None
    except Exception:
        pass
    # numpy timedelta64
    if isinstance(t, np.timedelta64):
        try:
            s = t / np.timedelta64(1, "s")
            return float(s) if s > 0 else None
        except Exception:
            return None
    # Angka langsung (detik)
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
            mins = float(parts[0])
            secs = float(parts[1])
            val = mins * 60.0 + secs
        else:
            val = float(s)
    except (ValueError, IndexError):
        return None
    return val if val > 0 else None


def build_quali_gap(quali_dict: dict[int, pd.DataFrame]) -> dict[int, dict[str, float]]:
    """
    Per round, gap-to-pole (%) per driver.
    best_time = Q3 → Q2 → Q1 (yang pertama valid).
    gap_pct = (driver_time - pole_time) / pole_time * 100.
    Driver tanpa waktu valid → _IMPUTE_QUALI_GAP (107.0).
    Return {round: {driver_code: gap_pct}}.
    """
    out: dict[int, dict[str, float]] = {}
    if not quali_dict:
        return out
    for rnd, df in quali_dict.items():
        if df is None or len(df) == 0 or "driverCode" not in df.columns:
            continue
        times: dict[str, float | None] = {}
        for _, row in df.iterrows():
            code = row.get("driverCode")
            if not isinstance(code, str) or not code:
                continue
            best: float | None = None
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
        rd: dict[str, float] = {}
        for code, v in times.items():
            if v is None or pole <= 0:
                rd[code] = _IMPUTE_QUALI_GAP
            else:
                rd[code] = (v - pole) / pole * 100.0
        out[rnd] = rd
    return out


def build_standings_before_round(
    race_results: dict[int, pd.DataFrame],
    year: int,
) -> dict[int, dict[str, dict]]:
    """
    Championship standings driver SEBELUM tiap round (akumulasi round < R).
    Round pertama: semua 0 pts, posisi neutral (10).
    Return {round: {driver_code: {driver_pts_before, driver_pos_before}}}.
    """
    out: dict[int, dict[str, dict]] = {}
    if not race_results:
        return out
    rounds = sorted(race_results.keys())
    cum: dict[str, float] = {}  # driver_code -> running points

    for i, R in enumerate(rounds):
        df_R = race_results[R]
        codes = (
            [c for c in df_R["driverCode"].dropna().unique() if isinstance(c, str)]
            if (df_R is not None and "driverCode" in df_R.columns) else []
        )
        if i == 0:
            out[R] = {
                c: {"driver_pts_before": 0.0, "driver_pos_before": int(_DEF_DRIVER_POS)}
                for c in codes
            }
        else:
            ranked = sorted(cum.items(), key=lambda kv: kv[1], reverse=True)
            pos_map = {code: idx + 1 for idx, (code, _) in enumerate(ranked)}
            fallback_pos = len(pos_map) + 1
            out[R] = {
                c: {
                    "driver_pts_before": float(cum.get(c, 0.0)),
                    "driver_pos_before": int(pos_map.get(c, fallback_pos)),
                }
                for c in codes
            }
        # Akumulasi points round R untuk iterasi berikutnya
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


def build_constructor_standings_before_round(
    race_results: dict[int, pd.DataFrame],
) -> dict[int, dict[str, dict]]:
    """
    Constructor standings SEBELUM tiap round, dikelompokkan per team kanonik.
    Return {round: {team_canonical: {constructor_pts_before, constructor_pos_before}}}.
    """
    out: dict[int, dict[str, dict]] = {}
    if not race_results:
        return out
    rounds = sorted(race_results.keys())
    cum: dict[str, float] = {}  # team_canonical -> running points

    for i, R in enumerate(rounds):
        df_R = race_results[R]
        teams_in_round = set()
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
            fallback_pos = len(pos_map) + 1
            out[R] = {
                t: {
                    "constructor_pts_before": float(cum.get(t, 0.0)),
                    "constructor_pos_before": int(pos_map.get(t, fallback_pos)),
                }
                for t in teams_in_round
            }
        # Akumulasi
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


def build_rolling_form(
    race_results: dict[int, pd.DataFrame],
    n_races: int = 3,
) -> dict[int, dict[str, float]]:
    """
    Rata-rata finish position driver di N race sebelumnya (DNF = 20).
    Round tanpa history → impute rata-rata grid season itu.
    Return {round: {driver_code: avg_finish_last_N}}.
    """
    out: dict[int, dict[str, float]] = {}
    if not race_results:
        return out
    rounds = sorted(race_results.keys())

    # Season-average grid untuk impute round-round awal
    grids: list[float] = []
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

    history: dict[str, list[int]] = {}  # driver_code -> finishes (chronological)

    for R in rounds:
        df_R = race_results[R]
        codes = (
            [c for c in df_R["driverCode"].dropna().unique() if isinstance(c, str)]
            if (df_R is not None and "driverCode" in df_R.columns) else []
        )
        rd: dict[str, float] = {}
        for c in codes:
            hist = history.get(c, [])
            if not hist:
                rd[c] = float(season_avg_grid)
            else:
                last = hist[-n_races:]
                rd[c] = float(sum(last) / len(last))
        out[R] = rd
        # Update history dengan hasil round R
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


def build_circuit_dnf_rate(
    all_race_res: dict[int, dict[int, pd.DataFrame]],
) -> dict[str, float]:
    """
    DNF rate per circuit dari semua tahun. circuit name = F1_ROUNDS[year][round].
    Return {circuit_name: dnf_rate} (0..1).
    """
    counts: dict[str, list[int]] = {}  # circuit -> [dnf, total]
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


def _enrich_features(
    df: pd.DataFrame,
    year: int,
    round_num: int,
    enrichment: dict,
    circuit_dnf: dict[str, float],
) -> pd.DataFrame:
    """
    Merge fitur Ergast-derived ke DataFrame per driver (in-place pada copy).
    Semua lookup pakai default kalau data hilang → tidak pernah NaN/crash.
    """
    enrichment = enrichment or _empty_enrichment()
    circuit = F1_ROUNDS.get(year, {}).get(round_num, "")
    qg_round   = enrichment.get("quali_gap", {}).get(round_num, {})
    st_round   = enrichment.get("standings", {}).get(round_num, {})
    cs_round   = enrichment.get("constructor", {}).get(round_num, {})
    form_round = enrichment.get("form", {}).get(round_num, {})
    dnf_rate   = circuit_dnf.get(circuit, _DEF_CIRCUIT_DNF) if circuit_dnf else _DEF_CIRCUIT_DNF

    for idx in df.index:
        code = df.at[idx, "driver_code"]
        team = df.at[idx, "team_name"]

        df.at[idx, "quali_gap_pct"] = float(qg_round.get(code, _DEF_QUALI_GAP))

        ds = st_round.get(code, {})
        df.at[idx, "driver_pts_before"] = float(ds.get("driver_pts_before", _DEF_DRIVER_PTS))
        df.at[idx, "driver_pos_before"] = float(ds.get("driver_pos_before", _DEF_DRIVER_POS))

        cs = cs_round.get(team, {})
        df.at[idx, "constructor_pts_before"] = float(
            cs.get("constructor_pts_before", _DEF_CONSTRUCTOR_PTS))
        df.at[idx, "constructor_pos_before"] = float(
            cs.get("constructor_pos_before", _DEF_CONSTRUCTOR_POS))

        df.at[idx, "driver_form_last3"] = float(form_round.get(code, _DEF_FORM))
        df.at[idx, "circuit_dnf_rate"]  = float(dnf_rate)

    return df


def _build_enrichment_for_year(year: int) -> dict:
    """Pre-load + build semua enrichment dicts untuk satu season. Guarded."""
    try:
        race_res  = get_season_races(year)
        quali_res = get_season_qualifying(year)
    except Exception:
        return _empty_enrichment()
    if not race_res:
        return _empty_enrichment()
    return {
        "standings":   build_standings_before_round(race_res, year),
        "constructor": build_constructor_standings_before_round(race_res),
        "form":        build_rolling_form(race_res),
        "quali_gap":   build_quali_gap(quali_res) if quali_res else {},
    }


def _enrich_single_race(df: pd.DataFrame, year: int, round_num: int) -> pd.DataFrame:
    """
    Enrich satu race df (dipakai di prediction path). Membangun enrichment
    untuk `year` + circuit_dnf dari seluruh SUPPORTED_YEARS supaya identik
    dengan rate yang dipakai saat training.
    """
    enrichment = _build_enrichment_for_year(year)
    try:
        all_res = {yr: get_season_races(yr) for yr in SUPPORTED_YEARS}
    except Exception:
        all_res = {}
    circuit_dnf = build_circuit_dnf_rate(all_res) if all_res else {}
    return _enrich_features(df, year, round_num, enrichment, circuit_dnf)


# ── Training dataset ───────────────────────────────────────────────────────

def build_training_dataset(
    years: list[int],
    rounds_per_year: dict[int, dict],
    progress_cb=None,
) -> tuple[pd.DataFrame | None, pd.Series | None, list[str]]:
    """
    Iterate all rounds across years, assemble + enrich features, return
    (X, y, feature_cols). X diurutkan kronologis (year, round) supaya
    train_race_model bisa pakai time-based split.
    """
    try:
        import fastf1
    except ImportError:
        return None, None, []

    # Pre-load enrichment per year (SEKALI, bukan per-race). season_loader
    # functions cached, jadi panggilan berikutnya di prediction path murah.
    enrichment_by_year: dict[int, dict] = {}
    for yr in years:
        enrichment_by_year[yr] = _build_enrichment_for_year(yr)

    # Circuit DNF rate dari semua tahun sekaligus
    try:
        all_race_res = {yr: get_season_races(yr) for yr in years}
    except Exception:
        all_race_res = {}
    circuit_dnf = build_circuit_dnf_rate(all_race_res) if all_race_res else {}

    all_rounds = []
    for yr in years:
        rds = rounds_per_year.get(yr, {})
        for rn in rds:
            all_rounds.append((yr, rn))

    total = len(all_rounds)
    frames = []

    for idx, (yr, rn) in enumerate(all_rounds):
        circuit = rounds_per_year.get(yr, {}).get(rn, f"Round {rn}")
        if progress_cb:
            progress_cb(idx, total, f"{yr} {circuit}")

        try:
            sess = fastf1.get_session(yr, rn, "R")
            sess.load(laps=False, telemetry=False, weather=True, messages=False)
        except Exception:
            continue

        df = assemble_race_features(yr, rn, sess)
        if df is not None and len(df) > 0:
            df = _enrich_features(
                df, yr, rn, enrichment_by_year.get(yr, _empty_enrichment()), circuit_dnf,
            )
            frames.append(df)

    if progress_cb:
        progress_cb(total, total, "Done")

    if not frames:
        return None, None, []

    combined = pd.concat(frames, ignore_index=True)

    # Urutkan kronologis untuk time-based split (year ASC, round ASC)
    combined["_sort_key"] = combined["year"] * 100 + combined["round_num"]
    combined = combined.sort_values("_sort_key").reset_index(drop=True)

    meta_cols = {"driver_code", "team_name", "year", "round_num",
                 "finish_pos", "status_raw", "outcome_class", "_sort_key"}
    feature_cols = [c for c in combined.columns if c not in meta_cols]

    X = combined[feature_cols].astype(float)
    y = combined["outcome_class"].astype(int)
    return X, y, feature_cols


# ── Model training ─────────────────────────────────────────────────────────

def train_race_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
) -> dict | None:
    """
    Train XGBClassifier (4 outcome classes). Return model bundle dict.

    Time-based split: X diasumsikan SUDAH terurut kronologis (dari
    build_training_dataset). Train = 80% race terawal, test = 20% terakhir.
    Ini menghindari leakage temporal (random split membiarkan race masa depan
    "mengajari" model tentang race masa lalu) dan memberi evaluasi lebih jujur.
    Tidak pakai stratify (tidak kompatibel dengan time-based split).
    """
    if not _ML_OK or X is None or y is None or len(X) < 20:
        return None

    try:
        X_ord = X[feature_cols].reset_index(drop=True)
        y_ord = y.reset_index(drop=True)
        split_point = int(len(X_ord) * 0.80)
        if split_point < 10 or split_point >= len(X_ord):
            return None
        X_train, X_test = X_ord.iloc[:split_point], X_ord.iloc[split_point:]
        y_train, y_test = y_ord.iloc[:split_point], y_ord.iloc[split_point:]
    except Exception:
        return None

    weights = compute_sample_weight("balanced", y_train)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    try:
        model.fit(X_train, y_train, sample_weight=weights)
        y_pred = model.predict(X_test)
    except Exception:
        return None

    class_names = ["DNF", "Podium", "Points", "Outside Points"]
    return {
        "model": model,
        "feature_cols": feature_cols,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]),
        "class_names": class_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "X_train": X_train,
        "y_train": y_train,
    }


# ── Prediction ─────────────────────────────────────────────────────────────

def predict_race_outcome(
    model_bundle: dict,
    session,
    year: int,
    round_num: int,
) -> pd.DataFrame | None:
    """Predict outcome probabilities for every driver in a race session."""
    df = assemble_race_features(year, round_num, session)
    if df is None or len(df) == 0:
        return None

    # Enrich dengan fitur Ergast yang sama seperti saat training, supaya
    # kolom prediksi cocok dengan feature_cols (kalau tidak, reindex fill_value=0
    # akan memberi nilai palsu dan prediksi jadi ngawur).
    df = _enrich_single_race(df, year, round_num)

    model = model_bundle["model"]
    fcols = model_bundle["feature_cols"]

    X_pred = df.reindex(columns=fcols, fill_value=0).astype(float)

    try:
        proba = model.predict_proba(X_pred)
    except Exception:
        return None

    class_names = model_bundle["class_names"]
    out = pd.DataFrame({
        "driver_code": df["driver_code"].values,
        "team_name": df["team_name"].values,
        "grid_position": df["grid_position"].values,
        "prob_dnf": proba[:, 0],
        "prob_podium": proba[:, 1],
        "prob_points": proba[:, 2],
        "prob_outside": proba[:, 3],
        "predicted_class": proba.argmax(axis=1),
    })
    out["predicted_label"] = out["predicted_class"].map(
        {i: n for i, n in enumerate(class_names)}
    )
    return out.sort_values("prob_podium", ascending=False).reset_index(drop=True)


def build_prediction_features(
    model_bundle: dict,
    session,
    year: int,
    round_num: int,
) -> pd.DataFrame | None:
    """
    Aligned per-race feature matrix untuk SHAP, di-index by driver_code.

    Ini feature matrix yang SAMA dengan yang dipakai predict_race_outcome,
    sehingga SHAP yang dihitung di atasnya benar-benar menjelaskan prediksi
    driver di race ini (bukan baris acak dari training data). Row order =
    urutan session.results; index = driver_code untuk lookup per-driver.
    """
    if model_bundle is None:
        return None
    df = assemble_race_features(year, round_num, session)
    if df is None or len(df) == 0:
        return None
    # Enrich identik dengan predict_race_outcome → SHAP menjelaskan prediksi nyata.
    df = _enrich_single_race(df, year, round_num)
    fcols = model_bundle["feature_cols"]
    X = df.reindex(columns=fcols, fill_value=0).astype(float)
    X.index = df["driver_code"].values
    return X


# ── SHAP ───────────────────────────────────────────────────────────────────

def compute_race_shap(
    model_bundle: dict,
    X: pd.DataFrame,
) -> tuple | None:
    """
    Return (shap_values, expected_value, feature_names).
    shap_values shape: [n_samples, n_classes, n_features] or list of arrays.
    Falls back to feature_importances_ if SHAP is unavailable.
    """
    if model_bundle is None or X is None or len(X) == 0:
        return None

    model = model_bundle["model"]
    fcols = model_bundle["feature_cols"]
    X_aligned = X.reindex(columns=fcols, fill_value=0).astype(float)
    sample = X_aligned.sample(min(200, len(X_aligned)), random_state=42) if len(X_aligned) > 200 else X_aligned

    if _SHAP_OK:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(sample)
            # Normalisasi SELALU ke [n_samples, n_classes, n_features].
            # Multi-class XGBoost punya 2 format tergantung versi SHAP:
            #   SHAP < 0.40 : list of n_classes arrays, masing-masing [n, n_features]
            #   SHAP >= 0.40: ndarray [n, n_features, n_classes]
            if isinstance(sv, list):
                sv = np.stack(sv, axis=1)             # → [n, n_classes, n_features]
            else:
                sv = np.asarray(sv)
                if sv.ndim == 3:
                    sv = np.transpose(sv, (0, 2, 1))  # → [n, n_classes, n_features]
                elif sv.ndim == 2:
                    sv = sv[:, np.newaxis, :]          # binary/regresi → 1 class axis
            ev = explainer.expected_value
            if not hasattr(ev, "__len__"):
                ev = [ev]
            return sv, np.array(ev), fcols
        except Exception:
            pass

    try:
        importances = model.feature_importances_
        return importances, None, fcols
    except Exception:
        return None


# ── Persistence ────────────────────────────────────────────────────────────

def save_model_bundle(bundle: dict, path: str) -> bool:
    """Save model bundle to disk via joblib. Returns True on success."""
    try:
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(bundle, path)
        return True
    except Exception:
        return False


def load_model_bundle(path: str) -> dict | None:
    """Load a pre-trained model bundle from disk. Returns None if file missing."""
    try:
        import joblib
        if not os.path.exists(path):
            return None
        return joblib.load(path)
    except Exception:
        return None
