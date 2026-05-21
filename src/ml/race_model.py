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

import re

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
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

from src.utils.config import TEAM_COLORS, F1_ROUNDS


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
    "Monza":           {"overtake_idx": 9, "track_type": 1},
}

TEAM_NAMES = sorted(TEAM_COLORS.keys())

_TEAM_NORMALIZE = {
    "AlphaTauri":           "RB",
    "Scuderia AlphaTauri":  "RB",
    "Alpha Tauri":          "RB",
    "Alfa Romeo":           "Kick Sauber",
    "Alfa Romeo Racing":    "Kick Sauber",
    "Alpine F1 Team":       "Alpine",
    "Aston Martin Aramco Mercedes":  "Aston Martin",
    "Aston Martin Aramco Cognizant": "Aston Martin",
}

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
        team = _TEAM_NORMALIZE.get(team_raw, team_raw)
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


# ── Training dataset ───────────────────────────────────────────────────────

def build_training_dataset(
    years: list[int],
    rounds_per_year: dict[int, dict],
    progress_cb=None,
) -> tuple[pd.DataFrame | None, pd.Series | None, list[str]]:
    """
    Iterate all rounds across years, assemble features, return (X, y, feature_cols).
    """
    try:
        import fastf1
    except ImportError:
        return None, None, []

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
            frames.append(df)

    if progress_cb:
        progress_cb(total, total, "Done")

    if not frames:
        return None, None, []

    combined = pd.concat(frames, ignore_index=True)
    meta_cols = {"driver_code", "team_name", "year", "round_num",
                 "finish_pos", "status_raw", "outcome_class"}
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
    """Train XGBClassifier (4 outcome classes). Return model bundle dict."""
    if not _ML_OK or X is None or y is None or len(X) < 20:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X[feature_cols], y, test_size=0.20, random_state=42, stratify=y,
        )
    except Exception:
        return None

    weights = compute_sample_weight("balanced", y_train)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
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
            if isinstance(sv, list):
                sv = np.stack(sv, axis=1)
            elif sv.ndim == 2:
                sv = sv[:, np.newaxis, :]
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
