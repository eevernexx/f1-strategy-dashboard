"""
Tyre degradation regression + pit stop optimizer.

Pipeline:
1. prepare_training_data(laps_df) → feature matrix X, target y (lap time)
2. train_pit_model(X, y) → fitted XGBoost regressor + metrics
3. predict_lap_time(model, ...) → expected lap time given state
4. optimize_pit_window(model, current_state, total_laps) → best pit lap
5. compute_shap_values(model, X) → SHAP per feature (optional, fallback ke
   XGBoost built-in feature_importances_ kalau shap unavailable)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Soft imports — ML deps mungkin gak available di semua env
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    _ML_OK = True
except ImportError:
    _ML_OK = False

try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False


# ── Feature engineering ─────────────────────────────────────────────────────

# Compound categories — fixed order untuk one-hot consistency
COMPOUND_LEVELS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def _compound_dummies(compound_series: pd.Series) -> pd.DataFrame:
    """One-hot encode compound dengan fixed kolom (Soft/Medium/Hard/Inter/Wet)."""
    s = compound_series.fillna("UNKNOWN").astype(str).str.upper()
    out = pd.DataFrame({
        f"Comp_{c}": (s == c).astype(int) for c in COMPOUND_LEVELS
    })
    return out


def prepare_training_data(
    laps_df: pd.DataFrame,
    weather_data: pd.DataFrame | None = None,
    total_laps: int | None = None,
) -> tuple[pd.DataFrame | None, pd.Series | None, list[str]]:
    """
    Return (X, y, feature_cols). None kalau data tidak cukup.

    Features:
    - TyreLife (laps on current tyre)
    - Stint (stint number 1, 2, 3, ...)
    - FuelLoad (proxy: 1 - lap/total_laps; berkurang linear sepanjang race)
    - TrackTempAvg, AirTempAvg (dari weather_data atau default)
    - Comp_SOFT, Comp_MEDIUM, Comp_HARD, Comp_INTERMEDIATE, Comp_WET (one-hot)
    """
    if laps_df is None or len(laps_df) == 0:
        return None, None, []

    needed = ["LapTimeSeconds", "TyreLife", "Compound", "Stint", "LapNumber"]
    if not all(c in laps_df.columns for c in needed):
        return None, None, []

    df = laps_df.dropna(subset=needed).copy()
    if len(df) < 20:
        return None, None, []

    # Fuel load proxy: decreases linearly with lap number
    if total_laps is None or total_laps <= 0:
        total_laps = max(int(df["LapNumber"].max()), 1)
    df["FuelLoad"] = 1.0 - (df["LapNumber"].astype(float) / float(total_laps))
    df["FuelLoad"] = df["FuelLoad"].clip(lower=0.0, upper=1.0)

    # Track + air temp dari weather (session average kalau ada)
    track_temp = 30.0
    air_temp = 25.0
    if weather_data is not None and len(weather_data) > 0:
        if "TrackTemp" in weather_data.columns and weather_data["TrackTemp"].notna().any():
            track_temp = float(weather_data["TrackTemp"].mean())
        if "AirTemp" in weather_data.columns and weather_data["AirTemp"].notna().any():
            air_temp = float(weather_data["AirTemp"].mean())
    df["TrackTempAvg"] = track_temp
    df["AirTempAvg"] = air_temp

    # One-hot compound
    compound_dum = _compound_dummies(df["Compound"])
    df = pd.concat([df.reset_index(drop=True), compound_dum.reset_index(drop=True)], axis=1)

    base_cols = ["TyreLife", "Stint", "FuelLoad", "TrackTempAvg", "AirTempAvg"]
    feature_cols = base_cols + [f"Comp_{c}" for c in COMPOUND_LEVELS]

    X = df[feature_cols].astype(float)
    y = df["LapTimeSeconds"].astype(float)

    return X, y, feature_cols


# ── Model training ──────────────────────────────────────────────────────────

def train_pit_model(X: pd.DataFrame, y: pd.Series) -> tuple | None:
    """
    Train XGBoost regressor. Return (model, metrics_dict) atau None kalau ML
    deps gak available atau data tidak cukup.
    """
    if not _ML_OK or X is None or y is None or len(X) < 20:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42,
        )
    except Exception:
        return None

    model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=1,           # Streamlit Cloud = 0.078 CPU, jangan oversubscribe
        verbosity=0,
    )
    try:
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
    except Exception:
        return None

    metrics = {
        "r2_test":   float(r2_score(y_test, y_pred_test)),
        "r2_train":  float(r2_score(y_train, y_pred_train)),
        "mae_test":  float(mean_absolute_error(y_test, y_pred_test)),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "y_test":    y_test.tolist(),
        "y_pred":    y_pred_test.tolist(),
    }
    return model, metrics


# ── Single-row feature builder (untuk prediction) ───────────────────────────

def build_state_features(
    tyre_life: float,
    stint: int,
    fuel_load: float,
    track_temp: float,
    air_temp: float,
    compound: str,
) -> pd.DataFrame:
    """Build 1-row DataFrame dengan kolom-feature yang match training data."""
    row = {
        "TyreLife":      float(tyre_life),
        "Stint":         int(stint),
        "FuelLoad":      float(fuel_load),
        "TrackTempAvg":  float(track_temp),
        "AirTempAvg":    float(air_temp),
    }
    compound_upper = (compound or "UNKNOWN").upper()
    for c in COMPOUND_LEVELS:
        row[f"Comp_{c}"] = 1 if c == compound_upper else 0
    return pd.DataFrame([row])


# ── Pit window optimizer ────────────────────────────────────────────────────

def optimize_pit_window(
    model,
    current_lap: int,
    current_tyre_life: int,
    current_compound: str,
    current_stint: int,
    fresh_compound: str,
    total_laps: int,
    track_temp: float,
    air_temp: float,
    pit_loss_seconds: float = 22.0,
) -> dict | None:
    """
    Untuk tiap kandidat pit lap dari `current_lap+1` sampai `total_laps-2`,
    estimasi total waktu sisa race:
      time_stay = sum predicted lap time pakai current compound (tyre age makin tua)
      time_after = pit_loss + sum predicted lap time pakai fresh compound (mulai dari tyre age 1)
      total = time_stay + time_after

    Return dict dengan all_results (list) + optimal_pit_lap + best_total_time.
    Plus baseline "no pit" total time (lanjut current compound sampai akhir).
    """
    if model is None or current_lap >= total_laps:
        return None

    laps_remaining = list(range(current_lap + 1, total_laps + 1))
    if len(laps_remaining) < 2:
        return None

    # Pre-build features untuk setiap lap remaining dengan KEDUA skenario:
    # 1. Stay on current compound (TyreLife = current_tyre_life + offset)
    # 2. Fresh compound after pit (TyreLife = 1, 2, 3, ...)
    n_remaining = len(laps_remaining)

    def _features_stay(lap_offset: int) -> pd.DataFrame:
        """Feature di lap_offset (1-indexed from current_lap+1) untuk stay scenario."""
        tyre_life = current_tyre_life + lap_offset
        fuel = max(0.0, 1.0 - (current_lap + lap_offset) / float(total_laps))
        return build_state_features(
            tyre_life=tyre_life,
            stint=current_stint,
            fuel_load=fuel,
            track_temp=track_temp,
            air_temp=air_temp,
            compound=current_compound,
        )

    def _features_fresh(lap_offset_after_pit: int, lap_global: int) -> pd.DataFrame:
        """Feature di lap_offset_after_pit (1-indexed) setelah pit (fresh compound)."""
        tyre_life = lap_offset_after_pit
        fuel = max(0.0, 1.0 - lap_global / float(total_laps))
        return build_state_features(
            tyre_life=tyre_life,
            stint=current_stint + 1,   # new stint after pit
            fuel_load=fuel,
            track_temp=track_temp,
            air_temp=air_temp,
            compound=fresh_compound,
        )

    # Predict lap times for all stay scenarios (current compound, increasing age)
    stay_feats = pd.concat(
        [_features_stay(i) for i in range(1, n_remaining + 1)],
        ignore_index=True,
    )
    try:
        stay_times = model.predict(stay_feats)
    except Exception:
        return None

    # Predict lap times for all fresh scenarios (fresh compound from pit lap)
    # Untuk efisiensi, predict semua possible (pit_lap, lap_after_pit) once.
    # Tapi karena fresh tyre time hanya tergantung age + fuel + lap_global, kita
    # bisa predict semua (lap_global, tyre_age) pairs lalu pickup yang relevan.

    # Sederhana: untuk tiap kandidat pit_lap, build features post-pit, predict
    candidates = []
    for pit_idx, pit_lap in enumerate(laps_remaining[:-1]):
        # Time before pit: stay times for laps current_lap+1 to pit_lap (inclusive)
        time_stay = float(stay_times[:pit_idx + 1].sum())

        # Time after pit: fresh tyres from pit_lap+1 to total_laps
        post_laps = list(range(pit_lap + 1, total_laps + 1))
        if not post_laps:
            continue
        fresh_feats = pd.concat(
            [_features_fresh(i + 1, post_laps[i]) for i in range(len(post_laps))],
            ignore_index=True,
        )
        try:
            fresh_times = model.predict(fresh_feats)
        except Exception:
            continue
        time_after = float(fresh_times.sum()) + pit_loss_seconds

        total = time_stay + time_after
        candidates.append({
            "pit_lap":    int(pit_lap),
            "time_stay":  time_stay,
            "time_after": time_after,
            "total_time": total,
        })

    if not candidates:
        return None

    # Baseline: no pit (stay full race)
    baseline_total = float(stay_times.sum())

    optimal = min(candidates, key=lambda c: c["total_time"])
    return {
        "candidates":      candidates,
        "optimal_pit_lap": optimal["pit_lap"],
        "best_total":      optimal["total_time"],
        "baseline_no_pit": baseline_total,
        "saving":          baseline_total - optimal["total_time"],
        "pit_loss_used":   pit_loss_seconds,
    }


# ── SHAP / feature importance ───────────────────────────────────────────────

def compute_feature_importance(model, X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Return DataFrame: Feature, Importance.
    Prefer SHAP mean absolute value (lebih reliable), fallback ke XGBoost
    built-in feature_importances_.
    """
    if model is None:
        return pd.DataFrame(columns=["Feature", "Importance", "Method"])

    # Try SHAP first
    if _SHAP_OK and X is not None and len(X) > 0:
        try:
            # Sample max 200 rows untuk speed (SHAP slow on large data)
            sample_X = X.sample(min(200, len(X)), random_state=42) if len(X) > 200 else X
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(sample_X)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            return pd.DataFrame({
                "Feature": feature_cols,
                "Importance": mean_abs,
                "Method": "SHAP",
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    # Fallback: XGBoost built-in
    try:
        importances = model.feature_importances_
        return pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances,
            "Method": "XGBoost gain",
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance", "Method"])
