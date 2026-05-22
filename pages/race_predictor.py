"""Race Outcome Predictor page."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import SUPPORTED_YEARS, F1_ROUNDS

MODEL_PATH = "models/race_model.joblib"

try:
    from src.ml.race_model import (
        _ML_OK,
        CIRCUIT_META,
        build_training_dataset,
        train_race_model,
        predict_race_outcome,
        build_prediction_features,
        compute_race_shap,
        load_model_bundle,
    )
except ImportError:
    _ML_OK = False

try:
    from src.viz.predictor_charts import (
        build_outcome_probability_bar,
        build_win_probability_bar,
        build_shap_importance_bar,
        build_shap_waterfall,
        build_confusion_matrix_heatmap,
    )
except ImportError:
    pass

CLASS_NAMES = ["DNF", "Podium", "Points", "Outside Points"]
CLASS_COLORS = {
    "DNF": "#888888",
    "Podium": "#E8002D",
    "Points": "#FF8000",
    "Outside Points": "#444444",
}


@st.cache_resource(show_spinner=False)
def _load_or_train_model(years_key: str):
    """
    Try to load a pre-trained bundle from MODEL_PATH first.
    Falls back to training from FastF1 data if no file found.
    years_key used only as cache key for the fallback path.
    """
    bundle = load_model_bundle(MODEL_PATH)
    if bundle is not None:
        return bundle, True  # (bundle, is_pretrained)

    with st.spinner("Training race prediction model (first run — takes a few minutes) …"):
        years = [int(y) for y in years_key.split("-")]
        X, y, fcols = build_training_dataset(years, F1_ROUNDS)
        if X is None or y is None or len(X) == 0:
            return None, False
        bundle = train_race_model(X, y, fcols)
        return bundle, False


def render():
    st.title("Race Outcome Predictor")

    if not _ML_OK:
        st.error(
            "XGBoost not installed. Run: `pip install xgboost scikit-learn`"
        )
        return

    # ── Sidebar controls ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="section-label">Predictor Settings</div>',
            unsafe_allow_html=True,
        )

        _year_opts = sorted(SUPPORTED_YEARS, reverse=True)
        _cur_year = st.session_state.get("selected_year", _year_opts[0])
        year = st.selectbox(
            "Year", _year_opts,
            index=_year_opts.index(_cur_year) if _cur_year in _year_opts else 0,
            key="rp_year",
        )
        rounds = F1_ROUNDS.get(year, {})
        round_options = list(rounds.items())
        round_labels = [f"R{num} — {name}" for num, name in round_options]
        round_idx = st.selectbox(
            "Round", range(len(round_labels)),
            format_func=lambda i: round_labels[i],
            key="rp_round",
        )
        selected_round_num, selected_circuit = round_options[round_idx]

        run = st.button("RUN PREDICTION", key="rp_run")

    # ── Model loading / training (cached) ─────────────────────────────
    years_key = "-".join(str(y) for y in SUPPORTED_YEARS)
    result = _load_or_train_model(years_key)
    if result is None or result[0] is None:
        st.warning("Could not load or train model. Check that FastF1 data is accessible.")
        return
    model_bundle, is_pretrained = result

    if is_pretrained:
        st.success("Pre-trained model loaded from `models/race_model.joblib`.", icon="✅")
    else:
        st.info("Model trained on-the-fly from FastF1 data. Upload `models/race_model.joblib` from Colab for better results.", icon="ℹ️")

    # ── Info row ───────────────────────────────────────────────────────
    meta = CIRCUIT_META.get(selected_circuit, {"overtake_idx": 5, "track_type": 1})

    col_metrics, col_circuit = st.columns([3, 2])
    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Training rows", f"{model_bundle['n_train'] + model_bundle['n_test']} driver-races")
        m2.metric("Accuracy", f"{model_bundle['accuracy']:.1%}")
        report_lines = model_bundle["report"].strip().split("\n")
        f1_line = [l for l in report_lines if "weighted avg" in l]
        f1_val = f1_line[0].split()[-2] if f1_line else "—"
        m3.metric("Weighted F1", f1_val)

    with col_circuit:
        track_label = "Street Circuit" if meta["track_type"] == 0 else "Permanent Circuit"
        oi = meta["overtake_idx"]
        bar_filled = int(oi)
        bar_html = "".join(
            f'<span style="display:inline-block;width:16px;height:10px;'
            f'background:{("#E8002D" if i < bar_filled else "#2A2A2A")}; '
            f'margin-right:2px;border-radius:1px;"></span>'
            for i in range(10)
        )
        st.markdown(
            f"**{selected_circuit}** — {track_label}<br>"
            f"Overtake Index: {bar_html} **{oi}/10**",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Prediction ─────────────────────────────────────────────────────
    proba_df = None
    pred_X = None
    if run or "rp_proba_df" in st.session_state:
        if run:
            try:
                import fastf1
                sess = fastf1.get_session(year, selected_round_num, "R")
                sess.load(laps=False, telemetry=False, weather=True, messages=False)
                proba_df = predict_race_outcome(
                    model_bundle, sess, year, selected_round_num,
                )
                # Feature matrix race ini (index = driver_code) untuk SHAP per-driver
                pred_X = build_prediction_features(
                    model_bundle, sess, year, selected_round_num,
                )
                st.session_state["rp_proba_df"] = proba_df
                st.session_state["rp_pred_X"] = pred_X
                st.session_state["rp_last_round"] = (year, selected_round_num)
            except Exception as exc:
                st.error(f"Could not load race session: {exc}")
                return
        else:
            last = st.session_state.get("rp_last_round")
            if last == (year, selected_round_num):
                proba_df = st.session_state.get("rp_proba_df")
                pred_X = st.session_state.get("rp_pred_X")

    if proba_df is None or len(proba_df) == 0:
        st.info("Select a round and click **RUN PREDICTION** to see results.")
        return

    st.caption(
        "⚠️ Disclaimer: model dilatih pada seluruh round 2022-2024, termasuk "
        "race yang sedang diprediksi (in-sample). Akurasi & probabilitas di sini "
        "cenderung optimistis — bukan prediksi out-of-sample murni. Untuk evaluasi "
        "realistis, perlu hold-out race yang tidak masuk training set."
    )

    # ── Tabs ───────────────────────────────────────────────────────────
    tab_outcome, tab_win, tab_shap, tab_eval = st.tabs([
        "Outcome Probabilities",
        "Win Probability",
        "SHAP Analysis",
        "Model Evaluation",
    ])

    with tab_outcome:
        fig = build_outcome_probability_bar(proba_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="rp_chart_outcome")
        else:
            st.warning("No data to display.")

    with tab_win:
        fig = build_win_probability_bar(proba_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="rp_chart_win")
        else:
            st.warning("No data to display.")

    with tab_shap:
        # SHAP dihitung pada feature matrix RACE INI (bukan training data),
        # sehingga importance & waterfall benar-benar menjelaskan prediksi
        # driver di race yang dipilih. shap_vals row order = pred_X.index order.
        shap_result = compute_race_shap(model_bundle, pred_X)

        if shap_result is None or pred_X is None:
            st.warning("SHAP analysis unavailable.")
        elif isinstance(shap_result[0], np.ndarray) and shap_result[0].ndim >= 2:
            shap_vals, expected_vals, feat_names = shap_result
            shap_driver_order = list(pred_X.index)  # baris i ↔ driver ini

            st.subheader("Global Feature Importance")
            st.caption(
                f"Mean |SHAP| across {len(shap_driver_order)} drivers in "
                f"{selected_circuit} {year}, averaged over all 4 outcome classes."
            )
            fig_imp = build_shap_importance_bar(shap_vals, feat_names)
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True, key="rp_chart_shap_global")

            st.subheader("Driver Deep Dive")
            # Selectbox urut by podium prob (UX), lookup posisi di shap_vals via driver_code
            driver_list = proba_df["driver_code"].tolist()
            sel_driver = st.selectbox(
                "Driver", driver_list, key="rp_shap_driver",
            )
            if sel_driver in shap_driver_order:
                safe_idx = shap_driver_order.index(sel_driver)
            else:
                safe_idx = 0
            safe_idx = min(safe_idx, shap_vals.shape[0] - 1)
            ev = float(expected_vals[1]) if len(expected_vals) > 1 else float(expected_vals[0])

            st.caption(
                f"Why **{sel_driver}** is predicted toward the **Podium** class — "
                "green pushes probability up, red pushes it down."
            )
            fig_wf = build_shap_waterfall(
                shap_vals, ev, feat_names,
                driver_idx=safe_idx, class_idx=1,
            )
            if fig_wf:
                st.plotly_chart(fig_wf, use_container_width=True, key="rp_chart_shap_wf")
        else:
            importances, _, feat_names = shap_result
            st.subheader("Feature Importance (XGBoost built-in)")
            st.caption("SHAP unavailable — showing XGBoost split-gain importance.")
            fig_imp = build_shap_importance_bar(importances, feat_names)
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True, key="rp_chart_imp_fallback")

    with tab_eval:
        st.subheader("Confusion Matrix")
        fig_cm = build_confusion_matrix_heatmap(
            model_bundle["confusion_matrix"], model_bundle["class_names"],
        )
        if fig_cm:
            st.plotly_chart(fig_cm, use_container_width=True, key="rp_chart_cm")

        st.subheader("Classification Report")
        report_text = model_bundle["report"]
        lines = report_text.strip().split("\n")
        header = lines[0].split()
        data_rows = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 5:
                label = " ".join(parts[:-4])
                nums = parts[-4:]
                data_rows.append([label] + nums)
        if data_rows:
            report_df = pd.DataFrame(
                data_rows, columns=["Class"] + header[-4:],
            )
            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True,
                key="rp_df_report",
            )
        else:
            st.code(report_text)
