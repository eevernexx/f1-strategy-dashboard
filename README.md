# F1 Race Strategy Intelligence Dashboard

Interactive F1 data visualization dashboard built with Streamlit + FastF1 + Plotly.
Covers 2022, 2023, 2024 seasons.

## Features

- **Telemetry Analyzer** — Speed/Throttle/Brake/Gear/RPM channels per lap per driver,
  with corner annotations, DRS zones, sector splits, track dominance map, and
  Q-vs-R comparison
- **Race Overview** — Final classification, race summary narrative,
  Driver of the Day, fastest lap badge, race pace heatmap, position chart with
  SC/VSC/Rain overlays, year-on-year comparison, teammate H2H, and more
- **Tyre Strategy** — Stint breakdown, compound usage, pit timeline,
  tyre degradation curves, stint pace evolution, undercut/overcut detector
- **Driver Comparison** (placeholder)
- **Race Predictor** (placeholder)

## Run locally

```bash
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Cloud](https://share.streamlit.io). Main file: `app.py`.

## Tech stack

- **FastF1** — official F1 data source (telemetry, lap times, sessions)
- **Streamlit** — interactive web UI
- **Plotly** — charts
- **pandas / numpy / scipy** — data analysis
