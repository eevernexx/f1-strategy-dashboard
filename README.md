# F1 Race Strategy Intelligence Dashboard

Interactive Formula 1 analytics for the 2022–2024 seasons — telemetry breakdowns,
race strategy analysis, head-to-head driver comparisons, and an ML race-outcome
predictor. Built on official F1 timing data.

The project ships **two frontends** from one data pipeline:

| App | Stack | Best for | Live |
| --- | --- | --- | --- |
| **Season Dashboard** | Next.js · Tailwind · Framer Motion | Polished season overview, fully static, mobile-friendly | **[f1-strategy-dashboard-phi.vercel.app](https://f1-strategy-dashboard-phi.vercel.app/)** |
| **Advanced Visualization** | Streamlit · Plotly · FastF1 | Deep, interactive telemetry & strategy analysis | **[Open on Streamlit Cloud](https://f1-strategy-dashboard-pnq5i4cjdnqx4cqv7f53dv.streamlit.app/)** |

<p>
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Next.js 15" src="https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
</p>

---

## Features

### Season Dashboard (Next.js — `web/`)
A Finrise-style, fully static season overview: championship leader and cumulative
points, constructor points share, points standings (drivers/teams toggle),
per-driver per-round points heatmap, recent results feed, title contenders, and
reliability/momentum stats. Dark card layout with an F1-red accent and smooth
on-load animations. Deploys to Vercel with no backend.

### Advanced Visualization (Streamlit — `app.py`)
- **Telemetry Analyzer** — Speed / Throttle / Brake / Gear / RPM channels per lap
  per driver, with corner annotations, DRS zones, sector splits, a track-dominance
  map, and qualifying-vs-race comparison.
- **Race Overview** — Final classification, race narrative, Driver of the Day,
  fastest-lap badge, race-pace heatmap, position chart with SC/VSC/Rain overlays,
  year-on-year comparison, and teammate head-to-head.
- **Tyre Strategy** — Stint breakdown, compound usage, pit timeline, tyre
  degradation curves, stint pace evolution, and an undercut/overcut detector.
- **Driver Comparison** — Head-to-head between two drivers across three modes:
  single-race H2H, full-season H2H, and circuit-history H2H (2022–2024).
- **Race Predictor** — An ML model that predicts race outcome class
  (DNF / Podium / Points / Outside Points) and win probability, with SHAP
  explanations for each prediction.

## Architecture

```
                 FastF1 / Ergast (official F1 timing data)
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                         ▼
   Streamlit app (live)                      export_data.py (one-off)
   app.py · pages/ · src/                    writes static season JSON
   live telemetry & analysis                 web/public/data/<year>.json
            │                                         │
            ▼                                         ▼
   Streamlit Cloud                            Next.js app (web/) → Vercel
```

- The **Streamlit app** queries FastF1/Ergast live for deep, per-session analysis.
- The **Next.js app** is fully static: `export_data.py` exports real 2022–2024
  season data to JSON, which the app fetches client-side (no backend).

## Project structure

```
.
├── app.py                  # Streamlit entrypoint (nav + shared styling)
├── pages/                  # Streamlit views (telemetry, race overview, tyre, comparison, predictor)
├── src/                    # Pipeline, ML, viz, and util modules
│   ├── ml/                 # Race-outcome model (training, prediction, SHAP)
│   ├── pipeline/           # Data loading & feature engineering
│   ├── viz/                # Plotly chart builders
│   └── utils/              # Config (seasons, rounds) & helpers
├── export_data.py          # Exports season JSON for the Next.js app
├── models/                 # Trained model artifacts (.joblib)
├── notebooks/              # Model training / exploration
├── requirements.txt        # Python dependencies
└── web/                    # Next.js season dashboard (deployed to Vercel)
    ├── app/                # App Router pages
    ├── components/         # React components
    ├── lib/                # Data fetching & helpers
    └── public/data/        # Exported season JSON (2022–2024)
```

## Getting started

### Streamlit app (Python)

Requires Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

### Next.js app (web)

Requires Node.js 18+.

```bash
cd web
npm install
npm run dev                     # http://localhost:3000
```

### Refreshing the web app's data

The static JSON consumed by the Next.js app is generated by the Python exporter
(uses `fastf1.ergast`, HTTP only):

```bash
# from the repo root, with the project venv active
python export_data.py
```

This writes `web/public/data/<year>.json` and `web/public/data/index.json`.

## Deployment

- **Next.js → Vercel**: New Project → Import the repo, set **Root Directory** to
  `web`, framework preset **Next.js** (auto-detected). No environment variables
  required.
- **Streamlit → Streamlit Cloud**: Main file `app.py`, Python 3.11.

## Tech stack

- **FastF1 + Ergast** — official F1 data (telemetry, lap times, sessions, results)
- **Streamlit** — interactive analysis UI
- **Plotly** — charts
- **pandas · numpy · scipy** — data analysis
- **scikit-learn · XGBoost · SHAP** — race-outcome model and explanations
- **Next.js (App Router) · Tailwind CSS · Framer Motion** — static season dashboard

## License

Released under the [MIT License](LICENSE).
