"""
F1 Race Strategy Intelligence Dashboard
========================================
Main entrypoint — handles navigation and shared state.
Run: streamlit run app.py
"""

import base64
from pathlib import Path

import streamlit as st

from src.utils.config import F1_ROUNDS


def _load_logo_b64(filename: str) -> str:
    """Base64-encode an image next to this script so it can be embedded in HTML."""
    path = Path(__file__).parent / filename
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


F1_LOGO_B64 = _load_logo_b64("F1-Logo-PNG-Photo.png")


st.set_page_config(
    page_title="F1 Strategy Intelligence",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — F1 Racing Aesthetic ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@400;500;600&display=swap');

  /* ── Base ── */
  .stApp {
    background-color: #080808;
    background-image:
      repeating-linear-gradient(
        0deg,
        transparent,
        transparent 60px,
        rgba(255,255,255,0.012) 60px,
        rgba(255,255,255,0.012) 61px
      );
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #0D0D0D;
    border-right: 1px solid #1A1A1A;
  }
  [data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
  }

  /* ── Typography ── */
  h1 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.8rem !important;
    letter-spacing: -0.01em !important;
    text-transform: uppercase !important;
    color: #FFFFFF !important;
  }
  h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
    color: #FFFFFF !important;
  }
  p, div, span, label {
    font-family: 'Barlow', sans-serif !important;
  }

  /* ── Red accent line under title ── */
  h1::after {
    content: '';
    display: block;
    width: 48px;
    height: 3px;
    background: #E8002D;
    margin-top: 8px;
    border-radius: 2px;
  }

  /* ── Buttons ── */
  .stButton > button {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    border: none !important;
    background: #E8002D !important;
    color: #FFFFFF !important;
    transition: all 0.15s ease !important;
    padding: 10px 20px !important;
  }
  .stButton > button:hover {
    background: #FF1744 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(232,0,45,0.4) !important;
  }
  .stButton > button:active {
    transform: translateY(0) !important;
  }

  /* ── Selectboxes ── */
  [data-testid="stSelectbox"] > div > div {
    background: #111111 !important;
    border: 1px solid #222222 !important;
    border-radius: 4px !important;
    font-family: 'Barlow', sans-serif !important;
    color: #CCCCCC !important;
  }
  [data-testid="stSelectbox"] > div > div:hover {
    border-color: #E8002D !important;
  }

  /* ── Checkboxes ── */
  [data-testid="stCheckbox"] label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    color: #999 !important;
  }
  [data-testid="stCheckbox"] input:checked + div {
    background: #E8002D !important;
    border-color: #E8002D !important;
  }

  /* ── Radio buttons (nav) ── */
  [data-testid="stRadio"] label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 13px !important;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: #111111;
    border: 1px solid #1E1E1E;
    border-top: 2px solid #E8002D;
    border-radius: 0 0 4px 4px;
    padding: 16px;
  }

  /* ── Dividers ── */
  hr {
    border-color: #1A1A1A !important;
    margin: 12px 0 !important;
  }

  /* ── Progress bar ── */
  [data-testid="stProgress"] > div > div {
    background: #E8002D !important;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    border: 1px solid #1A1A1A !important;
    border-radius: 4px !important;
    background: #0D0D0D !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border: 1px solid #1A1A1A !important;
    border-radius: 4px !important;
  }

  /* ── Info/warning boxes ── */
  [data-testid="stAlert"] {
    border-radius: 4px !important;
    border-left: 3px solid #E8002D !important;
    background: #1A0A0A !important;
  }

  /* ── Spinner ── */
  [data-testid="stSpinner"] {
    color: #E8002D !important;
  }

  /* ── Hide Streamlit branding ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ── Hide Streamlit auto-generated multipage nav (app/telemetry/race overview list) ── */
  [data-testid="stSidebarNav"] { display: none !important; }

  /* ── Hide sidebar collapse button (the "keyboard_double_arrow_left" icon) ── */
  [data-testid="stSidebarCollapseButton"] { display: none !important; }
  [data-testid="stSidebarCollapsedControl"] { display: none !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: #0D0D0D; }
  ::-webkit-scrollbar-thumb { background: #E8002D; border-radius: 2px; }

  /* ── Section label style ── */
  .section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #E8002D;
    margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Tahun terpilih dibaca dari state — ditulis oleh selectbox di tiap page
    # (key="selected_year"). Sidebar render lebih dulu dari page; pertama-load
    # state belum ada → fallback 2024 (matching default index page).
    sidebar_year = st.session_state.get("selected_year", 2024)
    sidebar_rounds = len(F1_ROUNDS.get(sidebar_year, F1_ROUNDS[2024]))

    # Logo/Header
    logo_img = (
        f'<img src="data:image/png;base64,{F1_LOGO_B64}" '
        f'width="80" style="margin-bottom:10px;display:block" alt="F1 logo"/>'
        if F1_LOGO_B64 else ""
    )
    st.markdown(f"""
    <div style="padding: 20px 16px 16px; border-bottom: 1px solid #1A1A1A; margin-bottom: 16px;">
        {logo_img}
        <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 20px; font-weight: 800; color: #FFFFFF; text-transform: uppercase; letter-spacing: 0.02em; line-height: 1.15;">Data Visualisation<br>Intelligence</div>
        <div style="font-family: 'Barlow', sans-serif; font-size: 11px; color: #444; margin-top: 6px;">{sidebar_year} FIA Formula One World Championship</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "Telemetry Analyzer",
            "Race Overview",
            "Tyre Strategy",
            "Driver Comparison",
            "Race Predictor",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Season badge
    st.markdown(f"""
    <div style="padding: 0 4px;">
        <div style="
            background: #E8002D;
            color: white;
            font-family: 'Barlow Condensed', sans-serif;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 4px 10px;
            border-radius: 2px;
            display: inline-block;
            margin-bottom: 8px;
        ">Season {sidebar_year}</div>
        <div style="font-size: 11px; color: #333; font-family: 'Barlow', sans-serif;">
            {sidebar_rounds} rounds · 20 drivers<br>
            Data via FastF1 + Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Telemetry Analyzer":
    from pages import telemetry
    telemetry.render()

elif page == "Race Overview":
    from pages import race_overview
    race_overview.render()

elif page == "Tyre Strategy":
    from pages import tyre_strategy
    tyre_strategy.render()

elif page == "Driver Comparison":
    from pages import driver_comparison
    driver_comparison.render()

elif page == "Race Predictor":
    st.title("Race Predictor")
    st.info(
        "Coming soon — race outcome classification (podium/top-10/DNF) "
        "with SHAP explainability."
    )
