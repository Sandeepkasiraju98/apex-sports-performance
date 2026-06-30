import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import time
import os
from collections import deque
from datetime import datetime

from src.pose.extractor          import PoseExtractor
from src.pose.multi_tracker      import MultiAthleteTracker
from src.features.engineer       import RunningFeatureEngineer
from src.features.stride_counter import StrideCounter
from src.features.injury_risk    import InjuryRiskScorer
from src.models.lstm_model       import PerformanceAnalyzer
from src.models.explainability   import GradientExplainer
from src.utils.session_manager   import SessionManager
from src.utils.voice_coach       import VoiceCoach
from src.utils.webcam_handler    import WebcamProcessor
from src.coaching.corrective_cues   import CorrectiveCueEngine, CueSession, Severity
from src.features.gait_phase        import GaitPhaseAnalyzer
from src.utils.tracking_confidence  import TrackingConfidenceMonitor, TrackLevel
from src.utils.fps_normalizer       import FPSNormalizer, LiveFPSEstimator

# ── Page config ──
st.set_page_config(
    page_title="APEX — Sports Performance AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
# GLOBAL CSS — Cinematic sports-tech dark theme
# Fonts: Bebas Neue (display) + JetBrains Mono (data)
# Palette: deep black + electric cyan + hot coral
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=JetBrains+Mono:wght@300;400;500;700&family=Barlow:wght@300;400;500;600;700;900&display=swap');

:root {
  --black:    #030508;
  --surface:  #080d14;
  --card:     #0c1220;
  --card2:    #101828;
  --border:   rgba(0,210,255,0.12);
  --cyan:     #00d2ff;
  --cyan2:    #00a8cc;
  --coral:    #ff4757;
  --amber:    #ffa502;
  --green:    #2ed573;
  --purple:   #a55eea;
  --text:     #e8f4f8;
  --muted:    #3d5a6e;
  --mono:     'JetBrains Mono', monospace;
  --display:  'Bebas Neue', sans-serif;
  --body:     'Barlow', sans-serif;
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--black) !important;
  color: var(--text);
  font-family: var(--body);
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}

/* Hide default Streamlit branding (NOT the header — it holds the
   sidebar collapse/expand arrow, so hiding it strands a collapsed
   sidebar with no way to reopen it). */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Force sidebar visible & open across all Streamlit versions ── */
/* Streamlit 1.58 collapses the sidebar to zero width; pinning an
   explicit width + zeroing the transform/margin keeps it on-screen and
   open regardless of the collapse state. */
section[data-testid="stSidebar"],
[data-testid="stSidebar"] {
  transform: none !important;
  visibility: visible !important;
  min-width: 244px !important;
  width: 244px !important;
  margin-left: 0 !important;
  left: 0 !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
  transform: none !important;
  margin-left: 0 !important;
}
/* The collapse/expand arrow. Test id varies across versions, so target
   every spelling. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
  visibility: visible !important;
  display: flex !important;
  z-index: 999999 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: var(--black); }
::-webkit-scrollbar-thumb {
  background: var(--cyan2);
  border-radius: 2px;
}

/* ── Scanline overlay ── */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,210,255,0.012) 2px,
    rgba(0,210,255,0.012) 4px
  );
  pointer-events: none;
  z-index: 9990;
}

/* ── Grid background ── */
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,210,255,0.025) 1px,
                    transparent 1px),
    linear-gradient(90deg,
                    rgba(0,210,255,0.025) 1px,
                    transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

/* ── APEX header ── */
.apex-header {
  padding: 32px 0 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 32px;
  position: relative;
}
.apex-wordmark {
  font-family: var(--display);
  font-size: 72px;
  letter-spacing: 0.08em;
  line-height: 1;
  background: linear-gradient(135deg,
    var(--cyan) 0%, #ffffff 50%, var(--cyan2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: inline-block;
}
.apex-sub {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--cyan);
  letter-spacing: 0.25em;
  text-transform: uppercase;
  margin-top: 4px;
  opacity: 0.7;
}
.apex-status {
  position: absolute;
  top: 36px;
  right: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px var(--green);
  animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.3; }
}

/* ── BIG metric cards ── */
.big-metric {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 24px 20px 20px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s;
}
.big-metric::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent-color, var(--cyan));
}
.big-metric::after {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 60px; height: 60px;
  background: radial-gradient(
    circle at top right,
    rgba(var(--accent-rgb, 0,210,255), 0.08),
    transparent 70%
  );
}
.big-metric-label {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.big-metric-value {
  font-family: var(--display);
  font-size: 56px;
  line-height: 1;
  letter-spacing: 0.04em;
  color: var(--accent-color, var(--cyan));
  text-shadow: 0 0 20px rgba(var(--accent-rgb, 0,210,255), 0.3);
}
.big-metric-unit {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--muted);
  margin-left: 4px;
}
.big-metric-sub {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  margin-top: 6px;
  letter-spacing: 0.05em;
}

/* ── Section headers ── */
.section-head {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.section-head-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(
    90deg, var(--border), transparent
  );
}
.section-head-text {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--cyan);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  white-space: nowrap;
}

/* ── Alert / rec boxes ── */
.alert-box {
  background: rgba(255,71,87,0.07);
  border-left: 3px solid var(--coral);
  padding: 10px 14px;
  margin: 6px 0;
  font-family: var(--mono);
  font-size: 12px;
  color: #ff8a96;
  letter-spacing: 0.03em;
  border-radius: 0 4px 4px 0;
}
.rec-box {
  background: rgba(0,210,255,0.05);
  border-left: 3px solid var(--cyan2);
  padding: 10px 14px;
  margin: 6px 0;
  font-family: var(--mono);
  font-size: 12px;
  color: #7ab8c8;
  letter-spacing: 0.03em;
  border-radius: 0 4px 4px 0;
}

/* ── Coaching cue cards ── */
.cue-card {
  border-left: 3px solid var(--cyan);
  padding: 12px 16px;
  margin: 8px 0;
  border-radius: 0 4px 4px 0;
  background: rgba(0,210,255,0.04);
}
.cue-tag {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
}
.cue-fix {
  font-family: var(--body);
  font-size: 14px;
  color: var(--text);
  margin: 6px 0 4px;
  font-weight: 500;
}
.cue-drill {
  font-family: var(--mono);
  font-size: 11px;
  color: #7ab8c8;
}
.cue-why {
  font-family: var(--body);
  font-size: 11px;
  color: var(--muted);
  margin-top: 4px;
  font-style: italic;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 4px !important;
  transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--cyan) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  border-radius: 2px !important;
  padding: 10px 24px !important;
  transition: all 0.2s !important;
  position: relative;
  overflow: hidden;
}
.stButton > button:hover {
  background: rgba(0,210,255,0.08) !important;
  box-shadow: 0 0 16px rgba(0,210,255,0.2) !important;
  transform: translateY(-1px) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] .stRadio label {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  color: var(--muted) !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSlider {
  accent-color: var(--cyan);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  border: none !important;
  padding: 12px 20px !important;
  border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom: 2px solid var(--cyan) !important;
}

/* ── Metrics (st.metric) ── */
[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  padding: 16px !important;
  border-radius: 4px !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  color: var(--muted) !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--display) !important;
  font-size: 32px !important;
  color: var(--text) !important;
  letter-spacing: 0.05em !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}
[data-testid="stExpanderToggleIcon"] {
  color: var(--cyan) !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] input {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 13px !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] {
  accent-color: var(--cyan) !important;
}

/* ── Divider ── */
hr {
  border-color: var(--border) !important;
  margin: 24px 0 !important;
}

/* ── Caption / small text ── */
.stCaption, [data-testid="stCaption"] {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  color: var(--muted) !important;
  letter-spacing: 0.08em !important;
}

/* ── Info / success / error boxes ── */
[data-testid="stAlert"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  border-radius: 4px !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
  background: rgba(0,210,255,0.06) !important;
  border: 1px solid var(--cyan2) !important;
  color: var(--cyan) !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
}

/* ── Live badge ── */
.live-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(46,213,115,0.08);
  border: 1px solid rgba(46,213,115,0.25);
  padding: 5px 12px;
  border-radius: 2px;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--green);
  letter-spacing: 0.15em;
  text-transform: uppercase;
}

/* ── Phase badge ── */
.phase-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(0,210,255,0.06);
  border: 1px solid var(--border);
  padding: 4px 10px;
  border-radius: 2px;
  font-family: var(--mono);
  font-size: 9px;
  color: var(--cyan);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 24px;
}

/* ── XAI bar ── */
.xai-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
  border-bottom: 1px solid rgba(0,210,255,0.05);
}
.xai-label {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  width: 160px;
  flex-shrink: 0;
}
.xai-bar-track {
  flex: 1;
  height: 3px;
  background: rgba(255,255,255,0.04);
  border-radius: 2px;
  overflow: hidden;
}
.xai-bar-fill {
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg,
    var(--cyan2), var(--cyan));
}
.xai-val {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--cyan);
  width: 40px;
  text-align: right;
  flex-shrink: 0;
}

/* ── Sidebar logo area ── */
.sidebar-apex {
  font-family: var(--display);
  font-size: 36px;
  letter-spacing: 0.1em;
  background: linear-gradient(135deg,
    var(--cyan), #fff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 2px;
}
.sidebar-sub {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 20px;
}

/* ── Troubleshoot box ── */
.trouble-box {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 16px 20px;
  border-radius: 4px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
  line-height: 1.8;
}
.trouble-box b { color: var(--cyan); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PLOTLY THEME — matches the dark sports-tech look
# ══════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(8,13,20,0.8)',
    font=dict(family='JetBrains Mono, monospace',
              color='#3d5a6e', size=11),
    margin=dict(l=40, r=20, t=20, b=36),
    xaxis=dict(
        gridcolor='rgba(0,210,255,0.06)',
        linecolor='rgba(0,210,255,0.1)',
        tickfont=dict(size=10),
        title_font=dict(size=10)
    ),
    yaxis=dict(
        gridcolor='rgba(0,210,255,0.06)',
        linecolor='rgba(0,210,255,0.1)',
        tickfont=dict(size=10),
        title_font=dict(size=10)
    ),
    legend=dict(
        bgcolor='rgba(12,18,32,0.9)',
        bordercolor='rgba(0,210,255,0.15)',
        borderwidth=1,
        font=dict(size=10)
    )
)


# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════
def init_state():
    defaults = {
        'fatigue_history':  deque(maxlen=300),
        'perf_history':     deque(maxlen=300),
        'quality_history':  deque(maxlen=300),
        'cadence_history':  deque(maxlen=300),
        'injury_history':   deque(maxlen=300),
        'stride_history':   deque(maxlen=300),
        'timestamps':       deque(maxlen=300),
        'alerts':           [],
        'frame_count':      0,
        'models_loaded':    False,
        'analyzer':         None,
        'extractor':        None,
        'feature_eng':      None,
        'stride_counter':   None,
        'injury_scorer':    None,
        'session_mgr':      None,
        'explainer':        None,
        'multi_tracker':    None,
        'webcam_proc':      None,
        'webcam_running':   False,
        'last_importance':  None,
        # ── Upgrade: new engines + their rolling state ──
        'cue_engine':       None,
        'gait_analyzer':    None,
        'track_monitor':    None,
        'gait_history':     deque(maxlen=300),
        'last_cues':        [],
        'last_track':       None,
        'source_fps':       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def load_models(voice_enabled=False):
    if not st.session_state.models_loaded:
        mp = "models/sports_lstm.pth" \
            if os.path.exists("models/sports_lstm.pth") \
            else None
        st.session_state.analyzer      = PerformanceAnalyzer(mp)
        st.session_state.extractor     = PoseExtractor()
        st.session_state.feature_eng   = RunningFeatureEngineer()
        st.session_state.stride_counter = StrideCounter()
        st.session_state.injury_scorer  = InjuryRiskScorer()
        st.session_state.session_mgr    = SessionManager()
        st.session_state.explainer      = GradientExplainer(
            st.session_state.analyzer)
        st.session_state.multi_tracker  = MultiAthleteTracker(mp)
        # ── Upgrade: instantiate the new engines ──
        st.session_state.cue_engine    = CorrectiveCueEngine()
        st.session_state.gait_analyzer = GaitPhaseAnalyzer()
        st.session_state.track_monitor = TrackingConfidenceMonitor()
        st.session_state.models_loaded  = True


# ══════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════
def build_timeline(show_injury=True):
    if not st.session_state.timestamps:
        fig = go.Figure()
        fig.update_layout(
            height=240, **PLOTLY_LAYOUT,
            annotations=[dict(
                text='NO DATA YET',
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                font=dict(
                    family='Bebas Neue',
                    size=24,
                    color='rgba(0,210,255,0.2)'
                ),
                showarrow=False
            )]
        )
        return fig

    t   = list(st.session_state.timestamps)
    fat = list(st.session_state.fatigue_history)
    per = list(st.session_state.perf_history)
    inj = list(st.session_state.injury_history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=fat, name='FATIGUE',
        line=dict(color='#ff4757', width=2),
        fill='tozeroy',
        fillcolor='rgba(255,71,87,0.06)',
        hovertemplate='%{y:.1f}<extra>Fatigue</extra>'
    ))
    fig.add_trace(go.Scatter(
        x=t, y=per, name='PERFORMANCE',
        line=dict(color='#2ed573', width=2),
        fill='tozeroy',
        fillcolor='rgba(46,213,115,0.06)',
        hovertemplate='%{y:.1f}<extra>Performance</extra>'
    ))
    if show_injury:
        fig.add_trace(go.Scatter(
            x=t, y=inj, name='INJURY RISK',
            line=dict(color='#ffa502',
                      width=1.5, dash='dot'),
            hovertemplate='%{y:.1f}<extra>Injury Risk</extra>'
        ))
    fig.update_layout(
        height=240,
        yaxis=dict(range=[0, 100],
                   **PLOTLY_LAYOUT['yaxis']),
        xaxis=dict(title='seconds',
                   **PLOTLY_LAYOUT['xaxis']),
        **{k:v for k,v in PLOTLY_LAYOUT.items()
           if k not in ('xaxis','yaxis')}
    )
    return fig


def build_injury_radar(report):
    cats   = ['Knee', 'Hip', 'Ankle', 'Trunk', 'Knee']
    values = [
        report.knee_stress,
        report.hip_imbalance,
        report.ankle_instability,
        report.overstriding,
        report.knee_stress
    ]
    colors = ['#ff4757' if v > 60 else
              '#ffa502' if v > 30 else '#2ed573'
              for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=cats,
        fill='toself',
        fillcolor='rgba(0,210,255,0.06)',
        line=dict(color='#00d2ff', width=2),
        name='Risk'
    ))
    fig.update_layout(
        height=220,
        polar=dict(
            bgcolor='rgba(8,13,20,0.8)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=8,
                              color='#3d5a6e'),
                gridcolor='rgba(0,210,255,0.08)'
            ),
            angularaxis=dict(
                tickfont=dict(size=10,
                    color='#7ab8c8',
                    family='JetBrains Mono'),
                gridcolor='rgba(0,210,255,0.08)'
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    return fig


def build_injury_bars(report):
    cats   = ['Knee', 'Hip', 'Ankle', 'Trunk']
    values = [
        report.knee_stress,
        report.hip_imbalance,
        report.ankle_instability,
        report.overstriding
    ]
    colors = ['#ff4757' if v > 60 else
              '#ffa502' if v > 30 else '#2ed573'
              for v in values]
    fig = go.Figure(go.Bar(
        x=cats, y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.0f}" for v in values],
        textposition='outside',
        textfont=dict(
            family='JetBrains Mono',
            size=11, color='#7ab8c8'
        ),
        width=0.5
    ))
    fig.update_layout(
        height=200,
        yaxis=dict(range=[0, 115],
                   **PLOTLY_LAYOUT['yaxis']),
        xaxis=dict(**PLOTLY_LAYOUT['xaxis']),
        showlegend=False,
        **{k:v for k,v in PLOTLY_LAYOUT.items()
           if k not in ('xaxis','yaxis')}
    )
    return fig


def build_gait_chart():
    """Ground-contact-time L vs R over the session + asymmetry band."""
    hist = list(st.session_state.gait_history)
    if not hist:
        fig = go.Figure()
        fig.update_layout(
            height=200, **PLOTLY_LAYOUT,
            annotations=[dict(
                text='GATHERING STRIDES',
                x=0.5, y=0.5, xref='paper', yref='paper',
                font=dict(family='Bebas Neue', size=20,
                          color='rgba(0,210,255,0.2)'),
                showarrow=False
            )]
        )
        return fig

    idx = list(range(len(hist)))
    lc  = [h['contact_time_left_ms']  for h in hist]
    rc  = [h['contact_time_right_ms'] for h in hist]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=lc, name='CONTACT L',
        line=dict(color='#00d2ff', width=2),
        hovertemplate='%{y:.0f} ms<extra>Left</extra>'
    ))
    fig.add_trace(go.Scatter(
        x=idx, y=rc, name='CONTACT R',
        line=dict(color='#a55eea', width=2),
        hovertemplate='%{y:.0f} ms<extra>Right</extra>'
    ))
    fig.update_layout(
        height=200,
        yaxis=dict(title='ms', **PLOTLY_LAYOUT['yaxis']),
        xaxis=dict(title='sample', **PLOTLY_LAYOUT['xaxis']),
        **{k:v for k,v in PLOTLY_LAYOUT.items()
           if k not in ('xaxis','yaxis')}
    )
    return fig


def build_comparison(compare):
    df_a = compare['df_a']
    df_b = compare['df_b']
    fig  = go.Figure()
    for df, label, dash in [
        (df_a,'A','solid'),
        (df_b,'B','dash')
    ]:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['fatigue'],
            name=f"SESSION {label} — FATIGUE",
            line=dict(color='#ff4757',
                      width=2, dash=dash)
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['performance'],
            name=f"SESSION {label} — PERF",
            line=dict(color='#2ed573',
                      width=2, dash=dash)
        ))
    fig.update_layout(
        height=300,
        yaxis=dict(range=[0,100],
                   **PLOTLY_LAYOUT['yaxis']),
        xaxis=dict(title='seconds',
                   **PLOTLY_LAYOUT['xaxis']),
        **{k:v for k,v in PLOTLY_LAYOUT.items()
           if k not in ('xaxis','yaxis')}
    )
    return fig


def build_quality_pie(df):
    if 'quality' not in df.columns:
        return go.Figure()
    qd = df['quality'].value_counts().reset_index()
    qd.columns = ['Quality', 'Count']
    fig = px.pie(
        qd, names='Quality', values='Count',
        color='Quality',
        color_discrete_map={
            'Optimal':  '#2ed573',
            'Degraded': '#ffa502',
            'Critical': '#ff4757'
        },
        hole=0.55
    )
    fig.update_traces(
        textfont=dict(
            family='JetBrains Mono', size=10
        ),
        marker=dict(
            line=dict(color='#030508', width=2)
        )
    )
    fig.update_layout(
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono',
                  color='#7ab8c8', size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            font=dict(size=10),
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0
        ),
        showlegend=True
    )
    return fig


def render_cues(placeholder, cues):
    """Render the top corrective cues into a placeholder."""
    if not cues:
        placeholder.markdown(
            '<div class="rec-box">FORM LOOKS CLEAN — '
            'no corrections needed.</div>',
            unsafe_allow_html=True
        )
        return
    html = ""
    for c in cues[:3]:
        html += f"""
<div class="cue-card" style="border-left-color:{c['color']}">
  <div class="cue-tag" style="color:{c['color']}">
    {c['severity']} · {c['headline']} · {c['metric_value']}</div>
  <div class="cue-fix">{c['fix']}</div>
  <div class="cue-drill">▸ {c['drill']}</div>
  <div class="cue-why">{c['why']}</div>
</div>"""
    placeholder.markdown(html, unsafe_allow_html=True)


# ── Upgrade: adapt PoseExtractor's PoseFrame to the name->point dict the
# gait analyzer and tracking monitor expect. PoseFrame stores a (33,3)
# keypoints array + (33,) visibility; we map the joints those modules use
# to {name: (x, y, conf)}. Respects the frame's `valid` flag.
_LANDMARK_IDX = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
}

def pose_frame_to_dict(pf):
    """PoseFrame -> {joint_name: (x, y, visibility)} for the new engines."""
    if pf is None or not getattr(pf, "valid", False):
        return {}
    kp  = getattr(pf, "keypoints", None)
    vis = getattr(pf, "visibility", None)
    if kp is None:
        return {}
    out = {}
    n = len(kp)
    for name, idx in _LANDMARK_IDX.items():
        if idx < n:
            v = float(vis[idx]) if vis is not None and idx < len(vis) else 1.0
            out[name] = (float(kp[idx][0]), float(kp[idx][1]), v)
    return out


def render_tracking(placeholder, track):
    """Render the tracking-confidence gauge into a placeholder."""
    if track is None:
        placeholder.empty()
        return
    placeholder.markdown(f"""
<div class="big-metric"
     style="--accent-color:{track['color']};--accent-rgb:0,210,255">
  <div class="big-metric-label">Tracking</div>
  <div class="big-metric-value" style="font-size:32px">
    {track['score']*100:.0f}<span class="big-metric-unit">%</span></div>
  <div class="big-metric-sub">{track['reason']}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# FRAME OVERLAY
# ══════════════════════════════════════════════════
def process_overlay(frame, pred, stride, injury):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (320,200),
                  (3,5,8), -1)
    cv2.addWeighted(overlay, 0.82, frame,
                    0.18, 0, frame)

    # Cyan corner accent
    cv2.line(frame, (0,0), (40,0), (0,210,255), 2)
    cv2.line(frame, (0,0), (0,40), (0,210,255), 2)

    rc = {
        'low':      (46, 213, 115),
        'medium':   (255, 165, 2),
        'high':     (255, 71, 87),
        'critical': (255, 71, 87),
    }
    col = rc.get(pred.risk_level, (255,255,255))

    lines = [
        (f"FATIGUE   {pred.fatigue_score:.0f}",    col),
        (f"PERF      {pred.performance_score:.0f}", (0,210,255)),
        (f"QUALITY   {pred.movement_quality}",
         (122, 184, 200)),
        (f"CADENCE   {stride.cadence_spm:.0f} spm",
         (46, 213, 115)),
        (f"STRIDES   {stride.total_strides}",
         (46, 213, 115)),
        (f"INJ RISK  {injury.overall_risk:.0f}",
         (255, 165, 2)),
        (f"RISK      {pred.risk_level.upper()}", col),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text,
            (12, 28 + i*24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, color, 1, cv2.LINE_AA)

    # Fatigue bar at bottom
    bar_w = w
    bar_h = 4
    fat_n = pred.fatigue_score / 100
    fat_color = (
        (255,71,87) if fat_n > 0.7 else
        (255,165,2) if fat_n > 0.4 else
        (46,213,115)
    )
    cv2.rectangle(frame,
        (0, h-bar_h), (w, h),
        (12,18,32), -1)
    cv2.rectangle(frame,
        (0, h-bar_h), (int(w*fat_n), h),
        fat_color, -1)

    return frame


# ══════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ══════════════════════════════════════════════════
def run_analysis_pipeline(
    video_path, fatigue_threshold,
    perf_threshold, multi_mode,
    voice_enabled,
    video_ph, chart_ph, stride_ph,
    injury_ph, xai_ph,
    gait_ph=None, cue_ph=None, track_ph=None,
    analyze_every: int = 3
):
    extractor      = st.session_state.extractor
    feature_eng    = st.session_state.feature_eng
    analyzer       = st.session_state.analyzer
    stride_counter = st.session_state.stride_counter
    injury_scorer  = st.session_state.injury_scorer
    explainer      = st.session_state.explainer
    multi_tracker  = st.session_state.multi_tracker
    cue_engine     = st.session_state.cue_engine
    gait_analyzer  = st.session_state.gait_analyzer
    track_monitor  = st.session_state.track_monitor
    voice          = VoiceCoach(enabled=voice_enabled)
    # Cooldown-aware cue gate for voice (so a cue isn't spoken every frame).
    cue_session    = CueSession(cooldown_s=12)

    feature_eng.reset()
    analyzer.reset()
    stride_counter.reset()
    injury_scorer.reset()
    gait_analyzer.reset()
    track_monitor.reset()
    if multi_mode:
        multi_tracker.reset()

    # ── Upgrade (#7): drive timing off the SOURCE FPS, not wall-clock. ──
    # On a slow CPU, time.time() stretches and corrupts cadence/stride
    # timing. media_time(frame_n) stays correct regardless of CPU speed.
    norm = FPSNormalizer.from_video(video_path)
    st.session_state.source_fps = norm.info.as_dict()

    xai_every = 90
    frame_n   = 0

    last_pred   = None
    last_stride = None
    last_injury = None
    last_gait   = None

    for annotated, pose_frame in extractor.process_video(
        video_path
    ):
        should_analyze = (frame_n % analyze_every == 0)

        # ── Upgrade (#6): rate the pose quality EVERY frame. ──
        pose_dict = pose_frame_to_dict(pose_frame)
        track = track_monitor.update(pose_dict)
        st.session_state.last_track = track.as_dict()

        if should_analyze:
            # Source-accurate media time for this frame.
            t = round(norm.media_time(frame_n), 2)

            features = feature_eng.extract(pose_frame)
            pred     = analyzer.update(features)
            stride   = stride_counter.update(features)
            injury   = injury_scorer.update(features)
            # ── Upgrade (#2): gait phase metrics from raw landmarks. ──
            gait     = gait_analyzer.update(pose_dict, t)

            last_pred   = pred
            last_stride = stride
            last_injury = injury
            last_gait   = gait

            # ── Upgrade (#1): corrective cues from all signals. ──
            cues = cue_engine.evaluate(
                knee_stress=injury.knee_stress,
                hip_imbalance=injury.hip_imbalance,
                ankle_instability=injury.ankle_instability,
                overstriding=injury.overstriding,
                cadence_spm=stride.cadence_spm,
                stride_regularity=stride.stride_regularity,
                fatigue_score=pred.fatigue_score,
                vertical_oscillation_cm=gait.vertical_oscillation_cm,
                gct_asymmetry_pct=gait.gct_asymmetry_pct,
            )
            st.session_state.last_cues = [c.as_dict() for c in cues]

            # Voice: only speak newly-due cues (cooldown-gated).
            if voice_enabled:
                for c in cue_session.update(
                    tick=t,
                    knee_stress=injury.knee_stress,
                    hip_imbalance=injury.hip_imbalance,
                    ankle_instability=injury.ankle_instability,
                    overstriding=injury.overstriding,
                    cadence_spm=stride.cadence_spm,
                    stride_regularity=stride.stride_regularity,
                    fatigue_score=pred.fatigue_score,
                    vertical_oscillation_cm=gait.vertical_oscillation_cm,
                    gct_asymmetry_pct=gait.gct_asymmetry_pct,
                ):
                    voice.say(c.fix)

            # ── Upgrade (#6): only log reliable frames into history. ──
            # Junk tracking (occlusion / out-of-frame) never pollutes
            # session stats — trustworthiness is a feature.
            if track.is_reliable:
                st.session_state.fatigue_history.append(
                    pred.fatigue_score)
                st.session_state.perf_history.append(
                    pred.performance_score)
                st.session_state.quality_history.append(
                    pred.movement_quality)
                st.session_state.cadence_history.append(
                    stride.cadence_spm)
                st.session_state.injury_history.append(
                    injury.overall_risk)
                st.session_state.stride_history.append(
                    stride.total_strides)
                st.session_state.gait_history.append(
                    gait.as_dict())
                st.session_state.timestamps.append(t)
                st.session_state.frame_count += 1

                for cond, msg in [
                    (pred.fatigue_score > fatigue_threshold,
                     f"FATIGUE {pred.fatigue_score:.0f} — {t}s"),
                    (pred.performance_score < perf_threshold,
                     f"PERF DROP {pred.performance_score:.0f} — {t}s"),
                    (injury.risk_level in ('high','critical'),
                     f"{injury.risk_level.upper()} INJURY RISK — {t}s")
                ]:
                    if cond and (
                        not st.session_state.alerts or
                        st.session_state.alerts[-1] != msg
                    ):
                        st.session_state.alerts.append(msg)

        if last_pred is not None:
            if multi_mode:
                annotated, _ = \
                    multi_tracker.process_frame(annotated)
            else:
                annotated = process_overlay(
                    annotated, last_pred,
                    last_stride, last_injury
                )

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_ph.image(rgb, use_container_width=True)

        # Live tracking gauge.
        if track_ph is not None:
            render_tracking(track_ph, st.session_state.last_track)

        if should_analyze and last_pred is not None:
            chart_ph.plotly_chart(
                build_timeline(),
                use_container_width=True,
                key=f"tl_{frame_n}"
            )

            # Stride cards
            s = last_stride
            stride_ph.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
  <div class="big-metric" style="--accent-color:#2ed573;--accent-rgb:46,213,115">
    <div class="big-metric-label">Total Strides</div>
    <div class="big-metric-value">{s.total_strides}</div>
    <div class="big-metric-sub">detected steps</div>
  </div>
  <div class="big-metric" style="--accent-color:#00d2ff;--accent-rgb:0,210,255">
    <div class="big-metric-label">Cadence</div>
    <div class="big-metric-value">{s.cadence_spm:.0f}<span class="big-metric-unit">spm</span></div>
    <div class="big-metric-sub">{s.current_pace}</div>
  </div>
  <div class="big-metric" style="--accent-color:#a55eea;--accent-rgb:165,94,234">
    <div class="big-metric-label">Regularity</div>
    <div class="big-metric-value">{s.stride_regularity*100:.0f}<span class="big-metric-unit">%</span></div>
    <div class="big-metric-sub">stride consistency</div>
  </div>
  <div class="big-metric" style="--accent-color:#ffa502;--accent-rgb:255,165,2">
    <div class="big-metric-label">Avg Stride Time</div>
    <div class="big-metric-value">{s.avg_stride_time:.2f}<span class="big-metric-unit">s</span></div>
    <div class="big-metric-sub">per cycle</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # Injury panel
            with injury_ph.container():
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        build_injury_bars(last_injury),
                        use_container_width=True,
                        key=f"injb_{frame_n}"
                    )
                with c2:
                    st.plotly_chart(
                        build_injury_radar(last_injury),
                        use_container_width=True,
                        key=f"injr_{frame_n}"
                    )
                for flag in last_injury.flags[:2]:
                    st.markdown(
                        f'<div class="alert-box">'
                        f'{flag}</div>',
                        unsafe_allow_html=True
                    )

            # ── Upgrade (#2): GAIT tab ──
            if gait_ph is not None and last_gait is not None:
                g = last_gait.as_dict()
                with gait_ph.container():
                    if g["confidence"] >= 0.5:
                        a1, a2, a3 = st.columns(3)
                        a1.metric("CONTACT L",
                                  f"{g['contact_time_left_ms']:.0f} ms")
                        a2.metric("CONTACT R",
                                  f"{g['contact_time_right_ms']:.0f} ms")
                        asym_c = ('🔴' if g['gct_asymmetry_pct'] > 6 else
                                  '🟡' if g['gct_asymmetry_pct'] > 3 else '🟢')
                        a3.metric("L/R ASYM",
                                  f"{g['gct_asymmetry_pct']:.1f} %",
                                  asym_c)
                        b1, b2, b3 = st.columns(3)
                        b1.metric("VERT OSC",
                                  f"{g['vertical_oscillation_cm']:.1f} cm")
                        b2.metric("FLIGHT",
                                  f"{g['flight_ratio']*100:.0f} %")
                        b3.metric("DUTY",
                                  f"{g['duty_factor']*100:.0f} %")
                        st.plotly_chart(
                            build_gait_chart(),
                            use_container_width=True,
                            key=f"gait_{frame_n}"
                        )
                    else:
                        st.caption(
                            "Gathering strides… need a few seconds "
                            "of clean tracking.")

            # ── Upgrade (#1): COACHING cues ──
            if cue_ph is not None:
                render_cues(cue_ph, st.session_state.last_cues)

        # XAI
        if frame_n % xai_every == 0 and \
           len(analyzer.buffer) >= 30:
            try:
                window = np.array(analyzer.buffer[-30:])
                imp    = explainer.compute_importance(window)
                st.session_state.last_importance = imp
                top5   = explainer.top_contributors(imp, n=5)

                with xai_ph.container():
                    st.plotly_chart(
                        explainer.build_skeleton_heatmap(imp),
                        use_container_width=True,
                        key=f"xai_{frame_n}"
                    )
                    bars_html = ""
                    for item in top5:
                        pct    = item['importance'] * 100
                        filled = int(pct)
                        bars_html += f"""
<div class="xai-row">
  <div class="xai-label">{item['feature']}</div>
  <div class="xai-bar-track">
    <div class="xai-bar-fill"
         style="width:{filled}%"></div>
  </div>
  <div class="xai-val">{pct:.1f}%</div>
</div>"""
                    st.markdown(bars_html,
                                unsafe_allow_html=True)
            # ── Upgrade (#5): narrow catch + log instead of bare except ──
            except Exception as exc:
                st.session_state.last_importance = None
                print(f"[XAI] importance failed: {exc!r}")

        frame_n += 1

    return last_stride, last_injury


# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div class="sidebar-apex">APEX</div>
<div class="sidebar-sub">Sports Performance AI</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation", [
            "ANALYZE VIDEO",
            "LIVE WEBCAM",
            "MULTI-ATHLETE",
            "SESSION HISTORY",
            "COMPARE SESSIONS",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    voice_on = st.toggle("VOICE COACH", value=False)

    st.markdown(
        '<p style="font-family:\'JetBrains Mono\';'
        'font-size:9px;color:#3d5a6e;'
        'letter-spacing:0.15em;text-transform:uppercase;'
        'margin:16px 0 8px">Alert Thresholds</p>',
        unsafe_allow_html=True
    )
    fatigue_alert = st.slider(
        "Fatigue", 40, 90, 70,
        label_visibility="collapsed"
    )
    st.caption(f"FATIGUE ALERT · {fatigue_alert}")
    perf_alert = st.slider(
        "Performance", 10, 60, 30,
        label_visibility="collapsed"
    )
    st.caption(f"PERF ALERT · {perf_alert}")

    st.markdown("---")

    if st.button("RESET SESSION"):
        for k in [
            'fatigue_history','perf_history',
            'quality_history','cadence_history',
            'injury_history','stride_history',
            'timestamps','alerts','gait_history',
        ]:
            s = st.session_state[k]
            if isinstance(s, deque): s.clear()
            elif isinstance(s, list):
                st.session_state[k] = []
        st.session_state.frame_count = 0
        st.session_state.last_cues   = []
        st.session_state.last_track  = None
        # Reset the rolling engines too, so a new run starts clean.
        if st.session_state.gait_analyzer is not None:
            st.session_state.gait_analyzer.reset()
        if st.session_state.track_monitor is not None:
            st.session_state.track_monitor.reset()
        st.rerun()

    st.markdown("---")

    st.markdown(f"""
<div style="font-family:'JetBrains Mono';
     font-size:9px;color:#3d5a6e;
     letter-spacing:0.12em;line-height:2">
  FRAMES &nbsp;&nbsp;{st.session_state.frame_count}<br>
  ALERTS &nbsp;&nbsp;{len(st.session_state.alerts)}<br>
  MODELS &nbsp;&nbsp;{'LOADED' if st.session_state.models_loaded else 'PENDING'}
</div>
""", unsafe_allow_html=True)


load_models(voice_enabled=voice_on)


# ══════════════════════════════════════════════════
# PAGE: ANALYZE VIDEO
# ══════════════════════════════════════════════════
if page == "ANALYZE VIDEO":

    st.markdown("""
<div class="apex-header">
  <div class="apex-wordmark">APEX</div>
  <div class="apex-sub">Real-Time Sports Performance Intelligence</div>
  <div class="apex-status">
    <div class="status-dot"></div>
    Systems Online
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="phase-badge">⚡ Phase 1–3 · Running Analysis</div>',
                unsafe_allow_html=True)

    # Top metric row
    m1,m2,m3,m4 = st.columns(4)
    last_fat  = list(st.session_state.fatigue_history)[-1] \
        if st.session_state.fatigue_history else 0
    last_perf = list(st.session_state.perf_history)[-1] \
        if st.session_state.perf_history else 100
    last_inj  = list(st.session_state.injury_history)[-1] \
        if st.session_state.injury_history else 0
    last_cad  = list(st.session_state.cadence_history)[-1] \
        if st.session_state.cadence_history else 0

    fat_color  = ('#ff4757' if last_fat > 70 else
                  '#ffa502' if last_fat > 40 else '#2ed573')
    fat_rgb    = ('255,71,87' if last_fat > 70 else
                  '255,165,2' if last_fat > 40 else '46,213,115')
    perf_color = '#2ed573'
    inj_color  = ('#ff4757' if last_inj > 60 else
                  '#ffa502' if last_inj > 30 else '#2ed573')
    inj_rgb    = ('255,71,87' if last_inj > 60 else
                  '255,165,2' if last_inj > 30 else '46,213,115')

    for col, label, val, color, rgb, unit, sub in [
        (m1,'Fatigue Score', last_fat, fat_color, fat_rgb,
         '/100','current level'),
        (m2,'Performance',   last_perf,perf_color,'46,213,115',
         '/100','movement quality'),
        (m3,'Injury Risk',   last_inj, inj_color, inj_rgb,
         '/100','biomechanical'),
        (m4,'Cadence',       last_cad, '#00d2ff','0,210,255',
         'spm','strides/min'),
    ]:
        col.markdown(f"""
<div class="big-metric"
     style="--accent-color:{color};--accent-rgb:{rgb}">
  <div class="big-metric-label">{label}</div>
  <div class="big-metric-value">{val:.0f}<span
       class="big-metric-unit">{unit}</span></div>
  <div class="big-metric-sub">{sub}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>",
                unsafe_allow_html=True)

    # Main layout
    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.markdown("""
<div class="section-head">
  <span class="section-head-text">Live Feed</span>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)
        video_ph = st.empty()
        # Tracking-confidence gauge sits under the feed.
        track_ph = st.empty()
        render_tracking(track_ph, st.session_state.last_track)

    with right_col:
        tab1, tab2, tab3, tab4 = st.tabs([
            "TIMELINE", "STRIDES", "INJURY", "GAIT"
        ])
        with tab1:
            chart_ph = st.empty()
            chart_ph.plotly_chart(
                build_timeline(),
                use_container_width=True,
                key="tl_init"
            )
        with tab2:
            stride_ph = st.empty()
        with tab3:
            injury_ph = st.empty()
        with tab4:
            gait_ph = st.empty()
            gait_ph.plotly_chart(
                build_gait_chart(),
                use_container_width=True,
                key="gait_init"
            )

    # ── Upgrade (#1): COACHING section ──
    st.markdown("""
<div class="section-head" style="margin-top:24px">
  <span class="section-head-text">
    Coaching — Corrective Cues
  </span>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)
    cue_ph = st.empty()
    render_cues(cue_ph, st.session_state.last_cues)

    st.markdown("""
<div class="section-head" style="margin-top:24px">
  <span class="section-head-text">
    Explainable AI — Fatigue Heatmap
  </span>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)
    xai_ph = st.empty()

    st.markdown("""
<div class="section-head" style="margin-top:8px">
  <span class="section-head-text">Session Alerts</span>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)
    alert_ph = st.empty()

    # Upload
    uploaded = st.file_uploader(
        "DROP RUNNING VIDEO",
        type=['mp4','mov','avi','mkv'],
        label_visibility="collapsed"
    )
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\';'
        'font-size:9px;color:#3d5a6e;'
        'letter-spacing:0.15em;margin-top:6px">'
        'SUPPORTED · MP4 · MOV · AVI · MKV</p>',
        unsafe_allow_html=True
    )

    if uploaded and st.button("⚡ INITIATE ANALYSIS"):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.mp4'
        ) as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        fs, fi = run_analysis_pipeline(
            path, fatigue_alert, perf_alert,
            multi_mode=False,
            voice_enabled=voice_on,
            video_ph=video_ph,
            chart_ph=chart_ph,
            stride_ph=stride_ph,
            injury_ph=injury_ph,
            xai_ph=xai_ph,
            gait_ph=gait_ph,
            cue_ph=cue_ph,
            track_ph=track_ph,
        )
        os.unlink(path)

        # Surface the detected source FPS (proves the timing is honest).
        if st.session_state.source_fps:
            sf = st.session_state.source_fps
            st.caption(
                f"SOURCE · {sf['fps']:.0f} fps ({sf['source']})  ·  "
                f"timing normalized to media-time")

        if st.session_state.alerts:
            alert_ph.markdown(
                "".join([
                    f'<div class="alert-box">{a}</div>'
                    for a in st.session_state.alerts[-5:]
                ]),
                unsafe_allow_html=True
            )

        if st.session_state.timestamps:
            sid = st.session_state.session_mgr\
                .save_session(
                timestamps  =list(st.session_state.timestamps),
                fatigue     =list(st.session_state.fatigue_history),
                performance =list(st.session_state.perf_history),
                quality     =list(st.session_state.quality_history),
                strides     =list(st.session_state.stride_history),
                cadence     =list(st.session_state.cadence_history),
                injury_risk =list(st.session_state.injury_history),
                alerts      =st.session_state.alerts
            )
            st.success(f"SESSION SAVED · {sid}")
            df = pd.DataFrame({
                'time':       list(st.session_state.timestamps),
                'fatigue':    list(st.session_state.fatigue_history),
                'performance':list(st.session_state.perf_history),
                'quality':    list(st.session_state.quality_history),
                'cadence':    list(st.session_state.cadence_history),
                'injury_risk':list(st.session_state.injury_history),
                'strides':    list(st.session_state.stride_history),
            })
            st.download_button(
                "⬇ EXPORT SESSION REPORT",
                df.to_csv(index=False),
                f"apex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )


# ══════════════════════════════════════════════════
# PAGE: LIVE WEBCAM
# ══════════════════════════════════════════════════
elif page == "LIVE WEBCAM":
    st.markdown("""
<div class="apex-header">
  <div class="apex-wordmark">LIVE</div>
  <div class="apex-sub">Real-Time Webcam Analysis</div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        start_btn = st.button(
            "▶ START",
            disabled=st.session_state.webcam_running
        )
    with c2:
        stop_btn = st.button(
            "■ STOP",
            disabled=not st.session_state.webcam_running
        )
    with c3:
        cam_idx = st.number_input(
            "CAM INDEX", min_value=0,
            max_value=4, value=0, step=1
        )

    if start_btn and not st.session_state.webcam_running:
        mp = "models/sports_lstm.pth" \
            if os.path.exists("models/sports_lstm.pth") \
            else None
        proc = WebcamProcessor(
            model_path=mp,
            voice_enabled=voice_on,
            camera_index=int(cam_idx),
            analyze_every=3
        )
        proc.start()
        st.session_state.webcam_proc    = proc
        st.session_state.webcam_running = True
        st.rerun()

    if stop_btn and st.session_state.webcam_running:
        if st.session_state.webcam_proc:
            st.session_state.webcam_proc.stop()
        st.session_state.webcam_running = False
        st.session_state.webcam_proc    = None
        st.rerun()

    if st.session_state.webcam_running and \
       st.session_state.webcam_proc:

        proc = st.session_state.webcam_proc

        st.markdown(
            '<div class="live-badge">'
            '<span style="animation:blink 1s infinite">'
            '●</span> LIVE ANALYSIS RUNNING'
            '</div>',
            unsafe_allow_html=True
        )

        feed_col, metric_col = st.columns([1.4, 1])

        with feed_col:
            frame_ph = st.empty()
        with metric_col:
            fat_ph  = st.empty()
            perf_ph = st.empty()
            cad_ph  = st.empty()
            inj_ph2 = st.empty()

        chart_ph2 = st.empty()
        # ── Upgrade (#1): live coaching cues under the chart ──
        cue_ph2 = st.empty()

        # Drain predictions
        preds = proc.get_predictions()
        for p in preds[-10:]:
            st.session_state.fatigue_history.append(
                p['fatigue'])
            st.session_state.perf_history.append(
                p['performance'])
            st.session_state.timestamps.append(
                p['timestamp'])
            st.session_state.cadence_history.append(
                p['cadence'])
            st.session_state.injury_history.append(
                p['injury_risk'])
            st.session_state.stride_history.append(
                p['strides'])
            st.session_state.frame_count += 1

        # ── Upgrade (#1): compute cues from the freshest live values. ──
        if preds:
            p = preds[-1]
            cues = st.session_state.cue_engine.evaluate(
                knee_stress=p.get('knee_stress'),
                hip_imbalance=p.get('hip_imbalance'),
                ankle_instability=p.get('ankle_instability'),
                overstriding=p.get('overstriding'),
                cadence_spm=p.get('cadence'),
                stride_regularity=p.get('stride_regularity'),
                fatigue_score=p.get('fatigue'),
                vertical_oscillation_cm=p.get('vertical_oscillation_cm'),
                gct_asymmetry_pct=p.get('gct_asymmetry_pct'),
            )
            st.session_state.last_cues = [c.as_dict() for c in cues]

        # Show frame
        frame = proc.get_latest_frame()
        if frame is not None:
            frame_ph.image(
                frame, channels="RGB",
                use_container_width=True
            )
        else:
            frame_ph.markdown(
                '<div style="background:#080d14;'
                'border:1px solid rgba(0,210,255,0.12);'
                'height:320px;display:flex;'
                'align-items:center;'
                'justify-content:center;'
                'font-family:\'Bebas Neue\';'
                'font-size:32px;'
                'color:rgba(0,210,255,0.2);'
                'letter-spacing:0.1em">'
                'AWAITING CAMERA FEED'
                '</div>',
                unsafe_allow_html=True
            )

        # Live metrics
        if st.session_state.fatigue_history:
            lf = list(st.session_state.fatigue_history)[-1]
            lp = list(st.session_state.perf_history)[-1]
            lc = list(st.session_state.cadence_history)[-1]
            li = list(st.session_state.injury_history)[-1]

            fc = ('#ff4757' if lf>70 else
                  '#ffa502' if lf>40 else '#2ed573')
            fr = ('255,71,87' if lf>70 else
                  '255,165,2' if lf>40 else '46,213,115')
            ic = ('#ff4757' if li>60 else
                  '#ffa502' if li>30 else '#2ed573')
            ir = ('255,71,87' if li>60 else
                  '255,165,2' if li>30 else '46,213,115')

            for ph, label, val, color, rgb, unit in [
                (fat_ph, 'Fatigue',    lf, fc, fr,    '/100'),
                (perf_ph,'Performance',lp,'#2ed573',
                 '46,213,115','/100'),
                (cad_ph, 'Cadence',    lc,'#00d2ff',
                 '0,210,255','spm'),
                (inj_ph2,'Injury Risk',li, ic, ir,    '/100'),
            ]:
                ph.markdown(f"""
<div class="big-metric"
     style="--accent-color:{color};
            --accent-rgb:{rgb};
            margin-bottom:8px">
  <div class="big-metric-label">{label}</div>
  <div class="big-metric-value" style="font-size:40px">
    {val:.0f}<span class="big-metric-unit">{unit}</span>
  </div>
</div>""", unsafe_allow_html=True)

        chart_ph2.plotly_chart(
            build_timeline(),
            use_container_width=True,
            key=f"live_tl_{st.session_state.frame_count}"
        )

        # Render live coaching cues.
        render_cues(cue_ph2, st.session_state.last_cues)

        st.caption(
            f"FPS · {proc.fps_actual:.1f}  ·  "
            f"ANALYZED · {proc.analyze_count} frames  ·  "
            f"DISPLAYED · {proc.frame_count} frames"
        )

        time.sleep(0.1)
        st.rerun()

    else:
        st.markdown("""
<div class="trouble-box">
  <b>QUICK START</b><br>
  Click ▶ START to begin live analysis.<br><br>
  <b>TROUBLESHOOTING</b><br>
  · Black screen → try camera index 1 or 2<br>
  · Another app using camera → close Teams / Zoom / OBS<br>
  · Slow / laggy → normal on CPU, analysis runs every 3rd frame<br>
  · Voice silent → toggle Voice Coach ON in sidebar
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PAGE: MULTI-ATHLETE
# ══════════════════════════════════════════════════
elif page == "MULTI-ATHLETE":
    st.markdown("""
<div class="apex-header">
  <div class="apex-wordmark">SQUAD</div>
  <div class="apex-sub">Multi-Athlete Tracking · YOLOv8 + LSTM</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<p style="font-family:\'JetBrains Mono\';'
        'font-size:11px;color:#3d5a6e;margin-bottom:24px">'
        'Detects up to 5 athletes simultaneously. '
        'Each athlete gets individual pose extraction '
        'and performance scoring.</p>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "DROP MULTI-ATHLETE VIDEO",
        type=['mp4','mov','avi','mkv'],
        label_visibility="collapsed"
    )
    if uploaded and st.button("⚡ ANALYZE SQUAD"):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.mp4'
        ) as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        run_analysis_pipeline(
            path, fatigue_alert, perf_alert,
            multi_mode=True,
            voice_enabled=voice_on,
            video_ph=st.empty(),
            chart_ph=st.empty(),
            stride_ph=st.empty(),
            injury_ph=st.empty(),
            xai_ph=st.empty()
        )
        os.unlink(path)


# ══════════════════════════════════════════════════
# PAGE: SESSION HISTORY
# ══════════════════════════════════════════════════
elif page == "SESSION HISTORY":
    st.markdown("""
<div class="apex-header">
  <div class="apex-wordmark">HISTORY</div>
  <div class="apex-sub">Session Archive · Performance Trends</div>
</div>
""", unsafe_allow_html=True)

    sessions = st.session_state.session_mgr\
        .list_sessions()

    if not sessions:
        st.markdown("""
<div class="trouble-box">
  NO SESSIONS RECORDED YET.<br>
  Run an analysis on the ANALYZE VIDEO page first.
</div>
""", unsafe_allow_html=True)
    else:
        for s in reversed(sessions[-10:]):
            fat_c = ('#ff4757' if s['avg_fatigue']>70 else
                     '#ffa502' if s['avg_fatigue']>40 else
                     '#2ed573')
            with st.expander(
                f"{s['date']}  ·  "
                f"Fatigue {s['avg_fatigue']}  ·  "
                f"Strides {s['total_strides']}  ·  "
                f"Risk {s['injury_risk']}"
            ):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("AVG FATIGUE",
                          s['avg_fatigue'])
                c2.metric("AVG PERF",
                          s['avg_performance'])
                c3.metric("PEAK FATIGUE",
                          s['peak_fatigue'])
                c4.metric("INJURY RISK",
                          s['injury_risk'])

                df = st.session_state.session_mgr\
                    .load_session(s['session_id'])
                if df is not None:
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df['time'],
                            y=df['fatigue'],
                            name='FATIGUE',
                            line=dict(color='#ff4757')
                        ))
                        fig.add_trace(go.Scatter(
                            x=df['time'],
                            y=df['performance'],
                            name='PERF',
                            line=dict(color='#2ed573')
                        ))
                        fig.update_layout(
                            height=180,
                            yaxis=dict(range=[0,100],
                                **PLOTLY_LAYOUT['yaxis']),
                            xaxis=dict(
                                **PLOTLY_LAYOUT['xaxis']),
                            **{k:v for k,v in
                               PLOTLY_LAYOUT.items()
                               if k not in
                               ('xaxis','yaxis')}
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"hist_{s['session_id']}"
                        )
                    with cc2:
                        st.plotly_chart(
                            build_quality_pie(df),
                            use_container_width=True,
                            key=f"pie_{s['session_id']}"
                        )


# ══════════════════════════════════════════════════
# PAGE: COMPARE SESSIONS
# ══════════════════════════════════════════════════
elif page == "COMPARE SESSIONS":
    st.markdown("""
<div class="apex-header">
  <div class="apex-wordmark">COMPARE</div>
  <div class="apex-sub">Session Overlay · Performance Delta</div>
</div>
""", unsafe_allow_html=True)

    sessions = st.session_state.session_mgr\
        .list_sessions()

    if len(sessions) < 2:
        st.markdown("""
<div class="trouble-box">
  NEED AT LEAST 2 SAVED SESSIONS TO COMPARE.
</div>
""", unsafe_allow_html=True)
    else:
        ids = [s['session_id'] for s in sessions]
        c1, c2 = st.columns(2)
        with c1:
            sid_a = st.selectbox(
                "SESSION A", ids, index=0)
        with c2:
            sid_b = st.selectbox(
                "SESSION B", ids,
                index=min(1,len(ids)-1))

        if st.button("⚡ RUN COMPARISON") and \
           sid_a != sid_b:
            cmp = st.session_state.session_mgr\
                .compare_sessions(sid_a, sid_b)

            if cmp:
                st.markdown("""
<div class="section-head" style="margin-top:8px">
  <span class="section-head-text">
    Delta Metrics
  </span>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)

                metrics = [
                    ('avg_fatigue',     'AVG FATIGUE'),
                    ('avg_performance', 'AVG PERF'),
                    ('peak_fatigue',    'PEAK FATIGUE'),
                    ('avg_cadence',     'AVG CADENCE'),
                ]
                cols = st.columns(len(metrics))
                for i,(k,label) in enumerate(metrics):
                    av = cmp['session_a'].get(k,0)
                    bv = cmp['session_b'].get(k,0)
                    cols[i].metric(
                        label,
                        f"A · {av}",
                        f"B · {bv-av:+.1f} vs A"
                    )

                st.markdown("""
<div class="section-head" style="margin-top:24px">
  <span class="section-head-text">
    Performance Overlay
  </span>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)

                st.plotly_chart(
                    build_comparison(cmp),
                    use_container_width=True,
                    key="compare_main"
                )

                st.markdown("""
<div class="section-head">
  <span class="section-head-text">
    Movement Quality Distribution
  </span>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)

                q1, q2 = st.columns(2)
                for col, df, label, sid in [
                    (q1, cmp['df_a'],
                     "SESSION A", sid_a),
                    (q2, cmp['df_b'],
                     "SESSION B", sid_b),
                ]:
                    col.markdown(
                        f'<p style="font-family:'
                        f'\'JetBrains Mono\';'
                        f'font-size:10px;'
                        f'color:#3d5a6e;'
                        f'letter-spacing:0.15em;'
                        f'text-transform:uppercase;'
                        f'margin-bottom:8px">'
                        f'{label}</p>',
                        unsafe_allow_html=True
                    )
                    col.plotly_chart(
                        build_quality_pie(df),
                        use_container_width=True,
                        key=f"cpie_{sid}"
                    )
