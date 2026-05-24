<div align="center">

```
 ▄▄▄· ▄▄▄·▄▄▄ .▄▄▄·
▐█ ▀█ ▐█ ▄█▀▄.▀·▐█ ▄█
▄█▀▀█  ██▀·▐▀▀▪▄ ██▀·
▐█ ▪▐▌▐█▪·•▐█▄▄▌▐█▪·•
 ▀  ▀ .▀    ▀▀▀ .▀
```

# ⚡ APEX — Sports Performance AI

**Real-time athlete analysis powered by pose estimation, LSTM prediction, and explainable AI**

[![Python](https://img.shields.io/badge/Python-3.10%2B-00d2ff?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-00d2ff?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-ff4757?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-2ed573?style=flat-square)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-ffa502?style=flat-square)](LICENSE)

</div>

---

## 🎯 What is APEX?

APEX is a **cinematic-grade sports performance analytics platform** that analyzes athletes in real time — from uploaded video or live webcam — using a full ML pipeline: pose extraction → biomechanical feature engineering → LSTM-based fatigue & performance prediction → injury risk scoring → explainable AI attribution.

Built for coaches, sports scientists, and serious athletes who need more than a stopwatch.

---

## 🖥️ Pages & Features

| Page | Description |
|------|-------------|
| **⚡ ANALYZE VIDEO** | Upload a video, run the full ML pipeline, get frame-by-frame fatigue, performance, stride, and injury risk overlays |
| **📡 LIVE WEBCAM** | Real-time webcam analysis with live charts, voice coaching, and HUD overlays |
| **👥 MULTI-ATHLETE** | Tracks up to 5 athletes simultaneously using YOLOv8 + individual LSTM scoring |
| **📁 SESSION HISTORY** | Archive of past sessions with trend charts and movement quality breakdowns |
| **📊 COMPARE SESSIONS** | Overlay any two sessions — performance delta, cadence comparison, quality pie charts |

---

## 🧠 ML Pipeline

```
Video / Webcam
      │
      ▼
┌─────────────────────┐
│   PoseExtractor     │  ← MediaPipe Pose, 33-keypoint skeleton per frame
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ RunningFeatureEng.  │  ← Joint angles, stride metrics, symmetry, velocity vectors
└─────────────────────┘
      │
      ▼
┌─────────────────────┐     ┌──────────────────┐
│  PerformanceAnalyzer│     │  StrideCounter   │  ← Cadence (SPM), stride regularity,
│  (LSTM Model)       │     └──────────────────┘     total stride count
│  → fatigue_score    │
│  → performance_score│     ┌──────────────────┐
│  → movement_quality │     │ InjuryRiskScorer │  ← Knee stress, hip imbalance,
│  → risk_level       │     └──────────────────┘     ankle instability, overstriding
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  GradientExplainer  │  ← XAI: which features drove the prediction
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   VoiceCoach        │  ← Real-time audio cues for fatigue, form, cadence
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   SessionManager    │  ← Saves session data, enables history & comparison
└─────────────────────┘
```

---

## 📊 Metrics Tracked

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Fatigue Score** | 0–100 scale, LSTM-derived from biomechanical degradation signals | Configurable (default 70) |
| **Performance Score** | 0–100 scale, composite of form, power, and consistency | Configurable (default 40) |
| **Movement Quality** | Categorical: `Optimal` / `Degraded` / `Critical` | Visual + voice alert |
| **Cadence** | Steps per minute (SPM) from stride counter | — |
| **Stride Regularity** | Consistency of stride pattern over time | — |
| **Injury Risk** | Composite score: knee stress, hip imbalance, ankle instability, overstriding | `high` / `critical` triggers alert |

---

## 🏗️ Project Structure

```
sports-performance-analyzer/
│
├── app.py                          ← Main Streamlit application
│
├── src/
│   ├── pose/
│   │   ├── extractor.py            ← MediaPipe pose extraction, video frame generator
│   │   └── multi_tracker.py        ← YOLOv8 multi-athlete tracking (up to 5 athletes)
│   │
│   ├── features/
│   │   ├── engineer.py             ← Biomechanical feature extraction from keypoints
│   │   ├── stride_counter.py       ← Cadence, stride count, regularity scoring
│   │   └── injury_risk.py          ← Knee/hip/ankle/trunk risk scoring
│   │
│   ├── models/
│   │   ├── lstm_model.py           ← LSTM PerformanceAnalyzer (PyTorch)
│   │   └── explainability.py       ← Gradient-based XAI feature importance
│   │
│   └── utils/
│       ├── session_manager.py      ← Session save/load/compare
│       ├── voice_coach.py          ← TTS real-time audio coaching
│       └── webcam_handler.py       ← Live webcam frame processor
│
├── models/
│   └── sports_lstm.pth             ← Pre-trained LSTM weights (optional)
│
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Webcam (for live analysis)
- CUDA GPU (optional, but recommended for real-time)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/sports-performance-analyzer.git
cd sports-performance-analyzer

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ⚙️ Configuration

Adjust real-time thresholds from the **sidebar**:

| Control | Description |
|--------|-------------|
| **Fatigue Alert Threshold** | Trigger alert when fatigue exceeds this value (0–100) |
| **Performance Alert Threshold** | Trigger alert when performance drops below this (0–100) |
| **Analyze Every N Frames** | Trade-off between accuracy and speed (default: every 3rd frame) |
| **Voice Coach** | Toggle real-time audio feedback on/off |
| **Camera Index** | Switch between camera devices (0, 1, 2…) |

---

## 🔍 Explainable AI (XAI)

APEX uses **gradient-based attribution** to show which biomechanical features drove each prediction — not just a score, but *why*.

The XAI panel displays feature importance bars for inputs like:
- Knee flexion angle
- Hip drop asymmetry
- Trunk lean deviation
- Ankle dorsiflexion
- Stride length variance

This transparency is critical for coaches who need to understand *what to correct*, not just *that something is wrong*.

---

## 🩺 Injury Risk Model

The `InjuryRiskScorer` evaluates four biomechanical stress vectors per frame:

```
Knee Stress        ──▶ Excessive valgus collapse, hyperextension patterns
Hip Imbalance      ──▶ Left/right asymmetry in hip drop and rotation
Ankle Instability  ──▶ Pronation/supination deviation from baseline
Overstriding       ──▶ Heel strike far ahead of center of mass
```

Risk levels: `low` 🟢 → `medium` 🟡 → `high` 🔴 → `critical` 🚨

Visualized as both a **radar chart** and **bar chart** per session.

---

## 🎙️ Voice Coach

The built-in voice coach delivers real-time audio cues when:
- Fatigue exceeds the configured threshold
- Movement quality degrades to `Degraded` or `Critical`
- Injury risk elevates to `high` or `critical`
- Cadence drops outside optimal range

Enable/disable via the sidebar toggle. Works with any TTS-capable system.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI Framework** | Streamlit |
| **Pose Estimation** | MediaPipe Pose (33 keypoints) |
| **Multi-Athlete Detection** | YOLOv8 |
| **Sequence Modeling** | PyTorch LSTM |
| **Explainability** | Gradient-based attribution |
| **Video Processing** | OpenCV |
| **Visualization** | Plotly (custom dark sports-tech theme) |
| **Voice** | Python TTS (VoiceCoach) |

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| Black webcam screen | Try camera index `1` or `2` in sidebar |
| Another app has camera | Close Teams / Zoom / OBS before launching |
| Slow / laggy playback | Normal on CPU — analysis runs every 3rd frame by design |
| Voice coach silent | Toggle Voice Coach OFF then ON in sidebar |
| `streamlit` won't start (Windows Python 3.12) | Run `net stop winmgmt /y && net start winmgmt` in admin PowerShell, or downgrade to Python 3.11 |

---

## 📈 Roadmap

- [ ] Export session reports as PDF
- [ ] Sport-specific models (sprint, cycling, swimming)
- [ ] Team dashboard with multi-session aggregation
- [ ] REST API endpoint for integration with wearables
- [ ] Mobile-optimized UI

---

## 👤 Author

**Sandeep Kasiraju**
Data Scientist · ML Engineer · AWS Certified ML Engineer (MLA-C01)

[![Email](https://img.shields.io/badge/Email-sandeepkasiraju793%40gmail.com-00d2ff?style=flat-square&logo=gmail)](mailto:sandeepkasiraju793@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077b5?style=flat-square&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-2ed573?style=flat-square&logo=github)](https://github.com/yourusername)

---

<div align="center">

**Built with ⚡ and PyTorch**

*APEX — because performance is data.*

</div>
