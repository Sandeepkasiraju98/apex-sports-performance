import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SessionSummary:
    session_id:      str
    date:            str
    duration_s:      float
    total_strides:   int
    avg_fatigue:     float
    avg_performance: float
    avg_cadence:     float
    peak_fatigue:    float
    min_performance: float
    injury_risk:     float
    quality_dist:    dict
    alerts_count:    int


class SessionManager:
    """
    Saves, loads, and compares running sessions.
    """

    def __init__(self,
                 sessions_dir: str = "data/sessions"):
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)

    def save_session(
        self,
        timestamps:    list,
        fatigue:       list,
        performance:   list,
        quality:       list,
        strides:       list,
        cadence:       list,
        injury_risk:   list,
        alerts:        list
    ) -> str:
        session_id = datetime.now().strftime(
            "session_%Y%m%d_%H%M%S"
        )

        df = pd.DataFrame({
            'time':        timestamps,
            'fatigue':     fatigue,
            'performance': performance,
            'quality':     quality,
            'strides':     strides,
            'cadence':     cadence,
            'injury_risk': injury_risk,
        })

        # Summary
        quality_dist = pd.Series(quality).value_counts().to_dict()
        summary = SessionSummary(
            session_id=session_id,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            duration_s=float(max(timestamps)) if timestamps else 0,
            total_strides=int(max(strides)) if strides else 0,
            avg_fatigue=round(float(np.mean(fatigue)), 1),
            avg_performance=round(float(np.mean(performance)), 1),
            avg_cadence=round(float(np.mean(
                [c for c in cadence if c > 0]
            )), 1) if any(c > 0 for c in cadence) else 0.0,
            peak_fatigue=round(float(np.max(fatigue)), 1),
            min_performance=round(float(np.min(performance)), 1),
            injury_risk=round(float(np.mean(injury_risk)), 1),
            quality_dist=quality_dist,
            alerts_count=len(alerts)
        )

        # Save CSV
        csv_path = os.path.join(
            self.sessions_dir, f"{session_id}.csv"
        )
        df.to_csv(csv_path, index=False)

        # Save summary JSON
        json_path = os.path.join(
            self.sessions_dir, f"{session_id}_summary.json"
        )
        with open(json_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        return session_id

    def list_sessions(self) -> list:
        summaries = []
        for f in sorted(os.listdir(self.sessions_dir)):
            if f.endswith('_summary.json'):
                with open(
                    os.path.join(self.sessions_dir, f)
                ) as fp:
                    summaries.append(json.load(fp))
        return summaries

    def load_session(self,
                     session_id: str) -> Optional[pd.DataFrame]:
        path = os.path.join(
            self.sessions_dir, f"{session_id}.csv"
        )
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def compare_sessions(
        self,
        session_a: str,
        session_b: str
    ) -> dict:
        df_a = self.load_session(session_a)
        df_b = self.load_session(session_b)

        if df_a is None or df_b is None:
            return {}

        def stats(df):
            return {
                'avg_fatigue':     round(df['fatigue'].mean(), 1),
                'avg_performance': round(df['performance'].mean(), 1),
                'peak_fatigue':    round(df['fatigue'].max(), 1),
                'min_performance': round(df['performance'].min(), 1),
                'avg_cadence':     round(
                    df[df['cadence'] > 0]['cadence'].mean(), 1
                ) if (df['cadence'] > 0).any() else 0,
            }

        return {
            'session_a': stats(df_a),
            'session_b': stats(df_b),
            'df_a':      df_a,
            'df_b':      df_b,
        }