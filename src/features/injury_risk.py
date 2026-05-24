import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List


@dataclass
class InjuryRiskReport:
    overall_risk:       float    # 0–100
    risk_level:         str      # low / medium / high / critical
    flags:              List[str]
    knee_stress:        float
    hip_imbalance:      float
    ankle_instability:  float
    overstriding:       float
    recommendations:    List[str]


class InjuryRiskScorer:
    """
    Computes injury risk from biomechanical asymmetries
    and threshold violations across a rolling window.
    """

    # Thresholds from sports science literature
    THRESHOLDS = {
        'knee_asymmetry':   15.0,   # degrees
        'hip_asymmetry':    12.0,   # degrees
        'trunk_lean':       20.0,   # degrees forward
        'ankle_asymmetry':  10.0,   # degrees
        'stride_asymmetry': 0.15,   # normalized
        'shoulder_drop':    0.05,   # normalized
    }

    def __init__(self, window: int = 60):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, features: np.ndarray) -> InjuryRiskReport:
        self.history.append(features.copy())

        if len(self.history) < 10:
            return self._empty_report()

        return self._compute_risk()

    def _compute_risk(self) -> InjuryRiskReport:
        arr = np.array(self.history)
        flags           = []
        recommendations = []
        risk_scores     = []

        # ── Knee stress ──
        l_knee = arr[:, 0]   # left knee angle
        r_knee = arr[:, 1]   # right knee angle
        knee_asym = float(np.mean(np.abs(l_knee - r_knee)))
        knee_stress = min(
            knee_asym / self.THRESHOLDS['knee_asymmetry'] * 100, 100
        )
        risk_scores.append(knee_stress * 0.30)

        if knee_asym > self.THRESHOLDS['knee_asymmetry']:
            flags.append(
                f"⚠️ Knee asymmetry: {knee_asym:.1f}° "
                f"(threshold {self.THRESHOLDS['knee_asymmetry']}°)"
            )
            recommendations.append(
                "Check for muscle imbalance — single-leg "
                "strength exercises recommended"
            )

        # ── Hip imbalance ──
        l_hip = arr[:, 2]
        r_hip = arr[:, 3]
        hip_asym = float(np.mean(np.abs(l_hip - r_hip)))
        hip_imbalance = min(
            hip_asym / self.THRESHOLDS['hip_asymmetry'] * 100, 100
        )
        risk_scores.append(hip_imbalance * 0.25)

        if hip_asym > self.THRESHOLDS['hip_asymmetry']:
            flags.append(
                f"⚠️ Hip imbalance: {hip_asym:.1f}° "
                f"(threshold {self.THRESHOLDS['hip_asymmetry']}°)"
            )
            recommendations.append(
                "Hip flexor tightness suspected — "
                "stretching and glute activation drills"
            )

        # ── Ankle instability ──
        l_ankle = arr[:, 6]
        r_ankle = arr[:, 7]
        ankle_asym = float(np.mean(np.abs(l_ankle - r_ankle)))
        ankle_instability = min(
            ankle_asym / self.THRESHOLDS['ankle_asymmetry'] * 100,
            100
        )
        risk_scores.append(ankle_instability * 0.20)

        if ankle_asym > self.THRESHOLDS['ankle_asymmetry']:
            flags.append(
                f"⚠️ Ankle instability: {ankle_asym:.1f}°"
            )
            recommendations.append(
                "Ankle proprioception training recommended"
            )

        # ── Overstriding ──
        trunk_lean = float(np.mean(np.abs(arr[:, 15])))
        overstriding = min(
            trunk_lean / self.THRESHOLDS['trunk_lean'] * 100, 100
        )
        risk_scores.append(overstriding * 0.15)

        if trunk_lean > self.THRESHOLDS['trunk_lean']:
            flags.append(
                f"⚠️ Excessive trunk lean: {trunk_lean:.1f}°"
            )
            recommendations.append(
                "Reduce forward lean — focus on upright posture "
                "and core engagement"
            )

        # ── Shoulder drop ──
        shoulder_drop = float(np.mean(arr[:, 23]))
        if shoulder_drop > self.THRESHOLDS['shoulder_drop']:
            flags.append(
                f"⚠️ Shoulder drop detected — "
                f"fatigue indicator"
            )
            recommendations.append(
                "Upper body fatigue detected — "
                "arm swing correction drills"
            )
            risk_scores.append(20)
        else:
            risk_scores.append(0)

        overall = round(min(sum(risk_scores), 100), 1)

        if overall >= 70:
            level = 'critical'
        elif overall >= 45:
            level = 'high'
        elif overall >= 20:
            level = 'medium'
        else:
            level = 'low'

        if not flags:
            recommendations.append(
                "✅ Biomechanics look good — maintain current form"
            )

        return InjuryRiskReport(
            overall_risk=overall,
            risk_level=level,
            flags=flags,
            knee_stress=round(knee_stress, 1),
            hip_imbalance=round(hip_imbalance, 1),
            ankle_instability=round(ankle_instability, 1),
            overstriding=round(overstriding, 1),
            recommendations=recommendations
        )

    def _empty_report(self) -> InjuryRiskReport:
        return InjuryRiskReport(
            overall_risk=0.0,
            risk_level='low',
            flags=[],
            knee_stress=0.0,
            hip_imbalance=0.0,
            ankle_instability=0.0,
            overstriding=0.0,
            recommendations=["Collecting data..."]
        )

    def reset(self):
        self.history.clear()