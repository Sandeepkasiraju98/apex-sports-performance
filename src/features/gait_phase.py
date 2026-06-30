"""
Gait Phase Analyzer
───────────────────
Segments running gait into stance / flight phases per foot and derives
the metrics elite running analysis cares about but free tools rarely
surface:

    • ground contact time (GCT) per foot, ms
    • left/right GCT asymmetry, %
    • flight time & flight ratio
    • vertical oscillation, cm  (bounce of the center of mass)
    • duty factor  (fraction of the cycle in contact)

It works off the same per-frame pose stream you already feed the rest
of the pipeline, so it slots in beside StrideCounter and InjuryRiskScorer
without new model dependencies.

How contact is detected
-----------------------
Without a force plate we infer ground contact from kinematics: a foot
is "in contact" when its vertical velocity is near zero AND it sits low
relative to its own recent range (i.e. near the bottom of its travel).
This is the standard kinematic proxy used in markerless gait work. It's
approximate — so every output also carries a `confidence` you can show
in the UI rather than presenting noisy numbers as fact.

Inputs
------
`update(landmarks, t)` expects a landmarks object/dict exposing foot and
hip keypoints. To stay decoupled from your exact PoseExtractor schema,
landmark access goes through `_get`, which tolerates:
    - dict:        landmarks["left_ankle"] -> (x, y) or (x, y, conf)
    - attribute:   landmarks.left_ankle
    - index map:   pass `index_map=` to map names -> integer indices
Returns None for a missing/low-confidence point; the analyzer degrades
gracefully and lowers its confidence instead of crashing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Optional


# Normalized-coordinate assumption: y grows downward (image space),
# values roughly 0..1. The class auto-calibrates scale from hip width
# so absolute pixel size doesn't matter.

@dataclass
class GaitReport:
    contact_time_left_ms: float = 0.0
    contact_time_right_ms: float = 0.0
    gct_asymmetry_pct: float = 0.0
    flight_time_ms: float = 0.0
    flight_ratio: float = 0.0           # flight / total cycle
    vertical_oscillation_cm: float = 0.0
    duty_factor: float = 0.0            # contact / total cycle
    confidence: float = 0.0            # 0..1, gate UI display on this

    def as_dict(self) -> dict:
        return {
            "contact_time_left_ms": round(self.contact_time_left_ms, 1),
            "contact_time_right_ms": round(self.contact_time_right_ms, 1),
            "gct_asymmetry_pct": round(self.gct_asymmetry_pct, 2),
            "flight_time_ms": round(self.flight_time_ms, 1),
            "flight_ratio": round(self.flight_ratio, 3),
            "vertical_oscillation_cm": round(self.vertical_oscillation_cm, 2),
            "duty_factor": round(self.duty_factor, 3),
            "confidence": round(self.confidence, 2),
        }


class _FootTracker:
    """Tracks one foot: detects contact onsets/offsets, logs durations."""

    def __init__(self, window: int = 90):
        self.y_hist: deque[tuple[float, float]] = deque(maxlen=window)  # (t, y)
        self.in_contact = False
        self.contact_start_t: Optional[float] = None
        self.contact_durations_ms: deque[float] = deque(maxlen=12)
        self.flight_durations_ms: deque[float] = deque(maxlen=12)
        self._last_offset_t: Optional[float] = None

    def update(self, t: float, y: Optional[float], scale: float,
               vel_eps: float) -> None:
        if y is None:
            return
        self.y_hist.append((t, y))
        if len(self.y_hist) < 5:
            return

        # Recent vertical velocity (per second), sign-agnostic.
        (t0, y0), (t1, y1) = self.y_hist[-2], self.y_hist[-1]
        dt = max(t1 - t0, 1e-3)
        vel = abs(y1 - y0) / dt

        # Foot's own travel range over the window → "low" threshold.
        ys = [p[1] for p in self.y_hist]
        lo, hi = min(ys), max(ys)
        rng = max(hi - lo, 1e-6)
        # In image space larger y = lower on screen = closer to ground.
        low_zone = y1 >= lo + 0.72 * rng

        contact_now = (vel < vel_eps * scale) and low_zone

        if contact_now and not self.in_contact:
            # Contact onset. Close any open flight phase.
            if self._last_offset_t is not None:
                self.flight_durations_ms.append((t - self._last_offset_t) * 1000)
            self.in_contact = True
            self.contact_start_t = t
        elif not contact_now and self.in_contact:
            # Contact offset.
            if self.contact_start_t is not None:
                self.contact_durations_ms.append(
                    (t - self.contact_start_t) * 1000)
            self.in_contact = False
            self._last_offset_t = t

    @property
    def avg_contact_ms(self) -> float:
        return mean(self.contact_durations_ms) if self.contact_durations_ms else 0.0

    @property
    def avg_flight_ms(self) -> float:
        return mean(self.flight_durations_ms) if self.flight_durations_ms else 0.0

    @property
    def samples(self) -> int:
        return len(self.contact_durations_ms)


class GaitPhaseAnalyzer:
    """Per-frame gait segmentation → GaitReport."""

    # Reference: average adult hip width ≈ 0.30 m. Used to convert the
    # normalized hip-width unit into centimeters for vertical oscillation.
    HIP_WIDTH_M = 0.30

    def __init__(self, vel_eps: float = 0.25):
        # vel_eps scales the "near-zero velocity" contact threshold; it's
        # multiplied by the body scale so it adapts to subject size.
        self.vel_eps = vel_eps
        self.left = _FootTracker()
        self.right = _FootTracker()
        self.hip_y_hist: deque[float] = deque(maxlen=120)
        self.scale: float = 1.0
        self._scale_seen = False
        self._missing = 0
        self._total = 0

    def reset(self) -> None:
        self.__init__(vel_eps=self.vel_eps)

    # ── landmark access shim (decoupled from PoseExtractor schema) ──
    @staticmethod
    def _get(landmarks, name: str, index_map: Optional[dict]) -> Optional[tuple]:
        try:
            if index_map and name in index_map:
                pt = landmarks[index_map[name]]
            elif isinstance(landmarks, dict):
                pt = landmarks.get(name)
            else:
                pt = getattr(landmarks, name, None)
            if pt is None:
                return None
            # Accept (x, y) | (x, y, conf) | object with .x/.y/.visibility
            if hasattr(pt, "x") and hasattr(pt, "y"):
                conf = getattr(pt, "visibility", getattr(pt, "conf", 1.0))
                if conf is not None and conf < 0.3:
                    return None
                return (float(pt.x), float(pt.y))
            if len(pt) >= 3 and pt[2] is not None and pt[2] < 0.3:
                return None
            return (float(pt[0]), float(pt[1]))
        except Exception:
            return None

    def update(self, landmarks, t: float,
               index_map: Optional[dict] = None) -> GaitReport:
        self._total += 1
        g = self._get

        l_ank = g(landmarks, "left_ankle", index_map)
        r_ank = g(landmarks, "right_ankle", index_map)
        l_hip = g(landmarks, "left_hip", index_map)
        r_hip = g(landmarks, "right_hip", index_map)

        # Body scale from hip width (falls back to last known).
        if l_hip and r_hip:
            hw = ((l_hip[0] - r_hip[0]) ** 2 + (l_hip[1] - r_hip[1]) ** 2) ** 0.5
            if hw > 1e-4:
                self.scale = hw
                self._scale_seen = True
            hip_cy = (l_hip[1] + r_hip[1]) / 2
            self.hip_y_hist.append(hip_cy)
        else:
            self._missing += 1

        # Feet drive contact detection.
        self.left.update(t, l_ank[1] if l_ank else None, self.scale, self.vel_eps)
        self.right.update(t, r_ank[1] if r_ank else None, self.scale, self.vel_eps)

        if not l_ank or not r_ank:
            self._missing += 1

        return self._build_report()

    def _build_report(self) -> GaitReport:
        r = GaitReport()
        lc = self.left.avg_contact_ms
        rc = self.right.avg_contact_ms
        r.contact_time_left_ms = lc
        r.contact_time_right_ms = rc

        if lc > 0 and rc > 0:
            r.gct_asymmetry_pct = abs(lc - rc) / ((lc + rc) / 2) * 100

        flight = mean([
            v for v in (self.left.avg_flight_ms, self.right.avg_flight_ms) if v > 0
        ]) if (self.left.avg_flight_ms or self.right.avg_flight_ms) else 0.0
        r.flight_time_ms = flight

        avg_contact = mean([v for v in (lc, rc) if v > 0]) if (lc or rc) else 0.0
        cycle = avg_contact + flight
        if cycle > 0:
            r.flight_ratio = flight / cycle
            r.duty_factor = avg_contact / cycle

        # Vertical oscillation: peak-to-peak of hip center, scaled to cm.
        if len(self.hip_y_hist) >= 10 and self._scale_seen and self.scale > 1e-4:
            pp = max(self.hip_y_hist) - min(self.hip_y_hist)
            cm_per_unit = (self.HIP_WIDTH_M * 100.0) / self.scale
            r.vertical_oscillation_cm = pp * cm_per_unit

        # Confidence: blend of data sufficiency and tracking completeness.
        stride_conf = min(1.0, (self.left.samples + self.right.samples) / 8.0)
        miss_rate = self._missing / max(self._total, 1)
        track_conf = max(0.0, 1.0 - miss_rate)
        r.confidence = round(0.5 * stride_conf + 0.5 * track_conf, 3)
        return r
