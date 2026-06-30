from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import mean
from typing import Optional


# Joints that must be visible for running analysis to be meaningful.
KEY_JOINTS = (
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


class TrackLevel(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    LOST = "lost"

    @property
    def color(self) -> str:
        return {
            TrackLevel.GOOD: "#2ed573",
            TrackLevel.FAIR: "#00d2ff",
            TrackLevel.POOR: "#ffa502",
            TrackLevel.LOST: "#ff4757",
        }[self]


@dataclass
class TrackingStatus:
    level: TrackLevel
    score: float           # 0..1 rolling confidence
    reason: str            # short UI string
    joints_visible: int    # this frame
    is_reliable: bool      # True if scores should be shown/logged

    def as_dict(self) -> dict:
        return {
            "level": self.level.value,
            "color": self.level.color,
            "score": round(self.score, 3),
            "reason": self.reason,
            "joints_visible": self.joints_visible,
            "is_reliable": self.is_reliable,
        }


class TrackingConfidenceMonitor:
    """Rolling pose-quality estimator."""

    def __init__(
        self,
        window: int = 20,
        reliable_threshold: float = 0.6,
        min_joints: int = 5,
        jump_limit: float = 0.12,   # normalized units per frame
    ):
        self.window = window
        self.reliable_threshold = reliable_threshold
        self.min_joints = min_joints
        self.jump_limit = jump_limit
        self._scores: deque[float] = deque(maxlen=window)
        self._prev: dict[str, tuple] = {}
        self._frames_since_good = 0

    def reset(self) -> None:
        self._scores.clear()
        self._prev.clear()
        self._frames_since_good = 0

    @staticmethod
    def _get(landmarks, name, index_map):
        try:
            if index_map and name in index_map:
                pt = landmarks[index_map[name]]
            elif isinstance(landmarks, dict):
                pt = landmarks.get(name)
            else:
                pt = getattr(landmarks, name, None)
            if pt is None:
                return None, 0.0
            if hasattr(pt, "x") and hasattr(pt, "y"):
                conf = getattr(pt, "visibility", getattr(pt, "conf", 1.0))
                return (float(pt.x), float(pt.y)), float(conf if conf is not None else 1.0)
            conf = float(pt[2]) if len(pt) >= 3 and pt[2] is not None else 1.0
            return (float(pt[0]), float(pt[1])), conf
        except Exception:
            return None, 0.0

    def update(self, landmarks, index_map: Optional[dict] = None) -> TrackingStatus:
        vises: list[float] = []
        present = 0
        jumps: list[float] = []
        cur: dict[str, tuple] = {}

        for j in KEY_JOINTS:
            pt, conf = self._get(landmarks, j, index_map)
            if pt is not None and conf >= 0.3:
                present += 1
                vises.append(conf)
                cur[j] = pt
                if j in self._prev:
                    p = self._prev[j]
                    d = ((pt[0] - p[0]) ** 2 + (pt[1] - p[1]) ** 2) ** 0.5
                    jumps.append(d)
            # missing joint contributes 0 visibility implicitly

        self._prev = cur

        # ── component scores ──
        # 1) coverage: fraction of key joints present
        coverage = present / len(KEY_JOINTS)
        # 2) visibility: mean model confidence on present joints
        visibility = mean(vises) if vises else 0.0
        # 3) stability: penalize big teleport jumps
        if jumps:
            worst = max(jumps)
            stability = max(0.0, 1.0 - (worst / self.jump_limit))
        else:
            stability = 1.0 if present else 0.0

        # Weighted blend gives the headline number, but we also gate on
        # the weakest axis: a track that's complete and steady but low
        # *confidence* (blur) should not read "solid". Multiplying by a
        # softened visibility/stability floor enforces that without
        # over-punishing a single marginal joint.
        blend = 0.40 * coverage + 0.40 * visibility + 0.20 * stability
        gate = min(
            1.0,
            (0.4 + 0.6 * visibility),   # blur pulls everything down
            (0.5 + 0.5 * stability),    # jitter pulls everything down
        )
        frame_score = blend * gate
        self._scores.append(frame_score)
        rolling = mean(self._scores) if self._scores else 0.0

        # Track "lost" streak for the reason string.
        if present < self.min_joints:
            self._frames_since_good += 1
        else:
            self._frames_since_good = 0

        level, reason = self._classify(
            rolling, present, coverage, visibility, stability)
        reliable = (rolling >= self.reliable_threshold
                    and present >= self.min_joints)

        return TrackingStatus(
            level=level,
            score=rolling,
            reason=reason,
            joints_visible=present,
            is_reliable=reliable,
        )

    def _classify(self, score, present, coverage, visibility, stability):
        if present < self.min_joints:
            return TrackLevel.LOST, f"Tracking lost — only {present}/6 joints visible"
        if score >= 0.78:
            return TrackLevel.GOOD, "Tracking solid"
        if score >= 0.55:
            # Name the weakest link so the reason is actionable.
            return TrackLevel.FAIR, self._weak_link(coverage, visibility, stability)
        return TrackLevel.POOR, self._weak_link(coverage, visibility, stability)

    @staticmethod
    def _weak_link(coverage, visibility, stability):
        worst = min(
            ("coverage", coverage),
            ("visibility", visibility),
            ("stability", stability),
            key=lambda kv: kv[1],
        )[0]
        return {
            "coverage": "Athlete partly out of frame",
            "visibility": "Low pose confidence — lighting or blur",
            "stability": "Jittery tracking — fast motion or occlusion",
        }[worst]

    @property
    def rolling_score(self) -> float:
        return mean(self._scores) if self._scores else 0.0
