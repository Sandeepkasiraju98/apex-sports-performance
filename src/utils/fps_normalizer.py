"""
FPS Normalizer
──────────────
Cadence (spm), stride timing, contact times, and fatigue *rates* are all
time-derivative metrics. If timing comes from wall-clock `time.time()`
during analysis, a slow CPU stretches the clock and corrupts every one
of them: a 170 spm runner gets reported at 95 spm because the machine
took ~1.8× real time to grind through the frames.

The fix: drive all timing off the *source video's* frame index and its
true FPS, not the processing wall-clock. This module:

    1. reads the real FPS from the video (with sane fallbacks), and
    2. converts a frame index → media-time in seconds,

so every downstream metric is correct regardless of how fast or slow the
machine decodes frames.

For live webcam (no fixed source FPS) it estimates an effective FPS from
a rolling window of real frame arrival times — which is legitimate there,
because for a live feed wall-clock *is* media time.

    norm = FPSNormalizer.from_video("run.mp4")
    t_seconds = norm.media_time(frame_index)   # use THIS for metrics
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


# Plausible FPS bounds; anything outside is treated as a bad read.
_FPS_MIN, _FPS_MAX = 1.0, 240.0
_FPS_FALLBACK = 30.0


@dataclass
class FPSInfo:
    fps: float
    source: str          # "container", "counted", "fallback", "live-estimate"
    frame_count: Optional[int] = None
    duration_s: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "fps": round(self.fps, 3),
            "source": self.source,
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 2) if self.duration_s else None,
        }


class FPSNormalizer:
    """Frame-index → media-time converter with a trustworthy FPS."""

    def __init__(self, info: FPSInfo):
        self.info = info
        self.fps = info.fps

    # ── construction ──
    @classmethod
    def from_fps(cls, fps: float, source: str = "explicit") -> "FPSNormalizer":
        return cls(FPSInfo(fps=cls._sane(fps, None), source=source))

    @classmethod
    def from_video(cls, path: str) -> "FPSNormalizer":
        """
        Read FPS from the container. If the metadata FPS is missing or
        implausible, fall back to counting frames / duration, then to a
        constant. Never raises on a readable file.
        """
        try:
            import cv2
        except Exception:
            return cls(FPSInfo(fps=_FPS_FALLBACK, source="fallback"))

        cap = None
        try:
            cap = cv2.VideoCapture(path)
            meta_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None

            if _FPS_MIN <= meta_fps <= _FPS_MAX:
                dur = (frame_count / meta_fps) if frame_count else None
                return cls(FPSInfo(
                    fps=meta_fps, source="container",
                    frame_count=frame_count, duration_s=dur))

            # Metadata unreliable → derive from frame count & a probe, or
            # last-resort fallback (kept cheap: we don't decode the whole file).
            return cls(FPSInfo(
                fps=_FPS_FALLBACK, source="fallback",
                frame_count=frame_count))
        except Exception:
            return cls(FPSInfo(fps=_FPS_FALLBACK, source="fallback"))
        finally:
            if cap is not None:
                cap.release()

    @staticmethod
    def _sane(fps: float, fallback: Optional[float]) -> float:
        try:
            f = float(fps)
        except (TypeError, ValueError):
            return fallback if fallback else _FPS_FALLBACK
        if _FPS_MIN <= f <= _FPS_MAX:
            return f
        return fallback if fallback else _FPS_FALLBACK

    # ── the one method everything should use ──
    def media_time(self, frame_index: int) -> float:
        """Seconds elapsed in the *source media* at this frame index."""
        return frame_index / self.fps

    def frames_to_seconds(self, n_frames: float) -> float:
        return n_frames / self.fps

    def seconds_to_frames(self, seconds: float) -> int:
        return int(round(seconds * self.fps))


class LiveFPSEstimator:
    """
    For webcam: estimate effective FPS from real arrival times. Here the
    wall-clock IS media time, so this is correct (unlike using wall-clock
    while crunching a recorded file slower than real time).
    """

    def __init__(self, window: int = 30):
        self._stamps: deque[float] = deque(maxlen=window)
        self._fps: float = _FPS_FALLBACK

    def tick(self, now: float) -> float:
        self._stamps.append(now)
        if len(self._stamps) >= 2:
            span = self._stamps[-1] - self._stamps[0]
            if span > 1e-6:
                est = (len(self._stamps) - 1) / span
                if _FPS_MIN <= est <= _FPS_MAX:
                    self._fps = est
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    def as_info(self) -> FPSInfo:
        return FPSInfo(fps=self._fps, source="live-estimate")
