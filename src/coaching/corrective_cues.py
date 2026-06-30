from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Severity model
# ──────────────────────────────────────────────────────────────
class Severity(IntEnum):
    """Higher = more urgent. Used for sort order and color mapping."""
    INFO = 0
    NOTICE = 1
    WARNING = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return self.name.capitalize()

    @property
    def color(self) -> str:
        return {
            Severity.INFO:     "#2ed573",
            Severity.NOTICE:   "#00d2ff",
            Severity.WARNING:  "#ffa502",
            Severity.CRITICAL: "#ff4757",
        }[self]


@dataclass(frozen=True)
class Cue:
    """A single corrective recommendation."""
    code: str                 # stable id, e.g. "OVERSTRIDE"
    headline: str             # short label for the UI chip
    fix: str                  # the one-line "do this now"
    drill: str                # a concrete off-track drill
    why: str                  # the rationale (builds trust)
    severity: Severity
    metric_value: float       # the number that triggered it
    target: str               # the goal range, human-readable

    @property
    def color(self) -> str:
        return self.severity.color

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "headline": self.headline,
            "fix": self.fix,
            "drill": self.drill,
            "why": self.why,
            "severity": self.severity.label,
            "color": self.color,
            "metric_value": round(self.metric_value, 1),
            "target": self.target,
        }


# ──────────────────────────────────────────────────────────────
# Thresholds — single source of truth, easy to tune / unit-test
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CueThresholds:
    # Risk sub-scores are 0–100 (matching InjuryRiskScorer output)
    knee_warn: float = 40.0
    knee_crit: float = 65.0

    hip_warn: float = 35.0
    hip_crit: float = 60.0

    ankle_warn: float = 40.0
    ankle_crit: float = 65.0

    overstride_warn: float = 35.0
    overstride_crit: float = 60.0

    # Cadence in steps/min — recommended distance-running band ~170–185
    cadence_low: float = 168.0
    cadence_very_low: float = 158.0

    # Vertical oscillation in cm — efficient runners ~6–9 cm
    vosc_high: float = 10.0
    vosc_very_high: float = 12.5

    # Ground-contact asymmetry, % difference L vs R
    gct_asym_warn: float = 3.0
    gct_asym_crit: float = 6.0

    # Fatigue 0–100
    fatigue_warn: float = 70.0
    fatigue_crit: float = 85.0

    # Stride regularity 0–1 (1 = perfectly even)
    regularity_low: float = 0.75


DEFAULT_THRESHOLDS = CueThresholds()


# ──────────────────────────────────────────────────────────────
# The engine
# ──────────────────────────────────────────────────────────────
class CorrectiveCueEngine:
    """Stateless evaluator: metrics in, prioritized cues out."""

    def __init__(self, thresholds: CueThresholds = DEFAULT_THRESHOLDS):
        self.t = thresholds

    # The signature accepts everything you already compute. All args
    # are optional so you can call it with whatever you have on hand.
    def evaluate(
        self,
        *,
        knee_stress: Optional[float] = None,
        hip_imbalance: Optional[float] = None,
        ankle_instability: Optional[float] = None,
        overstriding: Optional[float] = None,
        cadence_spm: Optional[float] = None,
        vertical_oscillation_cm: Optional[float] = None,
        gct_asymmetry_pct: Optional[float] = None,
        fatigue_score: Optional[float] = None,
        stride_regularity: Optional[float] = None,
    ) -> list[Cue]:
        cues: list[Cue] = []
        t = self.t

        # ── Overstriding (highest-leverage running fault) ──
        if overstriding is not None and overstriding >= t.overstride_warn:
            crit = overstriding >= t.overstride_crit
            cues.append(Cue(
                code="OVERSTRIDE",
                headline="Overstriding",
                fix="Shorten your stride and let your foot land under your hips, "
                    "not out in front.",
                drill="High-cadence strides: 6 × 20 s at +8 spm over your current "
                      "rate, walk-back recovery.",
                why="Landing ahead of your center of mass acts as a brake on every "
                    "step and spikes loading through the knee.",
                severity=Severity.CRITICAL if crit else Severity.WARNING,
                metric_value=overstriding,
                target="foot strike under hips",
            ))

        # ── Cadence (the master lever — fixes several faults at once) ──
        if cadence_spm is not None and cadence_spm < t.cadence_low:
            very = cadence_spm < t.cadence_very_low
            bump = 8 if very else 5
            cues.append(Cue(
                code="CADENCE_LOW",
                headline="Cadence low",
                fix=f"Lift your step rate by ~{bump} spm. Quick, light steps — "
                    f"think 'hot pavement'.",
                drill="Metronome run: set a beeper to your target spm and hold it "
                      "for 4 × 3 min.",
                why="A higher cadence shortens stride, reduces overstriding and "
                    "vertical bounce, and lowers peak impact — all at once.",
                severity=Severity.WARNING if very else Severity.NOTICE,
                metric_value=cadence_spm,
                target="170–185 spm",
            ))

        # ── Vertical oscillation ──
        if (vertical_oscillation_cm is not None
                and vertical_oscillation_cm >= t.vosc_high):
            very = vertical_oscillation_cm >= t.vosc_very_high
            cues.append(Cue(
                code="VERT_OSC",
                headline="Bouncing",
                fix="Drive forward, not up. Keep the top of your head moving on a "
                    "level line.",
                drill="Low-ceiling runs: imagine a bar just above your head; 4 × 30 s "
                      "staying under it.",
                why="Vertical bounce is wasted energy — every cm up is a cm you have "
                    "to absorb on landing.",
                severity=Severity.WARNING if very else Severity.NOTICE,
                metric_value=vertical_oscillation_cm,
                target="6–9 cm",
            ))

        # ── Ground-contact asymmetry (L vs R) ──
        if (gct_asymmetry_pct is not None
                and gct_asymmetry_pct >= t.gct_asym_warn):
            crit = gct_asymmetry_pct >= t.gct_asym_crit
            cues.append(Cue(
                code="GCT_ASYM",
                headline="L/R asymmetry",
                fix="You're spending longer on one foot. Focus on even, rhythmic "
                    "contact side to side.",
                drill="Single-leg stability: 3 × 30 s holds + single-leg hops, "
                      "weaker side first.",
                why="Persistent ground-contact asymmetry concentrates load on one "
                    "limb and is a known overuse-injury signal.",
                severity=Severity.CRITICAL if crit else Severity.WARNING,
                metric_value=gct_asymmetry_pct,
                target="< 3% difference",
            ))

        # ── Hip imbalance ──
        if hip_imbalance is not None and hip_imbalance >= t.hip_warn:
            crit = hip_imbalance >= t.hip_crit
            cues.append(Cue(
                code="HIP_DROP",
                headline="Hip drop",
                fix="Keep your pelvis level — don't let the non-stance hip sag at "
                    "footstrike.",
                drill="Single-leg glute bridges 3 × 12 each side + side planks "
                      "3 × 30 s.",
                why="A dropping hip (weak glute medius) drives knee valgus and is a "
                    "common root cause of runner's knee and IT-band pain.",
                severity=Severity.CRITICAL if crit else Severity.WARNING,
                metric_value=hip_imbalance,
                target="level pelvis",
            ))

        # ── Knee stress ──
        if knee_stress is not None and knee_stress >= t.knee_warn:
            crit = knee_stress >= t.knee_crit
            cues.append(Cue(
                code="KNEE_LOAD",
                headline="Knee loading",
                fix="Soften your landing and keep a slight knee bend at contact — "
                    "run quietly.",
                drill="Eccentric step-downs 3 × 10 each leg; box squats to groove a "
                      "controlled descent.",
                why="High knee loading usually traces back to overstriding and a "
                    "stiff, heel-first landing.",
                severity=Severity.CRITICAL if crit else Severity.WARNING,
                metric_value=knee_stress,
                target="quiet, bent-knee landing",
            ))

        # ── Ankle instability ──
        if (ankle_instability is not None
                and ankle_instability >= t.ankle_warn):
            crit = ankle_instability >= t.ankle_crit
            cues.append(Cue(
                code="ANKLE_STAB",
                headline="Ankle instability",
                fix="Aim for a stable midfoot landing; avoid rolling in or out at "
                    "contact.",
                drill="Calf raises 3 × 15 + single-leg balance on an unstable "
                      "surface 3 × 30 s.",
                why="An unstable ankle leaks force on push-off and raises sprain and "
                    "tendon-overload risk.",
                severity=Severity.CRITICAL if crit else Severity.WARNING,
                metric_value=ankle_instability,
                target="stable midfoot strike",
            ))

        # ── Stride regularity ──
        if (stride_regularity is not None
                and stride_regularity < t.regularity_low):
            cues.append(Cue(
                code="IRREGULAR",
                headline="Uneven rhythm",
                fix="Settle into a steady, repeatable cadence — uneven steps often "
                    "mean creeping fatigue.",
                drill="Cadence-locked easy run with a metronome to re-groove an even "
                      "rhythm.",
                why="Stride-to-stride variability tends to climb as form breaks "
                    "down, raising mis-step and injury risk.",
                severity=Severity.NOTICE,
                metric_value=stride_regularity * 100,
                target="> 75% consistency",
            ))

        # ── Fatigue collapse (meta-cue — overrides to top when critical) ──
        if fatigue_score is not None and fatigue_score >= t.fatigue_warn:
            crit = fatigue_score >= t.fatigue_crit
            cues.append(Cue(
                code="FATIGUE",
                headline="Fatigue high",
                fix=("Ease off — form is degrading. Drop the pace or take a recovery "
                     "interval." if crit else
                     "Fatigue climbing. Hold form: tall posture, relaxed shoulders, "
                     "steady cadence."),
                drill="Recovery: 2–3 min easy jog or walk, reset posture, then "
                      "reassess before pushing again.",
                why="Most form faults and the injuries that follow appear once "
                    "fatigue sets in and mechanics break down.",
                severity=Severity.CRITICAL if crit else Severity.WARNING,
                metric_value=fatigue_score,
                target="< 70 to sustain form",
            ))

        # Sort: most urgent first; stable within a severity tier.
        cues.sort(key=lambda c: (-int(c.severity), c.code))
        return cues

    def top(self, *args, n: int = 3, **kwargs) -> list[Cue]:
        """Convenience: evaluate and return the n most urgent cues."""
        return self.evaluate(*args, **kwargs)[:n]


# ──────────────────────────────────────────────────────────────
# Stateful wrapper — suppresses repeat cues (for live/voice use)
# ──────────────────────────────────────────────────────────────
@dataclass
class CueSession:
    """
    Wraps the engine with a per-code cooldown so the same cue doesn't
    fire every frame. `tick` should be your monotonic analysis time
    (seconds). Returns only cues that are *newly* due.
    """
    engine: CorrectiveCueEngine = field(default_factory=CorrectiveCueEngine)
    cooldown_s: float = 12.0
    _last_fired: dict[str, float] = field(default_factory=dict)

    def update(self, *, tick: float, **metrics) -> list[Cue]:
        due: list[Cue] = []
        for cue in self.engine.evaluate(**metrics):
            last = self._last_fired.get(cue.code, -1e9)
            # Critical cues bypass half the cooldown — they matter more.
            cd = self.cooldown_s * (0.5 if cue.severity == Severity.CRITICAL
                                    else 1.0)
            if tick - last >= cd:
                self._last_fired[cue.code] = tick
                due.append(cue)
        return due

    def reset(self) -> None:
        self._last_fired.clear()
