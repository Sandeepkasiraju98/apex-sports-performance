import cv2
import numpy as np
import threading
import queue
import time
import os

from src.pose.extractor          import PoseExtractor
from src.features.engineer       import RunningFeatureEngineer
from src.features.stride_counter import StrideCounter
from src.features.injury_risk    import InjuryRiskScorer
from src.models.lstm_model       import PerformanceAnalyzer
from src.utils.voice_coach       import VoiceCoach


def find_camera_index() -> int:
    print("Detecting camera...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"Camera found at index {i}")
                return i
    print("No camera found - defaulting to index 0")
    return 0


class WebcamProcessor:
    """
    Windows-optimised webcam processor with frame skipping.

    DISPLAY : every frame        (smooth video)
    ANALYSIS: every Nth frame    (fast predictions)

    analyze_every=3 means 10fps analysis at 30fps camera.
    Fatigue and quality change over seconds not milliseconds
    so skipping 2 in 3 frames has zero meaningful accuracy loss.
    """

    def __init__(self,
                 model_path:    str  = None,
                 voice_enabled: bool = True,
                 camera_index:  int  = None,
                 analyze_every: int  = 3):
        self.model_path    = model_path
        self.camera_index  = camera_index
        self.analyze_every = analyze_every

        self.extractor      = PoseExtractor()
        self.feature_eng    = RunningFeatureEngineer()
        self.stride_counter = StrideCounter()
        self.injury_scorer  = InjuryRiskScorer()
        self.analyzer       = PerformanceAnalyzer(model_path)
        self.voice          = VoiceCoach(enabled=voice_enabled)

        self.frame_q = queue.Queue(maxsize=2)
        self.pred_q  = queue.Queue(maxsize=60)

        self.running       = False
        self._thread       = None
        self.fps_actual    = 0.0
        self.frame_count   = 0
        self.analyze_count = 0

        # Cache last prediction so every display frame
        # has something to overlay
        self._last_pred   = None
        self._last_stride = None
        self._last_injury = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True
        )
        self._thread.start()
        print(f"Webcam started — analyzing every "
              f"{self.analyze_every} frames.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        self.voice.stop()
        print("Webcam stopped.")

    def _run(self):
        idx = self.camera_index \
            if self.camera_index is not None \
            else find_camera_index()

        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Could not open camera {idx}")
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        self.feature_eng.reset()
        self.analyzer.reset()
        self.stride_counter.reset()
        self.injury_scorer.reset()

        t0        = time.time()
        fps_timer = time.time()
        fps_count = 0
        frame_id  = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            ts             = time.time() - t0
            should_analyze = (frame_id % self.analyze_every == 0)

            if should_analyze:
                # Full pipeline on this frame
                pose_frame = self.extractor.extract_frame(
                    frame, frame_id, ts
                )
                annotated = self.extractor.draw_pose(
                    frame.copy(), pose_frame
                )
                features = self.feature_eng.extract(pose_frame)
                pred     = self.analyzer.update(features)
                stride   = self.stride_counter.update(features)
                injury   = self.injury_scorer.update(features)

                # Cache for skipped frames
                self._last_pred   = pred
                self._last_stride = stride
                self._last_injury = injury
                self.analyze_count += 1

                # Voice only on analyzed frames
                self.voice.analyze_and_coach(
                    fatigue_score=pred.fatigue_score,
                    performance_score=pred.performance_score,
                    movement_quality=pred.movement_quality,
                    injury_risk=injury.overall_risk,
                    cadence=stride.cadence_spm,
                    stride_regularity=stride.stride_regularity
                )

                # Push prediction
                self.pred_q.put({
                    'timestamp':   round(ts, 2),
                    'fatigue':     pred.fatigue_score,
                    'performance': pred.performance_score,
                    'quality':     pred.movement_quality,
                    'cadence':     stride.cadence_spm,
                    'strides':     stride.total_strides,
                    'injury_risk': injury.overall_risk,
                    'risk_level':  pred.risk_level,
                })

            else:
                # Skip analysis — just use the raw frame
                annotated = frame.copy()

            # Always overlay cached predictions
            if self._last_pred is not None:
                annotated = self._draw_overlay(
                    annotated,
                    self._last_pred,
                    self._last_stride,
                    self._last_injury,
                    ts,
                    self.fps_actual
                )

            # Push to display queue
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            self._push_frame(rgb)

            fps_count        += 1
            frame_id         += 1
            self.frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.fps_actual = fps_count / elapsed
                fps_count = 0
                fps_timer = time.time()

        cap.release()

    def _push_frame(self, rgb):
        try:
            self.frame_q.put_nowait(rgb)
        except queue.Full:
            try:
                self.frame_q.get_nowait()
                self.frame_q.put_nowait(rgb)
            except:
                pass

    def _draw_overlay(self, frame, pred, stride,
                      injury, ts, fps):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 190),
                      (13, 17, 23), -1)
        cv2.addWeighted(overlay, 0.75, frame,
                        0.25, 0, frame)
        risk_colors = {
            'low':      (0,   255, 136),
            'medium':   (0,   184, 255),
            'high':     (77,  77,  255),
            'critical': (77,  77,  255),
        }
        col = risk_colors.get(pred.risk_level,
                              (255, 255, 255))
        lines = [
            (f"Time:     {ts:.1f}s  FPS:{fps:.0f}",
             (140, 140, 140)),
            (f"Fatigue:  {pred.fatigue_score:.0f}/100", col),
            (f"Perf:     {pred.performance_score:.0f}/100",
             (0, 229, 255)),
            (f"Quality:  {pred.movement_quality}",
             (180, 180, 255)),
            (f"Cadence:  {stride.cadence_spm:.0f} spm",
             (0, 255, 136)),
            (f"Strides:  {stride.total_strides}",
             (0, 255, 136)),
            (f"Inj Risk: {injury.overall_risk:.0f}/100",
             (77, 184, 255)),
            (f"Risk:     {pred.risk_level.upper()}", col),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text,
                (10, 24 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)
        return frame

    def get_latest_frame(self):
        try:
            return self.frame_q.get_nowait()
        except queue.Empty:
            return None

    def get_predictions(self) -> list:
        preds = []
        while not self.pred_q.empty():
            try:
                preds.append(self.pred_q.get_nowait())
            except queue.Empty:
                break
        return preds

    @property
    def is_running(self) -> bool:
        return self.running