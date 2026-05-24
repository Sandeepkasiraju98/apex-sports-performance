import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Dict
from src.pose.extractor import PoseExtractor, PoseFrame
from src.features.engineer import RunningFeatureEngineer
from src.models.lstm_model import PerformanceAnalyzer


@dataclass
class AthleteState:
    athlete_id:   int
    bbox:         tuple      # x1, y1, x2, y2
    pose_frame:   PoseFrame
    features:     np.ndarray
    prediction:   object
    color:        tuple      # BGR color for this athlete


ATHLETE_COLORS = [
    (0, 229, 255),    # cyan
    (255, 77, 109),   # red
    (0, 255, 136),    # green
    (255, 184, 0),    # amber
    (180, 97, 255),   # purple
]


class MultiAthleteTracker:
    """
    Detects multiple athletes using YOLOv8,
    then runs individual pose + performance
    analysis on each detected person.
    """

    def __init__(self, model_path: str = None):
        print("Loading YOLOv8...")
        self.detector   = YOLO('yolov8n.pt')
        self.extractors: Dict[int, PoseExtractor] = {}
        self.engineers:  Dict[int, RunningFeatureEngineer] = {}
        self.analyzers:  Dict[int, PerformanceAnalyzer] = {}
        self.model_path = model_path
        self.frame_count = 0
        print("YOLOv8 loaded.")

    def _get_or_create_athlete(self, aid: int):
        if aid not in self.extractors:
            self.extractors[aid] = PoseExtractor()
            self.engineers[aid]  = RunningFeatureEngineer()
            self.analyzers[aid]  = PerformanceAnalyzer(
                self.model_path
            )

    def process_frame(
        self,
        frame: np.ndarray
    ) -> tuple[np.ndarray, List[AthleteState]]:
        """
        Detect all athletes in frame, run pose
        and performance analysis on each.
        Returns annotated frame + athlete states.
        """
        self.frame_count += 1
        athletes = []

        # Detect people with YOLOv8
        results = self.detector(
            frame,
            classes=[0],    # class 0 = person
            conf=0.4,
            verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            return frame, athletes

        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes[:5]):  # max 5 athletes
            x1, y1, x2, y2 = map(int, box)
            aid = i
            color = ATHLETE_COLORS[i % len(ATHLETE_COLORS)]

            self._get_or_create_athlete(aid)

            # Crop athlete region
            pad   = 20
            crop  = frame[
                max(0, y1-pad):min(frame.shape[0], y2+pad),
                max(0, x1-pad):min(frame.shape[1], x2+pad)
            ]

            if crop.size == 0:
                continue

            # Extract pose on crop
            pose_frame = self.extractors[aid].extract_frame(
                crop,
                frame_id=self.frame_count,
                timestamp=self.frame_count / 30.0
            )

            # Extract features
            features = self.engineers[aid].extract(
                pose_frame
            )

            # Get performance prediction
            prediction = self.analyzers[aid].update(
                features
            )

            athlete = AthleteState(
                athlete_id=aid,
                bbox=(x1, y1, x2, y2),
                pose_frame=pose_frame,
                features=features,
                prediction=prediction,
                color=color
            )
            athletes.append(athlete)

            # Draw bounding box + label
            frame = self._draw_athlete(
                frame, athlete
            )

        return frame, athletes

    def _draw_athlete(
        self,
        frame: np.ndarray,
        athlete: AthleteState
    ) -> np.ndarray:
        x1, y1, x2, y2 = athlete.bbox
        color = athlete.color
        pred  = athlete.prediction

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      color, 2)

        # Header background
        header_h = 80
        overlay  = frame.copy()
        cv2.rectangle(
            overlay,
            (x1, max(0, y1-header_h)),
            (x2, y1),
            (13, 17, 23), -1
        )
        cv2.addWeighted(
            overlay, 0.8, frame, 0.2, 0, frame
        )

        # Athlete ID
        cv2.putText(
            frame,
            f"Athlete {athlete.athlete_id + 1}",
            (x1+6, max(15, y1-60)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )

        # Fatigue bar
        fat = pred.fatigue_score / 100
        bar_w = x2 - x1
        cv2.rectangle(
            frame,
            (x1, max(0, y1-35)),
            (x2, max(0, y1-22)),
            (30, 30, 40), -1
        )
        fat_color = (
            (77, 77, 255) if fat > 0.7 else
            (0, 184, 255) if fat > 0.4 else
            (0, 255, 136)
        )
        cv2.rectangle(
            frame,
            (x1, max(0, y1-35)),
            (x1 + int(bar_w*fat),
             max(0, y1-22)),
            fat_color, -1
        )
        cv2.putText(
            frame,
            f"Fatigue {pred.fatigue_score:.0f}%",
            (x1+6, max(15, y1-38)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, fat_color, 1
        )

        # Quality label
        q_colors = {
            'Optimal':  (0, 255, 136),
            'Degraded': (0, 184, 255),
            'Critical': (77, 77, 255)
        }
        q_col = q_colors.get(
            pred.movement_quality, (255, 255, 255)
        )
        cv2.putText(
            frame,
            pred.movement_quality,
            (x1+6, max(15, y1-12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, q_col, 1
        )

        return frame

    def reset(self):
        self.extractors.clear()
        self.engineers.clear()
        self.analyzers.clear()
        self.frame_count = 0