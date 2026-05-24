import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode
import numpy as np
from dataclasses import dataclass
from typing import Optional
import urllib.request
import os


# MediaPipe landmark indices for running analysis
LANDMARKS = {
    'nose':           0,
    'left_shoulder':  11,
    'right_shoulder': 12,
    'left_elbow':     13,
    'right_elbow':    14,
    'left_wrist':     15,
    'right_wrist':    16,
    'left_hip':       23,
    'right_hip':      24,
    'left_knee':      25,
    'right_knee':     26,
    'left_ankle':     27,
    'right_ankle':    28,
    'left_heel':      29,
    'right_heel':     30,
    'left_foot':      31,
    'right_foot':     32,
}

# Skeleton connections (MediaPipe Pose indices)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(27,31),(28,32),(29,31),(30,32),
]

# Model file — downloaded once to local cache
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)


def _ensure_model():
    """Download the .task model file if not already cached."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe pose model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


@dataclass
class PoseFrame:
    frame_id:   int
    timestamp:  float
    keypoints:  np.ndarray   # shape (33, 3) — x, y, z
    visibility: np.ndarray   # shape (33,)
    valid:      bool


class PoseExtractor:
    """
    Extracts 33 body keypoints per frame using MediaPipe Pose Landmarker
    (Tasks API, compatible with mediapipe >= 0.10.13).
    Works with webcam feed and uploaded video files.
    """

    def __init__(self, min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5):
        _ensure_model()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            num_poses=1,
        )
        self.landmarker   = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms  = 0   # tracks last timestamp sent to MediaPipe

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_result(self, result, frame_id: int,
                      timestamp: float) -> PoseFrame:
        """Convert a PoseLandmarkerResult into a PoseFrame."""
        if not result.pose_landmarks:
            return PoseFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                keypoints=np.zeros((33, 3)),
                visibility=np.zeros(33),
                valid=False,
            )

        landmarks  = result.pose_landmarks[0]
        keypoints  = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        visibility = np.array([lm.visibility        for lm in landmarks])

        return PoseFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            keypoints=keypoints,
            visibility=visibility,
            valid=True,
        )

    def _to_mp_image(self, frame: np.ndarray) -> mp.Image:
        """Convert a BGR numpy frame to a MediaPipe Image."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_frame(self, frame: np.ndarray,
                      frame_id: int = 0,
                      timestamp: float = 0.0) -> PoseFrame:
        """Extract keypoints from a single BGR frame."""
        mp_image = self._to_mp_image(frame)
        # Guarantee monotonically increasing ms timestamp.
        # If the extractor is reused across Streamlit reruns the raw
        # frame-derived timestamp would reset to 0 and MediaPipe throws.
        ts_ms = max(int(timestamp * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        return self._parse_result(result, frame_id, timestamp)

    def draw_pose(self, frame: np.ndarray,
                  pose_frame: PoseFrame) -> np.ndarray:
        """Draw skeleton overlay on frame using OpenCV directly."""
        if not pose_frame.valid:
            return frame

        h, w = frame.shape[:2]
        keypoints = pose_frame.keypoints

        for start_idx, end_idx in POSE_CONNECTIONS:
            x1 = int(keypoints[start_idx, 0] * w)
            y1 = int(keypoints[start_idx, 1] * h)
            x2 = int(keypoints[end_idx,   0] * w)
            y2 = int(keypoints[end_idx,   1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

        for i in range(len(keypoints)):
            x = int(keypoints[i, 0] * w)
            y = int(keypoints[i, 1] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        return frame

    def process_video(self, video_path: str,
                      max_frames: Optional[int] = None):
        """
        Generator — yields (annotated_frame, PoseFrame)
        for every frame in a video file.
        """
        cap      = cv2.VideoCapture(video_path)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_id = 0
        self._last_ts_ms = 0   # reset for each new video

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_id >= max_frames:
                break

            timestamp  = frame_id / fps
            pose_frame = self.extract_frame(frame, frame_id, timestamp)
            annotated  = self.draw_pose(frame.copy(), pose_frame)

            yield annotated, pose_frame
            frame_id += 1

        cap.release()

    def process_webcam(self):
        """
        Generator — yields (annotated_frame, PoseFrame)
        from live webcam.
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_id = 0
        self._last_ts_ms = 0   # reset for each new webcam session

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp  = frame_id / fps
            pose_frame = self.extract_frame(frame, frame_id, timestamp)
            annotated  = self.draw_pose(frame.copy(), pose_frame)

            yield annotated, pose_frame
            frame_id += 1

        cap.release()

    def close(self):
        """Release the landmarker resources."""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()