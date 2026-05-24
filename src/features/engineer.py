import numpy as np
from src.pose.extractor import PoseFrame


def angle_between(a: np.ndarray,
                  b: np.ndarray,
                  c: np.ndarray) -> float:
    """
    Calculate joint angle at point b
    given three points a, b, c.
    Returns angle in degrees (0–180).
    """
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class RunningFeatureEngineer:
    """
    Extracts biomechanically meaningful features
    from raw MediaPipe keypoints for running analysis.

    Output: 24-dimensional feature vector per frame.
    """

    def __init__(self):
        self.prev_keypoints = None
        self.feature_names  = self._get_feature_names()

    def _get_feature_names(self):
        return [
            # Joint angles (8)
            'left_knee_angle',
            'right_knee_angle',
            'left_hip_angle',
            'right_hip_angle',
            'left_elbow_angle',
            'right_elbow_angle',
            'left_ankle_angle',
            'right_ankle_angle',
            # Velocities (4)
            'left_foot_velocity',
            'right_foot_velocity',
            'left_wrist_velocity',
            'right_wrist_velocity',
            # Symmetry (3)
            'knee_angle_symmetry',
            'hip_angle_symmetry',
            'stride_symmetry',
            # Posture (4)
            'trunk_lean_angle',
            'shoulder_hip_alignment',
            'head_position',
            'vertical_oscillation',
            # Cadence proxies (3)
            'left_foot_height',
            'right_foot_height',
            'hip_height',
            # Fatigue proxy (2)
            'arm_swing_amplitude',
            'shoulder_drop',
        ]

    def extract(self, pose_frame: PoseFrame) -> np.ndarray:
        """
        Extract 24 features from a single pose frame.
        Returns zero vector if pose is invalid.
        """
        if not pose_frame.valid:
            self.prev_keypoints = None
            return np.zeros(24)

        kp = pose_frame.keypoints

        # ── Shortcuts ──
        l_shoulder = kp[11, :2]
        r_shoulder = kp[12, :2]
        l_elbow    = kp[13, :2]
        r_elbow    = kp[14, :2]
        l_wrist    = kp[15, :2]
        r_wrist    = kp[16, :2]
        l_hip      = kp[23, :2]
        r_hip      = kp[24, :2]
        l_knee     = kp[25, :2]
        r_knee     = kp[26, :2]
        l_ankle    = kp[27, :2]
        r_ankle    = kp[28, :2]
        l_heel     = kp[29, :2]
        r_heel     = kp[30, :2]
        l_foot     = kp[31, :2]
        r_foot     = kp[32, :2]
        nose       = kp[0,  :2]

        # ── Joint angles ──
        l_knee_angle  = angle_between(l_hip,   l_knee,  l_ankle)
        r_knee_angle  = angle_between(r_hip,   r_knee,  r_ankle)
        l_hip_angle   = angle_between(l_shoulder, l_hip, l_knee)
        r_hip_angle   = angle_between(r_shoulder, r_hip, r_knee)
        l_elbow_angle = angle_between(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = angle_between(r_shoulder, r_elbow, r_wrist)
        l_ankle_angle = angle_between(l_knee, l_ankle, l_foot)
        r_ankle_angle = angle_between(r_knee, r_ankle, r_foot)

        # ── Velocities (compared to previous frame) ──
        if self.prev_keypoints is not None:
            prev = self.prev_keypoints
            l_foot_vel  = float(np.linalg.norm(
                kp[31, :2] - prev[31, :2]))
            r_foot_vel  = float(np.linalg.norm(
                kp[32, :2] - prev[32, :2]))
            l_wrist_vel = float(np.linalg.norm(
                kp[15, :2] - prev[15, :2]))
            r_wrist_vel = float(np.linalg.norm(
                kp[16, :2] - prev[16, :2]))
        else:
            l_foot_vel = r_foot_vel = l_wrist_vel = r_wrist_vel = 0.0

        # ── Symmetry ──
        knee_sym   = abs(l_knee_angle  - r_knee_angle)  / 180.0
        hip_sym    = abs(l_hip_angle   - r_hip_angle)   / 180.0
        stride_sym = abs(
            np.linalg.norm(l_foot - l_hip) -
            np.linalg.norm(r_foot - r_hip)
        )

        # ── Posture ──
        mid_shoulder  = (l_shoulder + r_shoulder) / 2
        mid_hip       = (l_hip + r_hip) / 2
        trunk_vec     = mid_shoulder - mid_hip
        trunk_lean    = float(np.degrees(
            np.arctan2(trunk_vec[0], -trunk_vec[1])
        ))

        shoulder_hip_align = abs(
            (l_shoulder[1] + r_shoulder[1]) / 2 -
            (l_hip[1]      + r_hip[1])      / 2
        )

        head_pos = float(nose[1] - mid_shoulder[1])

        vert_osc = 0.0
        if self.prev_keypoints is not None:
            prev_mid_hip = (
                self.prev_keypoints[23, 1] +
                self.prev_keypoints[24, 1]
            ) / 2
            vert_osc = abs(mid_hip[1] - prev_mid_hip)

        # ── Cadence proxies ──
        l_foot_height = float(1.0 - l_foot[1])
        r_foot_height = float(1.0 - r_foot[1])
        hip_height    = float(1.0 - mid_hip[1])

        # ── Fatigue proxies ──
        arm_swing = abs(l_wrist[1] - r_wrist[1])
        shoulder_drop = abs(l_shoulder[1] - r_shoulder[1])

        self.prev_keypoints = kp.copy()

        return np.array([
            l_knee_angle,  r_knee_angle,
            l_hip_angle,   r_hip_angle,
            l_elbow_angle, r_elbow_angle,
            l_ankle_angle, r_ankle_angle,
            l_foot_vel,    r_foot_vel,
            l_wrist_vel,   r_wrist_vel,
            knee_sym,      hip_sym,      stride_sym,
            trunk_lean,    shoulder_hip_align,
            head_pos,      vert_osc,
            l_foot_height, r_foot_height, hip_height,
            arm_swing,     shoulder_drop,
        ], dtype=np.float32)

    def reset(self):
        self.prev_keypoints = None