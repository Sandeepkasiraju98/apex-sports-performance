import numpy as np
from collections import deque
from scipy.signal import find_peaks
from dataclasses import dataclass


@dataclass
class StrideMetrics:
    total_strides:      int
    cadence_spm:        float    # strides per minute
    avg_stride_time:    float    # seconds between strides
    stride_regularity:  float    # 0–1 (1 = very regular)
    current_pace:       str      # estimated pace category


class StrideCounter:
    """
    Detects strides by finding peaks in foot velocity signal.
    Uses a rolling buffer of foot velocities.
    """

    def __init__(self, fps: float = 30.0,
                 buffer_size: int = 150):
        self.fps          = fps
        self.buffer_size  = buffer_size

        self.l_vel_buffer = deque(maxlen=buffer_size)
        self.r_vel_buffer = deque(maxlen=buffer_size)
        self.stride_times = deque(maxlen=50)

        self.total_strides    = 0
        self.last_stride_frame = 0
        self.frame_count      = 0

    def update(self, features: np.ndarray) -> StrideMetrics:
        """
        features[8] = left foot velocity
        features[9] = right foot velocity
        """
        self.frame_count += 1
        l_vel = float(features[8])
        r_vel = float(features[9])

        self.l_vel_buffer.append(l_vel)
        self.r_vel_buffer.append(r_vel)

        # Detect peaks every 30 frames
        if len(self.l_vel_buffer) >= 60:
            self._detect_strides()

        return self._compute_metrics()

    def _detect_strides(self):
        signal = np.array(self.l_vel_buffer)

        # Normalize
        if signal.max() > 0:
            signal = signal / signal.max()

        peaks, _ = find_peaks(
            signal,
            height=0.3,
            distance=int(self.fps * 0.3),   # min 0.3s between strides
            prominence=0.2
        )

        if len(peaks) > 0:
            new_strides = len(peaks)
            self.total_strides += new_strides

            # Record stride intervals
            for i in range(len(peaks) - 1):
                interval = (peaks[i+1] - peaks[i]) / self.fps
                self.stride_times.append(interval)

            # Reset buffer after processing
            self.l_vel_buffer.clear()

    def _compute_metrics(self) -> StrideMetrics:
        # Cadence
        cadence = 0.0
        if self.stride_times:
            avg_interval = np.mean(list(self.stride_times))
            cadence = 60.0 / avg_interval if avg_interval > 0 else 0

        # Stride time
        avg_stride_time = float(np.mean(
            list(self.stride_times)
        )) if self.stride_times else 0.0

        # Regularity — inverse of coefficient of variation
        regularity = 0.0
        if len(self.stride_times) > 2:
            times = np.array(list(self.stride_times))
            cv = times.std() / (times.mean() + 1e-8)
            regularity = round(
                float(np.clip(1.0 - cv, 0, 1)), 3
            )

        # Pace category
        if cadence >= 170:
            pace = "Sprint 🔴"
        elif cadence >= 150:
            pace = "Fast Run 🟠"
        elif cadence >= 130:
            pace = "Jog 🟡"
        elif cadence > 0:
            pace = "Walk/Slow 🟢"
        else:
            pace = "Detecting..."

        return StrideMetrics(
            total_strides=self.total_strides,
            cadence_spm=round(cadence, 1),
            avg_stride_time=round(avg_stride_time, 3),
            stride_regularity=regularity,
            current_pace=pace
        )

    def reset(self):
        self.l_vel_buffer.clear()
        self.r_vel_buffer.clear()
        self.stride_times.clear()
        self.total_strides     = 0
        self.last_stride_frame = 0
        self.frame_count       = 0