import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PerformancePrediction:
    fatigue_score:       float    # 0–100 (higher = more fatigued)
    movement_quality:    str      # optimal / degraded / critical
    performance_score:   float    # 0–100 (higher = better)
    quality_confidence:  float    # 0–1
    risk_level:          str      # low / medium / high


class SportsLSTM(nn.Module):
    """
    Bidirectional LSTM for sports performance analysis.
    Input:  (batch, seq_len=30, features=24)
    Output: fatigue (1) + quality class (3) + performance (1)
    """

    def __init__(self,
                 input_size:  int = 24,
                 hidden_size: int = 128,
                 num_layers:  int = 2,
                 dropout:     float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        self.fatigue_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.quality_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)   # optimal / degraded / critical
        )

        self.performance_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        lstm_out, _ = self.lstm(x)

        # Attention pooling
        attn_weights = self.attention(lstm_out)
        context      = (attn_weights * lstm_out).sum(dim=1)

        fatigue     = self.fatigue_head(context)
        quality     = self.quality_head(context)
        performance = self.performance_head(context)

        return fatigue, quality, performance


class PerformanceAnalyzer:
    """
    Wraps the LSTM with a sliding window buffer.
    Call update() each frame — get predictions every 30 frames.
    """

    QUALITY_LABELS = ['Optimal', 'Degraded', 'Critical']
    WINDOW_SIZE    = 30

    def __init__(self, model_path: str = None,
                 device: str = None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model  = SportsLSTM().to(self.device)
        self.buffer = []
        self.last_prediction = None

        if model_path:
            self.load(model_path)
        else:
            print("No model loaded — using untrained weights for demo.")

        self.model.eval()

    def update(self, features: np.ndarray
               ) -> PerformancePrediction:
        """
        Add one frame of features to the buffer.
        Returns a prediction every WINDOW_SIZE frames,
        or the last prediction otherwise.
        """
        self.buffer.append(features.copy())

        if len(self.buffer) >= self.WINDOW_SIZE:
            window = np.array(
                self.buffer[-self.WINDOW_SIZE:]
            )
            self.last_prediction = self._predict(window)
            self.buffer = self.buffer[-self.WINDOW_SIZE // 2:]

        return self.last_prediction or self._default_prediction()

    def _predict(self, window: np.ndarray
                 ) -> PerformancePrediction:
        x = torch.FloatTensor(window).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fatigue_raw, quality_raw, perf_raw = self.model(x)

        fatigue     = float(fatigue_raw.squeeze()) * 100
        quality_idx = int(quality_raw.squeeze().argmax())
        quality_probs = torch.softmax(
            quality_raw.squeeze(), dim=0
        ).cpu().numpy()
        performance = float(perf_raw.squeeze()) * 100

        if fatigue > 70:
            risk = 'high'
        elif fatigue > 40:
            risk = 'medium'
        else:
            risk = 'low'

        return PerformancePrediction(
            fatigue_score=round(fatigue, 1),
            movement_quality=self.QUALITY_LABELS[quality_idx],
            performance_score=round(performance, 1),
            quality_confidence=round(
                float(quality_probs[quality_idx]), 3
            ),
            risk_level=risk
        )

    def _default_prediction(self) -> PerformancePrediction:
        return PerformancePrediction(
            fatigue_score=0.0,
            movement_quality='Optimal',
            performance_score=100.0,
            quality_confidence=1.0,
            risk_level='low'
        )

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        print(f"Model loaded from {path}")

    def reset(self):
        self.buffer          = []
        self.last_prediction = None