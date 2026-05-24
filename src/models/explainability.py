import numpy as np
import cv2
import torch
from typing import List
import plotly.graph_objects as go


# MediaPipe landmark positions (normalized x,y)
# for drawing the skeleton diagram
SKELETON_JOINTS = {
    'nose':           (0.50, 0.05),
    'left_shoulder':  (0.35, 0.20),
    'right_shoulder': (0.65, 0.20),
    'left_elbow':     (0.25, 0.35),
    'right_elbow':    (0.75, 0.35),
    'left_wrist':     (0.20, 0.50),
    'right_wrist':    (0.80, 0.50),
    'left_hip':       (0.38, 0.50),
    'right_hip':      (0.62, 0.50),
    'left_knee':      (0.35, 0.68),
    'right_knee':     (0.65, 0.68),
    'left_ankle':     (0.33, 0.85),
    'right_ankle':    (0.67, 0.85),
    'left_foot':      (0.30, 0.95),
    'right_foot':     (0.70, 0.95),
}

SKELETON_CONNECTIONS = [
    ('left_shoulder',  'right_shoulder'),
    ('left_shoulder',  'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow',     'left_wrist'),
    ('right_elbow',    'right_wrist'),
    ('left_shoulder',  'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip',       'right_hip'),
    ('left_hip',       'left_knee'),
    ('right_hip',      'right_knee'),
    ('left_knee',      'left_ankle'),
    ('right_knee',     'right_ankle'),
    ('left_ankle',     'left_foot'),
    ('right_ankle',    'right_foot'),
]

# Which features map to which joints
FEATURE_JOINT_MAP = {
    0:  ['left_knee'],
    1:  ['right_knee'],
    2:  ['left_hip'],
    3:  ['right_hip'],
    4:  ['left_elbow'],
    5:  ['right_elbow'],
    6:  ['left_ankle'],
    7:  ['right_ankle'],
    8:  ['left_foot'],
    9:  ['right_foot'],
    10: ['left_wrist'],
    11: ['right_wrist'],
    12: ['left_knee',  'right_knee'],
    13: ['left_hip',   'right_hip'],
    14: ['left_foot',  'right_foot'],
    15: ['nose',       'left_hip', 'right_hip'],
    16: ['left_shoulder', 'right_shoulder',
         'left_hip',      'right_hip'],
    17: ['nose'],
    18: ['left_hip',   'right_hip'],
    19: ['left_foot'],
    20: ['right_foot'],
    21: ['left_hip',   'right_hip'],
    22: ['left_wrist', 'right_wrist'],
    23: ['left_shoulder', 'right_shoulder'],
}


class GradientExplainer:
    """
    Computes gradient-based feature importance
    showing which body parts most influenced
    the fatigue prediction.
    """

    def __init__(self, model):
        self.model = model

    def compute_importance(
        self,
        window: np.ndarray
    ) -> np.ndarray:
        """
        Returns feature importance vector (24,)
        using input gradients w.r.t. fatigue output.
        """
        x = torch.FloatTensor(window).unsqueeze(0)
        x.requires_grad_(True)

        self.model.model.eval()
        fatigue, _, _ = self.model.model(
            x.to(self.model.device)
        )
        fatigue.sum().backward()

        # Gradient magnitude averaged over time steps
        importance = x.grad.abs().squeeze(0).mean(0)\
            .detach().cpu().numpy()

        # Normalize to 0–1
        if importance.max() > 0:
            importance = importance / importance.max()

        return importance

    def build_skeleton_heatmap(
        self,
        importance: np.ndarray
    ) -> go.Figure:
        """
        Draws an annotated skeleton where each joint
        is colored by its contribution to fatigue.
        """
        # Compute per-joint importance
        joint_importance = {j: 0.0
                            for j in SKELETON_JOINTS}

        for feat_idx, joints in FEATURE_JOINT_MAP.items():
            if feat_idx < len(importance):
                val = float(importance[feat_idx])
                for joint in joints:
                    joint_importance[joint] = max(
                        joint_importance[joint], val
                    )

        fig = go.Figure()

        # Draw connections
        for j1, j2 in SKELETON_CONNECTIONS:
            x1, y1 = SKELETON_JOINTS[j1]
            x2, y2 = SKELETON_JOINTS[j2]
            imp = (joint_importance[j1] +
                   joint_importance[j2]) / 2

            # Color by importance
            r = int(255 * imp)
            g = int(255 * (1 - imp) * 0.5)
            b = int(255 * (1 - imp))
            line_color = f'rgb({r},{g},{b})'

            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[1-y1, 1-y2],
                mode='lines',
                line=dict(color=line_color,
                          width=3 + imp * 6),
                hoverinfo='skip',
                showlegend=False
            ))

        # Draw joints
        for joint, (x, y) in SKELETON_JOINTS.items():
            imp = joint_importance[joint]
            r   = int(255 * imp)
            g   = int(255 * (1 - imp) * 0.5)
            b   = int(255 * (1 - imp))
            color = f'rgb({r},{g},{b})'
            size  = 12 + imp * 20

            fig.add_trace(go.Scatter(
                x=[x], y=[1-y],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(color='white', width=1)
                ),
                text=[f"{imp:.2f}"],
                textposition='middle right',
                textfont=dict(size=9, color='#94a3b8'),
                hovertemplate=(
                    f"<b>{joint}</b><br>"
                    f"Importance: {imp:.3f}<extra></extra>"
                ),
                showlegend=False
            ))

        # Colorbar
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale='RdBu_r',
                showscale=True,
                cmin=0, cmax=1,
                colorbar=dict(
                    title='Fatigue<br>Contribution',
                    titlefont=dict(color='#94a3b8'),
                    tickfont=dict(color='#94a3b8'),
                    thickness=12,
                    len=0.7
                )
            ),
            hoverinfo='skip',
            showlegend=False
        ))

        fig.update_layout(
            title=dict(
                text='Fatigue Driver Heatmap',
                font=dict(color='#94a3b8', size=14)
            ),
            height=480,
            paper_bgcolor='#0a0e1a',
            plot_bgcolor='#0d1117',
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                range=[-0.1, 1.1]
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                range=[-0.1, 1.1],
                scaleanchor='x',
                scaleratio=1
            ),
            margin=dict(l=20, r=60, t=40, b=20)
        )

        return fig

    def top_contributors(
        self,
        importance: np.ndarray,
        n: int = 5
    ) -> list:
        feature_names = [
            'Left knee angle',   'Right knee angle',
            'Left hip angle',    'Right hip angle',
            'Left elbow angle',  'Right elbow angle',
            'Left ankle angle',  'Right ankle angle',
            'Left foot velocity','Right foot velocity',
            'Left wrist vel',    'Right wrist vel',
            'Knee symmetry',     'Hip symmetry',
            'Stride symmetry',   'Trunk lean',
            'Shoulder alignment','Head position',
            'Vertical oscillation','Left foot height',
            'Right foot height', 'Hip height',
            'Arm swing',         'Shoulder drop',
        ]
        idx = np.argsort(importance)[::-1][:n]
        return [
            {
                'feature':    feature_names[i],
                'importance': round(float(importance[i]), 3)
            }
            for i in idx
        ]