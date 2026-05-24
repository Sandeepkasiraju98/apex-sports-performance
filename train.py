import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import os

from src.models.lstm_model import SportsLSTM

WINDOW_SIZE  = 30
FEATURE_SIZE = 24
N_SAMPLES    = 3000


def generate_data(n_samples=N_SAMPLES):
    """
    Generate synthetic running sessions.
    Fatigue increases over time, degrading movement quality.
    """
    print(f"Generating {n_samples} training sequences...")
    X, y_fatigue, y_quality, y_perf = [], [], [], []

    for _ in range(n_samples):
        fatigue_level = np.random.uniform(0, 1)

        # As fatigue increases:
        # - Joint angles become more erratic
        # - Symmetry decreases
        # - Velocities drop
        noise_scale = 0.1 + fatigue_level * 0.4

        seq = []
        for t in range(WINDOW_SIZE):
            frame = np.array([
                # Knee angles (120–160 deg, degrade with fatigue)
                150 - fatigue_level*20 + np.random.randn()*noise_scale*15,
                150 - fatigue_level*20 + np.random.randn()*noise_scale*15,
                # Hip angles
                160 - fatigue_level*15 + np.random.randn()*noise_scale*10,
                160 - fatigue_level*15 + np.random.randn()*noise_scale*10,
                # Elbow angles
                90  + np.random.randn()*noise_scale*20,
                90  + np.random.randn()*noise_scale*20,
                # Ankle angles
                100 - fatigue_level*10 + np.random.randn()*noise_scale*8,
                100 - fatigue_level*10 + np.random.randn()*noise_scale*8,
                # Velocities (drop with fatigue)
                0.05*(1-fatigue_level*0.5) + np.random.randn()*0.01,
                0.05*(1-fatigue_level*0.5) + np.random.randn()*0.01,
                0.04*(1-fatigue_level*0.4) + np.random.randn()*0.01,
                0.04*(1-fatigue_level*0.4) + np.random.randn()*0.01,
                # Symmetry (increases with fatigue = worse)
                fatigue_level*0.3 + np.random.randn()*0.05,
                fatigue_level*0.3 + np.random.randn()*0.05,
                fatigue_level*0.2 + np.random.randn()*0.03,
                # Posture
                fatigue_level*10  + np.random.randn()*5,
                0.3 + np.random.randn()*0.05,
                0.1 + np.random.randn()*0.02,
                fatigue_level*0.02 + np.random.randn()*0.005,
                # Foot heights
                0.3 - fatigue_level*0.1 + np.random.randn()*0.05,
                0.3 - fatigue_level*0.1 + np.random.randn()*0.05,
                0.5 + np.random.randn()*0.02,
                # Arm swing & shoulder
                0.1*(1-fatigue_level*0.4) + np.random.randn()*0.02,
                fatigue_level*0.05 + np.random.randn()*0.01,
            ], dtype=np.float32)
            seq.append(frame)

        X.append(seq)
        y_fatigue.append(fatigue_level)
        if fatigue_level < 0.35:
            y_quality.append(0)   # optimal
        elif fatigue_level < 0.70:
            y_quality.append(1)   # degraded
        else:
            y_quality.append(2)   # critical
        y_perf.append(1.0 - fatigue_level * 0.7)

    return (
        np.array(X, dtype=np.float32),
        np.array(y_fatigue, dtype=np.float32),
        np.array(y_quality, dtype=np.int64),
        np.array(y_perf,    dtype=np.float32),
    )


def train():
    mlflow.set_experiment("sports-performance-lstm")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    X, y_fat, y_qual, y_perf = generate_data()

    X_train, X_test, \
    f_train, f_test, \
    q_train, q_test, \
    p_train, p_test = train_test_split(
        X, y_fat, y_qual, y_perf,
        test_size=0.2, random_state=42
    )

    def to_loader(X, f, q, p, shuffle=True):
        ds = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(f),
            torch.LongTensor(q),
            torch.FloatTensor(p)
        )
        return DataLoader(ds, batch_size=64, shuffle=shuffle)

    train_loader = to_loader(X_train, f_train, q_train, p_train)
    test_loader  = to_loader(X_test,  f_test,  q_test,  p_test,
                             shuffle=False)

    model    = SportsLSTM().to(device)
    optim    = torch.optim.Adam(model.parameters(), lr=1e-3,
                                weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=3, factor=0.5
    )

    fat_loss_fn  = nn.MSELoss()
    qual_loss_fn = nn.CrossEntropyLoss()
    perf_loss_fn = nn.MSELoss()

    with mlflow.start_run(run_name="lstm_running_v1"):
        mlflow.log_params({
            "hidden_size": 128,
            "num_layers":  2,
            "window_size": WINDOW_SIZE,
            "epochs":      30,
            "batch_size":  64
        })

        for epoch in range(30):
            model.train()
            total_loss = 0

            for X_b, f_b, q_b, p_b in train_loader:
                X_b = X_b.to(device)
                f_b = f_b.to(device).unsqueeze(1)
                q_b = q_b.to(device)
                p_b = p_b.to(device).unsqueeze(1)

                optim.zero_grad()
                fat_pred, qual_pred, perf_pred = model(X_b)

                loss = (
                    fat_loss_fn(fat_pred, f_b) * 0.4 +
                    qual_loss_fn(qual_pred, q_b) * 0.4 +
                    perf_loss_fn(perf_pred, p_b) * 0.2
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optim.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)

            # Validation
            model.eval()
            val_loss = 0
            correct  = 0
            total    = 0

            with torch.no_grad():
                for X_b, f_b, q_b, p_b in test_loader:
                    X_b = X_b.to(device)
                    f_b = f_b.to(device).unsqueeze(1)
                    q_b = q_b.to(device)
                    p_b = p_b.to(device).unsqueeze(1)

                    fat_pred, qual_pred, perf_pred = model(X_b)
                    v_loss = (
                        fat_loss_fn(fat_pred, f_b) * 0.4 +
                        qual_loss_fn(qual_pred, q_b) * 0.4 +
                        perf_loss_fn(perf_pred, p_b) * 0.2
                    )
                    val_loss += v_loss.item()
                    correct  += (
                        qual_pred.argmax(1) == q_b
                    ).sum().item()
                    total += len(q_b)

            acc      = correct / total * 100
            avg_val  = val_loss / len(test_loader)

            mlflow.log_metrics({
                "train_loss": round(avg_loss, 4),
                "val_loss":   round(avg_val,  4),
                "val_acc":    round(acc, 2)
            }, step=epoch)

            print(f"Epoch {epoch+1:02d}/30 | "
                  f"Train: {avg_loss:.4f} | "
                  f"Val: {avg_val:.4f} | "
                  f"Quality Acc: {acc:.1f}%")

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/sports_lstm.pth")
        mlflow.pytorch.log_model(model, "model")
        print("\nModel saved to models/sports_lstm.pth")


if __name__ == "__main__":
    train()