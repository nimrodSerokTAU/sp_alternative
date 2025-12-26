from __future__ import annotations
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

from dl_model.config.config import OutputConfig

class ArtifactWriter:
    def __init__(self, cfg: OutputConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save_scaler(self, scaler, name: str):
        # GroupAwareScalerZ has .save(); sklearn scalers can be joblib dumped
        path = self.out_dir / name
        if hasattr(scaler, "save"):
            scaler.save(str(path))
        else:
            joblib.dump(scaler, str(path))

    def save_model(self, model: tf.keras.Model, name: str):
        path = self.out_dir / name
        model.save(str(path))

    def save_scaled_csv(
        self,
        X_scaled: np.ndarray,
        code: pd.Series,
        code1: pd.Series,
        y: pd.Series,
        filename: str,
        feature_names: list[str] | None = None
    ):
        df = pd.DataFrame(X_scaled, columns=feature_names if feature_names else None)
        df["code"] = code.reset_index(drop=True)
        df["code1"] = code1.reset_index(drop=True)
        df["class_label"] = y.reset_index(drop=True)
        df.to_csv(self.out_dir / filename, index=False)

    def save_predictions(self, code1: pd.Series, code: pd.Series, y_pred: np.ndarray, filename: str):
        df = pd.DataFrame({"code1": code1.values, "code": code.values, "predicted_score": y_pred})
        df.to_csv(self.out_dir / filename, index=False)

    def save_features_csv(self, df: pd.DataFrame, filename: str):
        df.to_csv(self.out_dir / filename, index=False)

def save_loss_plot(history, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, format="png")
    plt.close()

