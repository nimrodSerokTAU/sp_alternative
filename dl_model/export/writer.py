from __future__ import annotations
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde

from dl_model.config.config import OutputConfig

class OutputsWriter:
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

def save_correlation_plot(y_test, y_pred, loss_fn: str, mode: int, true_score_name: str, mse: float, rid: str, out_path: str) -> None:

    #CORRELATION PLOT 1
    out = Path(f"{out_path}/regression_results_{rid}_mode{mode}_{true_score_name}.png")

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    corr_coefficient, _ = pearsonr(y_test, y_pred)
    # mse = np.mean((y_test - y_pred) ** 2)
    plt.text(
        0.05, 0.95,
        f'Pearson Correlation: {corr_coefficient:.2f}, MSE: {mse:.6f}',
        transform=plt.gca().transAxes,
        fontsize=18,
        verticalalignment='top'
    )
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    if loss_fn == "mse":
        title = "Model 1"
    elif loss_fn == "custom_mse":
        title = "Model 2"
    else:
        title = "DL Model"
    plt.title(f'{title}: Predicted vs. True Values')
    plt.grid(True)
    plt.savefig(out, format='png')
    plt.close()

    #CORRELATION PLOT 2 WITH DENSITY
    out = Path(f"{out_path}/regression_results_{rid}_mode{mode}_{true_score_name}_w_density.png")
    kde = gaussian_kde([y_pred, y_test], bw_method=0.1)
    density = kde([y_test, y_pred])
    plt.figure(figsize=(8, 6))
    r, _ = pearsonr(y_pred, y_test)
    plt.text(0.65, 0.95, f'Pearson r = {r:.3f}',
             ha='right', va='top',
             transform=plt.gca().transAxes,
             fontsize=16, color='black', weight='bold', zorder=2)

    plt.ylim(bottom=min(y_test), top=max(y_test) * 1.1)
    scatter = plt.scatter(y_pred, y_test, c=density, cmap='plasma', edgecolors='none',
                          alpha=0.7)
    cbar = plt.colorbar(scatter, label='Density')
    cbar.set_label('Density', fontsize=18, weight='bold', labelpad=10)

    plt.xlabel('Predicted distance', fontsize=16, weight='bold', labelpad=10)
    plt.ylabel('d_seq distance ("true distance")', fontsize=16, weight='bold', labelpad=10)
    plt.tight_layout()
    plt.savefig(out, format='png')
    plt.close()
