from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from dl_model.config.config import DataConfig, FeatureConfig, OutputConfig

from dl_model.data_processing.io import read_features
from dl_model.data_processing.preprocess import DatasetPreprocessor
from dl_model.data_processing.features_selector import FeatureSelector
from dl_model.data_processing.scaling import rank_percentile_per_group, zscore_per_group
from dl_model.data_processing.group_aware_zscore_scaler import GroupAwareScalerZ

from dl_model.export.writer import OutputsWriter

from dl_model.config.constants import CODE_COL, GROUP_COL


@dataclass(frozen=True)
class PretrainedPredictConfig:
    # inputs
    features_file: str
    true_score_name: str

    # must match training
    mode: int = 1
    remove_correlated_features: bool = False
    corr_threshold: float = 0.90
    scaler_type_features: str = "standard"  # standard | rank | zscore
    scaler_type_labels: str = "standard"    # standard | rank | zscore

    # pretrained files
    model_path: str = ""
    scaler_path: str = ""

    # output
    out_dir: str = "../out"
    run_id: str = "0"
    save_predictions_csv: bool = True
    verbose: bool = True

    # optional evaluation
    compute_metrics_if_possible: bool = True


def _load_scaler(path: str, scaler_type_features: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scaler not found: {p}")

    if scaler_type_features in {"rank", "zscore"}:
        mode = "rank" if scaler_type_features == "rank" else "zscore"
        scaler = GroupAwareScalerZ(mode=mode, use_global=False)  # attrs overwritten by load()
        scaler.load(str(p))
        return scaler

    if scaler_type_features == "standard":
        return joblib.load(str(p))

    raise ValueError(f"Unknown scaler_type_features: {scaler_type_features}")


def _load_model(path: str, custom_objects: Optional[Dict[str, Any]] = None) -> tf.keras.Model:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    # return tf.keras.models.load_model(str(p), custom_objects=custom_objects, safe_mode=False)
    return tf.keras.models.load_model(str(p), custom_objects=custom_objects, safe_mode=False, compile=False)

class PretrainedPredictor:
    def __init__(self, cfg: PretrainedPredictConfig):
        self.cfg = cfg

        self.data_cfg = DataConfig(
            features_file=cfg.features_file,
            true_score_name=cfg.true_score_name,
            test_size=0.0,
        )

        self.feat_cfg = FeatureConfig(
            mode=cfg.mode,
            remove_correlated_features=cfg.remove_correlated_features,
            corr_threshold=cfg.corr_threshold,
            scaler_type_features=cfg.scaler_type_features,
            scaler_type_labels=cfg.scaler_type_labels,
        )

        self.out_cfg = OutputConfig(
            out_dir=cfg.out_dir,
            run_id=cfg.run_id,
            verbose=cfg.verbose,
            save_predictions_csv=cfg.save_predictions_csv,
            save_model=False, #hardcoded , not to save for pretrained
            save_scaled_csv=False, #hardcoded , not to save for pretrained
            save_plots=False, #hardcoded , not to save for pretrained
            save_features_csv=False, #hardcoded , not to save for pretrained
            save_scaler=False, #hardcoded , not to save for pretrained
        )

        self.writer = OutputsWriter(self.out_cfg)

    def run(self, custom_objects: Optional[Dict[str, Any]] = None) -> dict:
        if self.cfg.verbose:
            print(f"[pretrained] reading: {self.cfg.features_file}")

        # 1) read + preprocess
        df = read_features(self.cfg.features_file)
        df = DatasetPreprocessor(self.data_cfg).preprocess(df)

        if self.cfg.verbose:
            print(f"[pretrained] rows after preprocess: {len(df)}")

        # 2) feature selection
        selector = FeatureSelector(self.feat_cfg, target_col=self.cfg.true_score_name)
        X, y = selector.select(df)

        # 3) load scaler + scale X
        scaler = _load_scaler(self.cfg.scaler_path, self.cfg.scaler_type_features)

        if self.cfg.scaler_type_features == "standard":
            X_scaled = scaler.transform(X).astype("float64")
            feature_names_out = list(X.columns)
        else:
            X_scaled = scaler.transform(df).astype("float64")
            feature_names_out = scaler.get_feature_names_out()

        if self.cfg.verbose:
            print(f"[pretrained] loading model: {self.cfg.model_path}")
        model = _load_model(self.cfg.model_path, custom_objects=custom_objects)

        y_pred = model.predict(X_scaled, verbose=0).reshape(-1).astype("float64")

        pred_filename = f"prediction_pretrained_{self.cfg.run_id}_mode{self.cfg.mode}_{self.cfg.true_score_name}.csv"
        if self.cfg.save_predictions_csv:
            self.writer.save_predictions(df[GROUP_COL], df[CODE_COL], y_pred, pred_filename)

        if self.cfg.verbose:
            print(f"[pretrained] wrote: {Path(self.cfg.out_dir) / pred_filename}")

        metrics: dict[str, float] = {}
        if self.cfg.compute_metrics_if_possible and self.cfg.true_score_name in df.columns:
            #only if we have a true label can check the quality of prediction
            y_eval = y.copy()

            if self.cfg.scaler_type_labels == "rank":
                y_eval = rank_percentile_per_group(y_eval, df[GROUP_COL])
            elif self.cfg.scaler_type_labels == "zscore":
                y_eval = zscore_per_group(y_eval, df[GROUP_COL])

            mse = float(mean_squared_error(y_eval, y_pred))
            r = float(pearsonr(y_eval, y_pred)[0]) if y_eval.nunique() > 1 else float("nan")
            metrics = {"mse": mse, "pearson_r": r}

            if self.cfg.verbose:
                print(f"[pretrained] metrics: mse={mse:.6f}, pearson_r={r:.4f}")

        return {
            "predictions_csv": str(Path(self.cfg.out_dir) / pred_filename),
            "n_rows": int(len(df)),
            "metrics": metrics,
            "feature_names_out": feature_names_out,
        }
