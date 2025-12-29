from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, RobustScaler

from dl_model.config.config import FeatureConfig
from dl_model.data_processing.group_aware_zscore_scaler import GroupAwareScalerZ


def rank_percentile_per_group(y: pd.Series, groups: pd.Series) -> pd.Series:
    if not y.index.equals(groups.index):
        raise ValueError("y and groups must share index")
    df = pd.DataFrame({"y": y, "g": groups})
    out = pd.Series(index=y.index, dtype=np.float32)
    for gval, gdf in df.groupby("g"):
        vals = gdf["y"].values
        if len(vals) == 1:
            scaled = np.array([0.0], dtype=np.float32)
        else:
            r = rankdata(vals, method="average")
            scaled = ((r - 1) / (len(vals) - 1)).astype(np.float32)
        out.loc[gdf.index] = scaled
    return out


def zscore_per_group(y: pd.Series, groups: pd.Series) -> pd.Series:
    if not y.index.equals(groups.index):
        raise ValueError("y and groups must share index")
    df = pd.DataFrame({"y": y, "g": groups})
    out = pd.Series(index=y.index, dtype=np.float32)
    for gval, gdf in df.groupby("g"):
        vals = gdf["y"].values.astype(np.float32)
        mu, sigma = float(vals.mean()), float(vals.std())
        out.loc[gdf.index] = 0.0 if sigma == 0 else (vals - mu) / sigma
    return out.astype(np.float32)


class FeatureScaler:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.scaler = None
        self.feature_names_out: list[str] | None = None

    def fit_transform_X(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.cfg.scaler_type_features == "standard":
            self.scaler = StandardScaler()
            Xtr = self.scaler.fit_transform(X_train)
            Xte = self.scaler.transform(X_test)
            self.feature_names_out = list(X_train.columns)
            return Xtr.astype("float64"), Xte.astype("float64")

        if self.cfg.scaler_type_features in {"rank", "zscore"}:
            mode = "rank" if self.cfg.scaler_type_features == "rank" else "zscore"
            self.scaler = GroupAwareScalerZ(mode=mode, use_global=False, global_scaler=RobustScaler())
            Xtr = self.scaler.fit_transform(train_df, group_col="code1", feature_cols=X_train.columns)
            Xte = self.scaler.transform(test_df)
            self.feature_names_out = self.scaler.get_feature_names_out()
            return Xtr.astype("float64"), Xte.astype("float64")

        raise ValueError(f"Unknown scaler_type_features: {self.cfg.scaler_type_features}")

    def transform_y(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        groups_train: pd.Series,
        groups_test: pd.Series
    ) -> tuple[pd.Series, pd.Series]:

        if self.cfg.scaler_type_labels == "standard":
            return y_train, y_test
        if self.cfg.scaler_type_labels == "rank":
            return rank_percentile_per_group(y_train, groups_train), rank_percentile_per_group(y_test, groups_test)
        if self.cfg.scaler_type_labels == "zscore":
            return zscore_per_group(y_train, groups_train), zscore_per_group(y_test, groups_test)

        raise ValueError(f"Unknown scaler_type_labels: {self.cfg.scaler_type_labels}")
