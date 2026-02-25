from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import pickle
class GroupAwareScalerZ:
    def __init__(self, mode: str = "rank", use_global: bool = False, global_scaler=None):
        """
        mode: 'rank'   → percentile rank per group
              'zscore' → per-group z-score
        use_global: if True, use global mean/std for zscore mode (for stability)
        global_scaler: pre-fitted scaler for global scaling (if use_global is True)
        """
        self.mode = mode
        self.use_global = use_global
        self.global_scaler = global_scaler or StandardScaler()
        self.feature_names = None
        self.group_col = None
        self.fitted = False

    def fit(self, df: pd.DataFrame, group_col: str, feature_cols: list):
        self.feature_names = feature_cols
        self.group_col = group_col
        # if self.use_global and self.mode == "zscore":
        #     self.global_scaler.fit(df[feature_cols])
        self.global_scaler.fit(df[feature_cols])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted.")

        df = df.copy()

        # Drop index level if group_col is both in index and columns
        if self.group_col in df.index.names and self.group_col in df.columns:
            df.index = df.index.droplevel(self.group_col)
        elif self.group_col in df.index.names:
            df = df.reset_index()

        global_scaled = self.global_scaler.transform(df[self.feature_names])

        scaled_array = np.zeros_like(global_scaled)

        for i, feature in enumerate(self.feature_names):
            scaled_feature = np.zeros(len(df))
            for group_value, group_df in df.groupby(self.group_col):
                group_idx = df.index.get_indexer(group_df.index)
                vals = group_df[feature].values

                if self.mode == "rank":
                    if len(vals) == 1:
                        scaled_feature[group_idx] = 0.0
                    else:
                        ranks = rankdata(vals, method="average")
                        scaled_feature[group_idx] = (ranks - 1) / (len(vals) - 1)

                elif self.mode == "zscore":
                    if self.use_global:
                        mean = self.global_scaler.mean_[i]
                        std = np.sqrt(self.global_scaler.var_[i])
                    else:
                        mean = np.mean(vals)
                        std = np.std(vals)
                    scaled_feature[group_idx] = 0.0 if std == 0 else (vals - mean) / std
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

            scaled_array[:, i] = scaled_feature

        return scaled_array

    def fit_transform(self, df: pd.DataFrame, group_col: str, feature_cols: list) -> np.ndarray:
        self.fit(df, group_col, feature_cols)
        return self.transform(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "use_global": self.use_global,
                "global_scaler": self.global_scaler,
                "feature_names": self.feature_names,
                "group_col": self.group_col,
                "fitted": self.fitted,
                "mode": self.mode
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.use_global = data["use_global"]
            self.global_scaler = data["global_scaler"]
            self.feature_names = data["feature_names"]
            self.group_col = data["group_col"]
            self.fitted = data["fitted"]
            self.mode = data["mode"] or "rank"

    def get_feature_names_out(self) -> list:
        scaled = [f"{f}_scaled" for f in self.feature_names]
        ranked = [f"{f}_{self.mode}" for f in self.feature_names]
        # return scaled + ranked
        return ranked #only ranked features are used

