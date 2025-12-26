from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import pickle
class GroupAwareScaler:
    def __init__(self, global_scaler=None):
        self.global_scaler = global_scaler or StandardScaler()
        self.feature_names = None
        self.group_col = None
        self.fitted = False

    def fit(self, df: pd.DataFrame, group_col: str, feature_cols: list):
        self.feature_names = feature_cols
        self.group_col = group_col
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

        rank_scaled = np.zeros_like(global_scaled)

        for i, feature in enumerate(self.feature_names):
            percentiles = np.zeros(len(df))
            for group_value, group_df in df.groupby(self.group_col):
                group_idx = df.index.get_indexer(group_df.index)
                vals = group_df[feature].values
                if len(vals) == 1:
                    p = np.array([0.0])
                else:
                    ranks = rankdata(vals, method="average")
                    p = (ranks - 1) / (len(vals) - 1)
                percentiles[group_idx] = p
            rank_scaled[:, i] = percentiles

        # Combine global_scaled and rank_scaled
        combined = np.concatenate([global_scaled, rank_scaled], axis=1)
        # return combined #TODO uncomment this line
        return rank_scaled #TODO - assumeed that only ranked features are used, and globally scaled are dropped

    def fit_transform(self, df: pd.DataFrame, group_col: str, feature_cols: list) -> np.ndarray:
        self.fit(df, group_col, feature_cols)
        return self.transform(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "global_scaler": self.global_scaler,
                "feature_names": self.feature_names,
                "group_col": self.group_col,
                "fitted": self.fitted
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.global_scaler = data["global_scaler"]
            self.feature_names = data["feature_names"]
            self.group_col = data["group_col"]
            self.fitted = data["fitted"]

    def get_feature_names_out(self) -> list:
        scaled = [f"{f}_scaled" for f in self.feature_names]
        ranked = [f"{f}_rank" for f in self.feature_names]
        # return scaled + ranked #TODO uncomment this line
        return ranked #TODO - assumeed that only ranked features are used, and globally scaled are dropped