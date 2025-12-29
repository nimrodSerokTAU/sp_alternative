from __future__ import annotations
import numpy as np
import pandas as pd
import logging

from dl_model.config.config import FeatureConfig
from dl_model.config.constants import DROP_COLS_EXTENDED, COLUMNS_TO_CHOOSE_MODE3

logger = logging.getLogger(__name__)


class FeatureSelector:
    DROP_COLS = DROP_COLS_EXTENDED
    COLUMNS_TO_CHOOSE = COLUMNS_TO_CHOOSE_MODE3
 
    def __init__(self, cfg: FeatureConfig, target_col: str):
        self.cfg = cfg
        self.target_col = target_col

    def select(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y = df[self.target_col]

        if self.cfg.mode == 1:
            drop = [c for c in self.DROP_COLS if c in df.columns]
            X = df.drop(columns=drop)
        elif self.cfg.mode == 3:
            X = df[self.COLUMNS_TO_CHOOSE].copy()
        else:
            raise ValueError(f"Unsupported mode: {self.cfg.mode}")

        if self.cfg.remove_correlated_features:
            X = self._drop_correlated(X)

        return X, y

    def _drop_correlated(self, X: pd.DataFrame) -> pd.DataFrame:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > self.cfg.corr_threshold)]
        if to_drop:
            logger.info("Dropping correlated features: %s", to_drop)
        return X.drop(columns=to_drop)
