from __future__ import annotations
import logging
import pandas as pd

from dl_model.config.config import DataConfig
from dl_model.config.constants import CODE_COL, GROUP_COL, ALIGNER_COL, TAXA_COL

logger = logging.getLogger(__name__)


def assign_aligner(code: str, code1: str) -> str:
    code_l = str(code).lower()
    not_mafft = ["muscle", "prank", "_true.fas", "true_tree.txt", "bali_phy", "baliphy", "original"]

    if code == code1:
        return "true"
    if not any(sub in code_l for sub in not_mafft):
        return "mafft"
    if "muscle" in code_l:
        return "muscle"
    if "prank" in code_l:
        return "prank"
    if "bali_phy" in code_l or "baliphy" in code_l:
        return "baliphy"
    return "true"


class DatasetPreprocessor:
    def __init__(self, configuration: DataConfig):
        self.configuration = configuration

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df[ALIGNER_COL] = [assign_aligner(c, c1) for c, c1 in zip(df[CODE_COL], df[GROUP_COL])]

        if self.configuration.true_score_name == "RF_phangorn_norm":
            df = df[df[self.configuration.true_score_name] != "ERROR"].copy()
            df[self.configuration.true_score_name] = df[self.configuration.true_score_name].astype(float)

        df = self._handle_duplicates(df)

        if self.configuration.true_score_name != "RF_phangorn_norm":
            df = self._balance(df)

        df = df.dropna()

        return df

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.configuration.deduplicated:
            return df.drop_duplicates(subset=[c for c in df.columns if c != CODE_COL])

        df2 = df.drop_duplicates(subset=[c for c in df.columns if c != CODE_COL])
        problematic_codes = (
            df2.groupby(GROUP_COL)
               .filter(lambda x: len(x) < self.configuration.min_rows_per_code_after_dedup)[GROUP_COL]
               .unique()
        )
        if len(problematic_codes) > 0:
            logger.info("Removing %d problematic codes due to duplicates. Example: %s",
                        len(problematic_codes), problematic_codes[:10])
        return df[~df[GROUP_COL].isin(problematic_codes)].copy()

    def _balance(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.configuration.empirical:
            threshold = self.configuration.number_of_msas_threshold_empirical
            keep_codes = df[GROUP_COL].value_counts()
            keep = keep_codes[keep_codes >= threshold].index
            out = df[df[GROUP_COL].isin(keep)].copy()
            return out

        threshold = self.configuration.number_of_msas_threshold_simulated
        counts = df[GROUP_COL].value_counts()
        frequent = counts[counts >= threshold].index
        filtered = df[df[GROUP_COL].isin(frequent)].copy()

        code1_taxa_counts = (
            filtered[[GROUP_COL, TAXA_COL]]
            .drop_duplicates()
            .groupby(TAXA_COL)
            .size()
        )
        min_code1_count = int(code1_taxa_counts.min())

        selected = []
        for taxa, group in filtered.groupby(TAXA_COL):
            valid_codes = group[GROUP_COL].unique()
            if len(valid_codes) >= min_code1_count:
                sampled = (
                    pd.Series(valid_codes)
                      .sample(n=min_code1_count, random_state=self.configuration.random_state)
                      .tolist()
                )
                selected.extend(sampled)
            else:
                logger.warning("Skipping taxa=%s only %d valid codes (<%d)",
                               taxa, len(valid_codes), min_code1_count)

        selected_set = set(selected)

        logger.info("Remaining codes after balancing: %d", len(selected_set))
        logger.info("Codes after balancing: %s", list(selected_set))

        return df[df[GROUP_COL].isin(selected_set)].copy()
