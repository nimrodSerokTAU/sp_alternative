import numpy as np
import pandas as pd

from classes.msa_basic_stats import BasicStats
from utils import calc_percentile


class EntropyStats(BasicStats):
    entropy_mean: float
    entropy_median: float
    entropy_var: float
    entropy_pct_25: float
    entropy_pct_75: float
    entropy_min: float
    entropy_max: float
    constant_sites_pct: float
    n_unique_sites: int

    def __init__(self, code: str, taxa_num: int, msa_length: int):
        super().__init__(code, taxa_num, msa_length,
                         [
                             'code',
                             'constant_sites_pct', 'n_unique_sites', 'entropy_mean',
                             'entropy_median', 'entropy_var', 'entropy_pct_25', 'entropy_pct_75', 'entropy_min',
                             'entropy_max',
                         ])
        self.constant_sites_pct = 0
        self.n_unique_sites = 0
        self.entropy_mean = 0
        self.entropy_median = 0
        self.entropy_var = 0
        self.entropy_pct_25 = 0
        self.entropy_pct_75 = 0
        self.entropy_min = 0
        self.entropy_max = 0

    def calc_entropy(self, aln: list[str]):  # Noa's part
        alignment_df, alignment_df_fixed, alignment_df_unique = get_alignment_df(aln)
        counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
        probabilities = [
            list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
            for col in
            list(alignment_df)]
        entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
        self.constant_sites_pct = sum([1 for et in entropy if et == 0]) / len(entropy)
        self.n_unique_sites = len(alignment_df_unique.columns)

        self.entropy_max = float(np.max(entropy))
        self.entropy_min = float(np.min(entropy))
        self.entropy_mean = float(np.mean(entropy))
        self.entropy_median = float(np.median(entropy))
        self.entropy_var = float(np.var(entropy))
        self.entropy_pct_25 = calc_percentile(entropy, 25)
        self.entropy_pct_75 = calc_percentile(entropy, 75)

def get_alignment_df(data: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_alignment_df = alignment_list_to_df(data)
    alignment_df_fixed = original_alignment_df.replace('-', np.nan)
    alignment_df_unique = original_alignment_df.T.drop_duplicates().T
    return original_alignment_df, alignment_df_fixed, alignment_df_unique

def alignment_list_to_df(alignment_data: list[str]) -> pd.DataFrame:
    alignment_list = [list(alignment_data[i]) for i in range(len(alignment_data))]
    loci_num = len(alignment_data[0])
    columns = list(range(0, loci_num))
    original_alignment_df = pd.DataFrame(alignment_list, columns=columns)
    return original_alignment_df

