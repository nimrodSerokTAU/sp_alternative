import numpy as np

from classes.msa_basic_stats import BasicStats
from utils import calc_percentile


class KMerStats(BasicStats):
    k_mer_10_max: int
    k_mer_10_mean: float
    k_mer_10_var: float
    k_mer_10_pct_95: int
    k_mer_10_pct_90: int
    k_mer_10_top_10_norm: float
    k_mer_10_norm: float
    k_mer_20_max: int
    k_mer_20_mean: float
    k_mer_20_var: float
    k_mer_20_pct_95: int
    k_mer_20_pct_90: int
    k_mer_20_top_10_norm: float
    k_mer_20_norm: float


    def __init__(self, code: str, taxa_num: int, msa_len: int):
        super().__init__(code, taxa_num, msa_len,
                         [
                             'code',
            'k_mer_10_max', 'k_mer_10_mean', 'k_mer_10_var', 'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_norm', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_norm', 'k_mer_20_top_10_norm',
                         ])
        self.k_mer_10_max = 0
        self.k_mer_10_mean = 0
        self.k_mer_10_var = 0
        self.k_mer_10_pct_95 = 0
        self.k_mer_10_pct_90 = 0
        self.k_mer_10_top_10_norm = 0
        self.k_mer_10_norm = 0
        self.k_mer_20_max = 0
        self.k_mer_20_mean = 0
        self.k_mer_20_var = 0
        self.k_mer_20_pct_95 = 0
        self.k_mer_20_pct_90 = 0
        self.k_mer_20_top_10_norm = 0
        self.k_mer_20_norm = 0


    def set_k_mer_features(self, aln: list[str]):
        k: int = 10
        histo: list[int] = calc_kmer_histo(aln, min(k, len(aln[0]) - 1))
        if len(histo) > 0:
            self.k_mer_10_max = int(np.max(histo))
            self.k_mer_10_mean = float(np.mean(histo))
            self.k_mer_10_var = float(np.var(histo))
            self.k_mer_10_pct_95 = int(calc_percentile(histo, 95))
            self.k_mer_10_pct_90 = int(calc_percentile(histo, 90))
            histo.sort(reverse=True)
            self.k_mer_10_top_10_norm = sum(histo[0:10]) / self.taxa_num / self.msa_len
            self.k_mer_10_norm = sum(histo) / self.taxa_num / self.msa_len

        k = 20
        histo_b: list[int] = calc_kmer_histo(aln, min(k, len(aln[0]) - 1))
        if len(histo_b) > 0:
            self.k_mer_20_max = int(np.max(histo_b))
            self.k_mer_20_mean = float(np.mean(histo_b))
            self.k_mer_20_var = float(np.var(histo_b))
            self.k_mer_20_pct_95 = int(calc_percentile(histo_b, 95))
            self.k_mer_20_pct_90 = int(calc_percentile(histo_b, 90))
            self.k_mer_20_top_10_norm = sum(histo_b[0:10]) / self.taxa_num / self.msa_len
            self.k_mer_20_norm = sum(histo_b) / self.taxa_num / self.msa_len


def calc_kmer_histo(aln: list[str], k: int) -> list[int]:
    histo: dict[str, int] = {}
    for seq in aln:
        for i in range(len(seq) - k):
            k_mer = seq[i:i+k]
            if k_mer not in histo:
                histo[k_mer] = 0
            histo[k_mer] += 1
    return list(filter(lambda x: x > 1, histo.values()))

