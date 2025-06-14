import numpy as np

from classes.msa_basic_stats import BasicStats
from utils import calc_percentile


class KMerStats(BasicStats):
    k_value: int
    k_mer_max: int
    k_mer_average: float
    k_mer_90_pct: int
    k_mer_95_pct: int
    kmer_sum_of_top_10: int

    def __init__(self, code: str, taxa_num: int, msa_length: int, k_value: int):
        super().__init__(code, taxa_num, msa_length,
                         [
                             'code',
            'k_mer_max', 'k_mer_average', 'k_mer_95_pct', 'k_mer_90_pct', 'kmer_sum_of_top_10',
                         ])
        self.k_value = k_value
        self.k_mer_max = 0
        self.k_mer_average = 0
        self.k_mer_95_pct = 0
        self.k_mer_90_pct = 0
        self.kmer_sum_of_top_10 = 0


    def set_k_mer_features(self, aln: list[str]):
        histo: list[int] = calc_kmer_histo(aln, min(self.k_value, len(aln[0]) - 1))
        if len(histo) > 0:
            self.k_mer_max = int(np.max(histo))
            self.k_mer_average = float(np.mean(histo))
            self.k_mer_95_pct = int(calc_percentile(histo, 95))
            self.k_mer_90_pct = int(calc_percentile(histo, 90))
            histo.sort(reverse=True)
            self.kmer_sum_of_top_10 = sum(histo[0:10])
            self.k_mer_max = histo[0]

    def get_ordered_col_names_with_k_value(self) -> list[str]:
        return [f'{col_name}_K{str(self.k_value)}' for col_name in self.ordered_col_names]


def calc_kmer_histo(aln: list[str], k: int) -> list[int]:
    histo: dict[str, int] = {}
    for seq in aln:
        for i in range(len(seq) - k + 1):
            k_mer = seq[i:i+k]
            if k_mer not in histo:
                histo[k_mer] = 0
            histo[k_mer] += 1
    return list(histo.values())

