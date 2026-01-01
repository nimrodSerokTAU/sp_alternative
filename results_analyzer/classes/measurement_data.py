from scipy.stats import pearsonr
import numpy as np


class MeasurementDataPerCode:
    measure_key: str
    code: str
    ordered_scores: list[float]
    r_value: float

    def __init__(self, measure_key: str, code: str):
        self.measure_key = measure_key
        self.code = code
        self.ordered_scores = []

    def append_single_msa_score(self, score: str):
        self.ordered_scores.append(float(score))

    def fill_correlation(self, x: np.array, continue_missing_code: bool):
        if len(x) < 2:
            if continue_missing_code:
                return
        self.r_value, p_value = pearsonr(x, np.array(self.ordered_scores))

