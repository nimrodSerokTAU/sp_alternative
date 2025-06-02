from classes.msa_basic_stats import BasicStats
from classes.sp_score import SPScore
from enums import WeightMethods


class SopStats(BasicStats):
    sop_score: float
    normalised_sop_score: float
    sp_score_subs_norm: float
    sp_go_score_norm: float
    sp_score_gap_e_norm: float
    sp_match_ratio: float
    sp_missmatch_ratio: float
    sp_score_subs: float
    sp_ge_count: int
    sp_go_count: int
    number_of_mismatches: int

    def __init__(self, code: str, taxa_num: int, msa_len: int):
        super().__init__(code, taxa_num, msa_len,
                         [
                             'code',
            'sop_score', 'normalised_sop_score', 'sp_score_subs_norm', 'sp_go_score_norm', 'sp_score_gap_e_norm',
            'sp_match_ratio', 'sp_missmatch_ratio', 'sp_score_subs', 'sp_ge_count', 'sp_go_count', 'number_of_mismatches',
                         ])
        self.sop_score = 0
        self.normalised_sop_score = 0
        self.sp_score_subs_norm = 0
        self.sp_go_score_norm = 0
        self.sp_score_gap_e_norm = 0
        self.sp_match_ratio = 0
        self.sp_missmatch_ratio = 0
        self.sp_score_subs = 0
        self.sp_ge_count = 0
        self.sp_go_count = 0
        self.number_of_mismatches = 0

    def set_my_sop_score_parts(self, sp:SPScore, sequences: list[str]):
        sp_score_subs, go_score, sp_score_gap_e, sp_match_count, sp_missmatch_count, sp_go_count, sp_ge_count = sp.compute_efficient_sp_parts(sequences)
        number_of_pairs = self.taxa_num * (self.taxa_num - 1) / 2 * self.msa_len
        self.sop_score = sp_score_subs + go_score + sp_score_gap_e
        self.normalised_sop_score = self.sop_score / number_of_pairs
        self.sp_score_subs_norm = sp_score_subs / number_of_pairs
        self.sp_go_score_norm = go_score / number_of_pairs
        self.sp_score_gap_e_norm = sp_score_gap_e / number_of_pairs
        self.sp_match_ratio = sp_match_count / number_of_pairs
        self.sp_missmatch_ratio = sp_missmatch_count / number_of_pairs
        self.number_of_mismatches = sp_missmatch_count
        self.sp_go_count = sp_go_count
        self.sp_score_subs = sp_score_subs
        self.sp_ge_count = sp_ge_count

    def set_my_sop_score(self, sop_score: float):
        self.sop_score = sop_score





