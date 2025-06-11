from classes.msa_basic_stats import BasicStats
from classes.sp_score import SPScore


class SopStats(BasicStats):
    sp_match: int
    sp_match_norm: float
    sp_mismatch: int
    sp_mismatch_norm: float
    sp_go: int
    sp_go_norm: float
    sp_ge: int
    sp_ge_norm: float

    sp: float
    sp_norm: float


    sp_score_subs_norm: float
    sp_go_score_norm: float
    sp_ge_score_norm: float
    sp_score_subs: float


    model_agnostic_col_names: list[str]

    def __init__(self, code: str, taxa_num: int, msa_length: int):
        super().__init__(code, taxa_num, msa_length,
            [
                'code',
                'sp_match', 'sp_match_norm', 'sp_mismatch', 'sp_mismatch_norm', 'sp_go', 'sp_go_norm', 'sp_ge', 'sp_ge_norm',
                'so', 'sp_norm', 'sp_score_subs_norm', 'sp_go_score_norm', 'sp_ge_score_norm', 'sp_score_subs',
            ])
        self.sp_match = -1 # TODO: should add count
        self.sp_match_norm = -1
        self.sp_mismatch = 0
        self.sp_mismatch_norm = -1
        self.sp_go = -1
        self.sp_go_norm = -1
        self.sp_ge = -1
        self.sp_ge_norm = -1

        self.sp = 0
        self.sp_norm = 0



        self.sp_score_subs_norm = 0  # TODO: should separate?
        self.sp_go_score_norm = 0
        self.sp_ge_score_norm = 0
        self.sp_score_subs = 0


        self.model_agnostic_col_names = ['sp_match', 'sp_match_norm', 'sp_mismatch', 'sp_mismatch_norm', 'sp_go', 'sp_go_norm', 'sp_ge', 'sp_ge_norm']

    def set_my_sop_score_parts(self, sp:SPScore, sequences: list[str]):
        sp_score_subs, go_score, sp_score_gap_e, sp_match_count, sp_mismatch_count, sp_go_count, sp_ge_count = sp.compute_efficient_sp_parts(sequences)
        number_of_pairs = self.taxa_num * (self.taxa_num - 1) / 2 * self.msa_length

        self.sp_match = sp_match_count
        self.sp_match_norm = sp_match_count / number_of_pairs
        self.sp_mismatch = sp_mismatch_count
        self.sp_mismatch_norm = sp_mismatch_count / number_of_pairs
        self.sp_go = sp_go_count
        self.sp_go_norm = sp_go_count / number_of_pairs
        self.sp_ge = sp_ge_count
        self.sp_ge_norm = sp_ge_count / number_of_pairs

        self.sp = sp_score_subs + go_score + sp_score_gap_e
        self.sp_norm = self.sp / number_of_pairs


        self.sp_score_subs_norm = sp_score_subs / number_of_pairs
        self.sp_go_score_norm = go_score / number_of_pairs
        self.sp_ge_score_norm = sp_score_gap_e / number_of_pairs



        self.sp_score_subs = sp_score_subs


    def set_my_sop_score(self, sop_score: float):
        self.sp = sop_score

    def get_ordered_col_names_with_model(self, model_name: str) -> list[str]:
        col_names: list[str] = []
        for col_name in self.ordered_col_names:
            if col_name not in self.model_agnostic_col_names:
                col_names.append(f'{col_name}_{model_name}')
            else:
                col_names.append(col_name)
        return col_names
