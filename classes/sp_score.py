import os

from classes.evo_model import EvoModel
from classes.gap_interval import GapInterval
from utils import read_matching_matrix, translate_to_matrix_index


class SPScore:
    w_matrix: list[list[int]]
    code_to_index_dict: dict[str, int]
    go_cost: int
    ge_cost: float
    go_cost_extremities: int
    model_name: str

    def __init__(self, evo_model: EvoModel):

        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(script_path)[0]
        blosum_file_path = os.path.join(script_dir, f'../input_config_files/{evo_model.matrix_file_name}.txt')
        w_matrix, code_to_index_dict = read_matching_matrix(blosum_file_path)
        self.w_matrix = w_matrix
        self.code_to_index_dict = code_to_index_dict
        self.go_cost = evo_model.go_cost
        self.ge_cost = evo_model.ge_cost
        self.model_name = evo_model.name

    def compute_naive_sp_score(self, profile: list[str], seq_w_options: list[list[float]] = None) -> list[int]:
        if seq_w_options is None:
            seq_w_options = [[1] * len(profile)]
        weight_options_count: int = len(seq_w_options)
        if len(profile) == 0:
            return [0] * weight_options_count
        seq_len: int = len(profile[0])
        sp_score_subs: list[int] = [0] * weight_options_count
        sp_score_gaps: list[int] = [0] * weight_options_count
        for i in range(len(profile)):
            seq_i = profile[i]
            for j in range(i + 1, len(profile)):
                seq_j = profile[j]
                clean_seq_i: list[str] = []
                clean_seq_j: list[str] = []
                seq_weights_multiplication = [seq_w_options[w_option_index][i] * seq_w_options[w_option_index][j] for w_option_index in range(weight_options_count) ]
                for k in range(seq_len):
                    if not (seq_i[k] == '-' and seq_j[k] == '-'):
                        clean_seq_i.append(seq_i[k])
                        clean_seq_j.append(seq_j[k])
                    if seq_i[k] != '-' and seq_j[k] != '-':
                        for w_option_index in range(weight_options_count):
                            sp_score_subs[w_option_index] += self.subst(seq_i[k], seq_j[k]) * seq_weights_multiplication[w_option_index]
                for gap_interval in (self.compute_gap_intervals(clean_seq_i) + self.compute_gap_intervals(clean_seq_j)):
                    for w_option_index in range(weight_options_count):
                        sp_score_gaps[w_option_index] += gap_interval.g_cost(self.go_cost, self.ge_cost) * seq_weights_multiplication[w_option_index]
        return [sp_score_subs[w_op] + sp_score_gaps[w_op] for w_op in range(weight_options_count)]


    def compute_naive_sp_score_per_col(self, profile: list[str]) -> tuple[list[float], list[float], list[float]]:
        seq_len: int = len(profile[0])
        sp_score_subs: list[float] = [0] * seq_len
        sp_score_gap_o: list[float] = [0] * seq_len
        sp_score_gap_e: list[float] = [0] * seq_len
        for i in range(len(profile)):
            seq_i = profile[i]
            for j in range(i + 1, len(profile)):
                seq_j = profile[j]
                clean_seq_i: list[str] = []
                clean_seq_j: list[str] = []
                for k in range(seq_len):
                    if not (seq_i[k] == '-' and seq_j[k] == '-'):
                        clean_seq_i.append(seq_i[k])
                        clean_seq_j.append(seq_j[k])
                    if seq_i[k] != '-' and seq_j[k] != '-':
                        sp_score_subs[k] += self.subst(seq_i[k], seq_j[k])
                    elif seq_i[k] == '-' and seq_j[k] != '-':
                        sp_score_gap_e[k] += self.ge_cost
                        if k == 0 or (clean_seq_i[-2] != '-'):
                            sp_score_gap_o[k] += self.go_cost
                    elif seq_j[k] == '-' and seq_i[k] != '-':
                        sp_score_gap_e[k] += self.ge_cost
                        if k == 0 or (clean_seq_j[-2] != '-'):
                            sp_score_gap_o[k] += self.go_cost
        return sp_score_subs, sp_score_gap_o, sp_score_gap_e


    @staticmethod
    def compute_gap_intervals(seq_i: list[str]) -> list[GapInterval]:
        seq_len: int = len(seq_i)
        gap_intervals_list: list[GapInterval] = []
        gap_interval = GapInterval()
        for k in range(seq_len):
            if seq_i[k] == '-' and gap_interval.is_empty():  # start a new gap interval
                gap_interval.set_start(start=k)
            if seq_i[k] != '-' and not gap_interval.is_empty():  # the current gap interval finish at previous position
                gap_interval.end = k - 1
                gap_intervals_list.append(gap_interval.copy_me())  # append a copy of gp_interval to the list gap_intervals_list
                gap_interval = GapInterval()
        if not gap_interval.is_empty():  # handle terminal gap if any
            gap_interval.end = seq_len - 1
            gap_intervals_list.append(gap_interval.copy_me())  # append a copy of gp_interval to the list gap_intervals_list
        return gap_intervals_list

    def compute_sp_s_and_sp_ge(self, profile: list[str]) -> tuple[float, float, float, int, int, int]:
        options_count = len(self.w_matrix[0])
        seq_len: int = len(profile[0])
        sp_match_score: float = 0
        sp_mismatch_score: float = 0
        ge_count: int = 0
        sp_match_count: int = 0
        sp_mismatch_count: int = 0
        for k in range(seq_len):
            histo: list[dict] = []
            for opt in range(options_count + 1):
                histo.append({'count': 0, 'w_sum': 0, 'sq_w_sum': 0})
            for i in range(len(profile)):
                char = profile[i][k]
                if char == '-':
                    histo[-1]['count'] += 1
                else:
                    char_index = translate_to_matrix_index(char, self.code_to_index_dict)
                    histo[char_index]['count'] += 1
            for i in range(options_count):
                if histo[i] != 0:
                    sp_match_score += float(self.w_matrix[i][i] *
                                           histo[i]['count'] * (histo[i]['count'] - 1) / 2)
                    sp_match_count += histo[i]['count'] * (histo[i]['count'] - 1) / 2
                    for j in range(i + 1, options_count):
                        if histo[j] != 0:
                            sp_mismatch_score += (self.w_matrix[i][j] *
                                            histo[i]['count'] * histo[j]['count'])
                            sp_mismatch_count += histo[i]['count'] * histo[j]['count']
            if histo[-1]['count'] > 0:
                ge_count += (len(profile) - histo[-1]['count']) * histo[-1]['count']
        return sp_match_score, sp_mismatch_score, ge_count * self.ge_cost, sp_match_count, sp_mismatch_count, ge_count

    def subst(self, a: str, b: str) -> int:
        return self.w_matrix[
            translate_to_matrix_index(a, self.code_to_index_dict)][
            translate_to_matrix_index(b, self.code_to_index_dict)]

    def compute_sp_gap_open(self, profile: list[str]) -> tuple[int, int]:
        if len(profile) == 0:
            return 0, 0
        seq_len: int = len(profile[0])
        n = len(profile)
        nb_open_gap = [0] * seq_len
        gap_closing = [[] for i in range(seq_len)]
        for seq_i in profile:
            # construct gap_intervals_list in O(L) and update nb_open_gap and gap_closing arrays
            gap_intervals_list = self.compute_gap_intervals(list(seq_i))
            for gap_interval in gap_intervals_list:
                for k in range(gap_interval.start, gap_interval.end + 1):
                    nb_open_gap[k] += 1
                gap_closing[gap_interval.end].append(gap_interval)
        sp_gp_open = 0  # part of the SP score related to gap opening costs
        sp_gpo_count = 0
        for i in range(seq_len):
            for gap_interval in gap_closing[i]:
                gpo_count = n - nb_open_gap[gap_interval.start]
                sp_gp_open += gpo_count * self.go_cost
                sp_gpo_count += gpo_count
            for gap_interval in gap_closing[i]:
                for k in range(gap_interval.start, gap_interval.end + 1):
                    nb_open_gap[k] -= 1
        return sp_gp_open, sp_gpo_count

    def compute_efficient_sp(self, profile: list[str]) -> float:
        sp_match_score, sp_mismatch_score, sp_score_gap_e, a, b, ge_count = self.compute_sp_s_and_sp_ge(profile)
        go_score, sp_gpo_count = self.compute_sp_gap_open(profile)
        return sp_match_score + sp_mismatch_score + sp_score_gap_e + go_score

    def compute_efficient_sp_parts(self, profile: list[str]) -> tuple[float, float, float, float, int, int, int, int]:
        sp_match_score, sp_mismatch_score, sp_score_gap_e, sp_match_count, sp_mismatch_count, ge_count = self.compute_sp_s_and_sp_ge(profile)
        go_score, go_count = self.compute_sp_gap_open(profile)
        return sp_match_score, sp_mismatch_score, go_score, sp_score_gap_e, sp_match_count, sp_mismatch_count, go_count, ge_count
