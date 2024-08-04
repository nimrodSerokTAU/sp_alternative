import os

from classes.gap_interval import GapInterval
from utils import read_matching_matrix, translate_to_matrix_index


class SPScore:
    w_matrix: list[list[int]]
    code_to_index_dict: dict[str, int]
    gs_cost: int
    ge_cost: int
    gs_cost_extremities: int

    def __init__(self, gs_cost: int, ge_cost: int, gap_ext: int):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(script_path)[0]
        blosum_file_path = os.path.join(script_dir, '../input_files/Blosum50.txt')
        w_matrix, code_to_index_dict = read_matching_matrix(blosum_file_path)
        self.w_matrix = w_matrix
        self.code_to_index_dict = code_to_index_dict
        self.gs_cost = gs_cost
        self.ge_cost = ge_cost
        self.gs_cost_extremities = gap_ext

    def compute_naive_sp_score(self, profile: list[str]) -> int:  # Nimrod: expect no global '-'
        if len(profile) == 0:
            return 0
        seq_len: int = len(profile[0])
        sp_score_subs: int = 0
        sp_score_gaps: int = 0
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
                        sp_score_subs += self.subst(seq_i[k], seq_j[k])  # Nimrod: bug on pseudo code
                for gap_interval in (self.compute_gap_intervals(clean_seq_i) + self.compute_gap_intervals(clean_seq_j)):  # Nimrod: only uses gaps that appears in one of them (restricted)
                    sp_score_gaps += gap_interval.g_cost()
        return sp_score_subs + sp_score_gaps

    def compute_gap_intervals(self, seq_i: list[str]) -> list[GapInterval]:
        seq_len: int = len(seq_i)
        gap_intervals_list: list[GapInterval] = []
        gap_interval = GapInterval(gs_cost=self.gs_cost, ge_cost=self.ge_cost)
        for k in range(seq_len):
            if seq_i[k] == '-' and gap_interval.is_empty():  # start a new gap interval
                gap_interval.set_start(start=k)
            if seq_i[k] != '-' and not gap_interval.is_empty():  # the current gap interval finish at previous position
                gap_interval.end = k - 1
                gap_intervals_list.append(gap_interval.copy_me())  # append a copy of gp_interval to the list gap_intervals_list
                gap_interval = GapInterval(gs_cost=self.gs_cost, ge_cost=self.ge_cost)
        if not gap_interval.is_empty():  # handle terminal gap if any
            gap_interval.end = seq_len
            gap_intervals_list.append(gap_interval.copy_me())  # append a copy of gp_interval to the list gap_intervals_list
        return gap_intervals_list

    def compute_sp_s_and_sp_ge(self, profile: list[str]) -> tuple[int, int]:
        options_count = len(self.w_matrix[0])
        seq_len: int = len(profile[0])
        sp_score_subs: int = 0
        sp_score_gap_e: int = 0
        for k in range(seq_len):
            histo = [0] * options_count
            for i in range(len(profile)):
                char = profile[i][k]
                if char == '-':
                    sp_score_gap_e += 1
                else:
                    char_index = translate_to_matrix_index(char, self.code_to_index_dict)
                    histo[char_index] += 1
            for i in range(options_count):
                if histo[i] != 0:
                    sp_score_subs += int(self.w_matrix[i][i] * histo[i] * (histo[i] - 1) / 2)
                    for j in range(i + 1, options_count):
                        if histo[j] != 0:
                            sp_score_subs += self.w_matrix[i][j] * histo[i] * histo[j]
        return sp_score_subs, sp_score_gap_e * self.ge_cost

    def subst(self, a: str, b: str) -> int:
        return self.w_matrix[
            translate_to_matrix_index(a, self.code_to_index_dict)][
            translate_to_matrix_index(b, self.code_to_index_dict)]

    def compute_sp_gap_open(self, profile: list[str]) -> int:
        if len(profile) == 0:
            return 0
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
        for i in range(seq_len):
            for gap_interval in gap_closing[i]:
                if i == seq_len - 1 or gap_interval.start == 0:
                    sp_gp_open += (n - nb_open_gap[gap_interval.start]) * self.gs_cost_extremities
                else:
                    sp_gp_open += (n - nb_open_gap[gap_interval.start]) * self.gs_cost
            for gap_interval in gap_closing[i]:
                for k in range(gap_interval.start, gap_interval.end + 1):
                    nb_open_gap[k] -= 1
        return sp_gp_open

    def calc_multiple_msa_sp_scores(self, profiles_input):
        pass


def compute_pairwise_restricted_gap_intervals(LGi: iter(list[GapInterval]), LGj: iter(list[GapInterval]), gs_cost: int,
                                              ge_cost: int) -> tuple[list[GapInterval], list[GapInterval]]:
    IGi: GapInterval or None = next(LGi, None)
    IGj: GapInterval or None = next(LGj, None)
    shift: int = 0
    LG_t_i: list[GapInterval] = []
    LG_t_j: list[GapInterval] = []
    IG_t_i = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
    IG_t_j = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
    while IGi is not None and IGj is not None:
        if IGi.start == IGj.start:
            if IGi.is_equal_to(IGj):  # // both intervals disappear when A is restricted to A|{Si,Sj}
                IGi = next(LGi, None)
                IGj = next(LGj, None)
                IG_t_i = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
                IG_t_j = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
            elif IGj.is_included_in(IGi):  # IGj disappear during restriction
                IGj = next(LGj, None)
                IG_t_j = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
                if IG_t_i.is_empty():
                    IG_t_i.start = IGi.start - shift
            else:  # IGi disappear during restriction
                IGi = next(LGi, None)
                IG_t_i = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
                if IG_t_j.is_empty():
                    IG_t_j.start = IGj.start - shift
            shift += gap_interval_intersection_length(IGi, IGj)
        elif IGi.start < IGj.start:
            if IGj.is_included_in(IGi):  # IGj disappear during restriction, shift increase
                if IG_t_i.is_empty():  # set IG_t_i.start, if not already done, before increasing shift
                    IG_t_i.start = IGi.start - shift
                shift += gap_interval_intersection_length(IGi, IGj)
                IGj = next(LGj, None)
                IG_t_j = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
            else:  # IGj start after IGi and is not included in IGi
                if IG_t_i.is_empty():
                    IG_t_i.start = IGi.start - shift
                if IGi.intersetion_with(IGj) > 0:  # IG_t_j.start is now known and shift increase
                    IG_t_j.start = IGj.start - shift
                    shift += gap_interval_intersection_length(IGi, IGj)
                IG_t_i.end = IGi.end - shift
                LG_t_i.append(IG_t_i.copy_me())
                IGi = next(LGi, None)
                IG_t_i = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
        else:  # IGj.start < IGi.start
            if IGi.is_included_in(IGj):  # IGi disappear during restriction, shift increase
                if IG_t_j.is_empty():  # set IG_t_j.start, if not already done, before increasing shift
                    IG_t_j.start = IGj.start - shift
                shift += gap_interval_intersection_length(IGi, IGj)
                IGi = next(LGi, None)
                IG_t_i = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
            else:  # IGi start after IGi and is not included in IGj
                if IG_t_j.is_empty():
                    IG_t_j.start = IGj.start - shift
                if IGj.intersection_with(IGi) > 0:  # IG_t_j.start is now known and shift increase
                    IG_t_i.start = IGi.start - shift
                    shift += gap_interval_intersection_length(IGi, IGj)
                IG_t_j.end = IGj.end - shift
                LG_t_j.append(IG_t_j.copy_me())
                IGj = next(LGj, None)
                IG_t_j = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
    if IGi is not None:  # handle last gaps in LGi
        if IG_t_i.is_empty():
            IG_t_i.start = IGi.start - shift
        IG_t_i.end = IGi.end - shift
        LG_t_i.append(IG_t_i.copy_me())
        while (IGi := next(LGi, None)) is not None:
            new_interval = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
            new_interval.set_start(IGi.start - shift)
            new_interval.set_end(IGi.end - shift)
            LG_t_i.append(new_interval)
    if IGj is not None:  # handle last gaps in LGi
        if IG_t_j.is_empty():
            IG_t_j.start = IGj.start - shift
        IG_t_j.end = IGj.end - shift
        LG_t_j.append(IG_t_j.copy_me())
        while (IGj := next(LGj, None)) is not None:
            new_interval = GapInterval(gs_cost=gs_cost, ge_cost=ge_cost)
            new_interval.set_start(IGj.start - shift)
            new_interval.set_end(IGj.end - shift)
            LG_t_j.append(new_interval)
    return LG_t_i, LG_t_j


def gap_interval_intersection_length(interval_a: GapInterval or None, interval_b: GapInterval or None) -> int:
    if interval_a is not None and interval_b is not None:
        return interval_a.intersection_with(interval_b)
    elif interval_a is not None:
        return interval_a.get_len()
    elif interval_b is not None:
        return interval_b.get_len()
    else:
        return 0
