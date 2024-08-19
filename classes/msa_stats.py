import numpy as np
import pandas as pd
from collections import defaultdict, Counter


class MSAStats:
    code: str
    sop_score: float
    normalised_sop_score: float
    dpos_dist_from_true: float
    taxa_num: int
    constant_sites_pct: float
    n_unique_sites: int
    pypythia_msa_difficulty: int
    entropy_mean: float
    entropy_median: float
    entropy_var: float
    entropy_pct_25: float
    entropy_pct_75: float
    entropy_min: float
    entropy_max: float
    av_gaps: float
    msa_len: int
    seq_max_len: int
    seq_min_len: int
    total_gaps: int
    gaps_len_one: int
    gaps_len_two: int
    gaps_len_three: int
    gaps_len_three_plus: int
    avg_unique_gap: float
    num_unique_gaps: int
    gaps_1seq_len1: int
    gaps_2seq_len1: int
    gaps_all_except_1_len1: int
    gaps_1seq_len2: int
    gaps_2seq_len2: int
    gaps_all_except_1_len2: int
    gaps_1seq_len3: int
    gaps_2seq_len3: int
    gaps_all_except_1_len3: int
    gaps_1seq_len3plus: int
    gaps_2seq_len3plus: int
    gaps_all_except_1_len3plus: int
    num_cols_no_gaps: int
    num_cols_1_gap: int
    num_cols_2_gaps: int
    num_cols_all_gaps_except1: int
    ordered_col_names: list[str]

    def __init__(self, code: str):
        self.code = code
        self.sop_score = 0
        self.normalised_sop_score = 0
        self.dpos_dist_from_true = 0
        self.taxa_num = 0
        self.constant_sites_pct = 0
        self.n_unique_sites = 0
        self.pypythia_msa_difficulty = 0
        self.entropy_mean = 0
        self.entropy_median = 0
        self.entropy_var = 0
        self.entropy_pct_25 = 0
        self.entropy_pct_75 = 0
        self.entropy_min = 0
        self.entropy_max = 0
        self.av_gaps = 0
        self.msa_len = -1
        self.seq_max_len = -1
        self.seq_min_len = -1
        self.total_gaps = 0
        self.gaps_len_one = 0
        self.gaps_len_two = 0
        self.gaps_len_three = 0
        self.gaps_len_three_plus = 0
        self.avg_unique_gap = 0
        self.num_unique_gaps = 0
        self.gaps_1seq_len1 = 0
        self.gaps_2seq_len1 = 0
        self.gaps_all_except_1_len1 = 0
        self.gaps_1seq_len2 = 0
        self.gaps_2seq_len2 = 0
        self.gaps_all_except_1_len2 = 0
        self.gaps_1seq_len3 = 0
        self.gaps_2seq_len3 = 0
        self.gaps_all_except_1_len3 = 0
        self.gaps_1seq_len3plus = 0
        self.gaps_2seq_len3plus = 0
        self.gaps_all_except_1_len3plus = 0
        self.num_cols_no_gaps = 0
        self.num_cols_1_gap = 0
        self.num_cols_2_gaps = 0
        self.num_cols_all_gaps_except1 = 0
        self.ordered_col_names = [
            'code', 'sop_score', 'normalised_sop_score', 'dpos_dist_from_true', 'taxa_num',
            'constant_sites_pct', 'n_unique_sites', 'pypythia_msa_difficulty', 'entropy_mean',
            'entropy_median', 'entropy_var', 'entropy_pct_25', 'entropy_pct_75', 'entropy_min', 'entropy_max',
            'av_gaps', 'msa_len', 'seq_max_len', 'seq_min_len', 'total_gaps', 'gaps_len_one', 'gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'avg_unique_gap', 'num_unique_gaps', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'num_cols_no_gaps',
            'num_cols_1_gap', 'num_cols_2_gaps', 'num_cols_all_gaps_except1']

    def set_my_sop_score(self, sop_score: float):
        self.sop_score = sop_score

    def set_my_normalised_sop(self, true_sop: float):
        self.normalised_sop_score = self.sop_score / true_sop

    def set_my_dpos_dist_from_true(self, dpos: float):
        self.dpos_dist_from_true = dpos

    def set_my_alignment_features(self, aln: list[str]):
        self.set_length(aln)
        self.set_taxa_num(aln)
        self.set_values(aln)

    def get_my_features(self) -> str:
        values = self.get_my_features_as_list()
        res = ''
        for v in values:
            if type(v) is float:
                v = round(v, 3)
            res += str(v) + ','
        return res

    def get_my_features_as_list(self) -> list:
        values: list = []
        attrs = vars(self)
        for col_name in self.ordered_col_names:
            values.append(attrs[col_name])
        return values

    def set_length(self, aln: list[str]):
        self.msa_len = len(aln[0])

    def set_taxa_num(self, aln: list[str]) -> None:
        self.taxa_num = len(aln)

    def set_values(self, aln: list[str]) -> None:
        min_length = 1000000000
        max_length = -1
        total_gap_char = 0
        i = 0
        gap_positions = {}
        total_gaps_count = defaultdict(int)
        #
        for seq_index, record in enumerate(aln):
            i += 1
            total_gap_char += str(record).count('-')
            len_no_gaps = len(str(record).replace('-', ''))
            if len_no_gaps < min_length:
                min_length = len_no_gaps
            if len_no_gaps > max_length:
                max_length = len_no_gaps
            self.record_gap_lengths(record, seq_index, gap_positions, total_gaps_count)
        self.calculate_counts(gap_positions)

        # per column
        for pos in range(self.msa_len):
            # Extract the column by iterating over all sequences
            column = [record[pos] for record in aln]

            num_gaps = column.count('-')
            # Check if any sequence in this column contains a gap
            if num_gaps == 0:
                self.num_cols_no_gaps += 1
            elif num_gaps == 1:
                self.num_cols_1_gap += 1
            elif num_gaps == 2:
                self.num_cols_2_gaps += 1
            elif num_gaps == (self.taxa_num - 1):
                self.num_cols_all_gaps_except1 += 1
        self.seq_min_len = min_length
        self.seq_max_len = max_length
        self.gaps_len_one = total_gaps_count[1]  # double counts the "same" gap in different sequences
        self.gaps_len_two = total_gaps_count[2]  # double counts the "same" gap in different sequences
        self.gaps_len_three = total_gaps_count[3]  # double counts the "same" gap in different sequences
        self.gaps_len_three_plus = sum(count for length, count in total_gaps_count.items() if
                                       length > 3)  # double counts the "same" gap in different sequences

        # Noa's part incorporated
        alignment_df, alignment_df_fixed, alignment_df_unique = get_alignment_df(aln)
        counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
        probabilities = [
            list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
            for col in
            list(alignment_df)]
        entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
        self.constant_sites_pct = sum([1 for et in entropy if et == 0]) / len(entropy)
        self.n_unique_sites = len(alignment_df_unique.columns)
        # try:  ### doesn't work, needs to be fixed!!
        #     self.pypythia_msa_difficulty = self.pypythia(model="GTR+G")
        # except Exception:
        #     self.pypythia_msa_difficulty = -1

        self.entropy_max = float(np.max(entropy))
        self.entropy_min = float(np.min(entropy))
        self.entropy_mean = float(np.mean(entropy))
        self.entropy_var = float(np.var(entropy))
        self.entropy_pct_25 = calc_percentile(entropy, 25)
        self.entropy_pct_75 = calc_percentile(entropy, 75)

    def get_min_len(self) -> int:
        return self.seq_min_len

    def get_max_len(self) -> int:
        return int(self.seq_max_len)

    def record_gap_lengths(self, sequence: str, seq_index: int, gap_positions: dict, total_gaps_count) -> None:
        current_length = 0
        start_index = None

        for i, char in enumerate(sequence):
            if char == '-':
                if current_length == 0:
                    start_index = i
                current_length += 1
            else:
                if current_length > 0:
                    if (current_length, start_index) in gap_positions and seq_index not in gap_positions[
                            (current_length, start_index)]:
                        gap_positions[(current_length, start_index)].append(seq_index)
                    else:
                        gap_positions[(current_length, start_index)] = [seq_index]
                    total_gaps_count[current_length] += 1
                current_length = 0

        # Record if the sequence ends with gaps
        if current_length > 0:
            if (current_length, start_index) in gap_positions and seq_index not in gap_positions[
                    (current_length, start_index)]:
                gap_positions[(current_length, start_index)].append(seq_index)
            else:
                gap_positions[(current_length, start_index)] = [seq_index]
            total_gaps_count[current_length] += 1

        self.total_gaps = sum(count for length, count in total_gaps_count.items())

    def calculate_counts(self, gap_positions: dict) -> None:
        length_count = {1: Counter(), 2: Counter(), 3: Counter()}
        length_plus_count = Counter()
        total_length = 0
        num_of_gaps = 0
        unique_gaps = 0
        unique_gaps_length = 0

        for (length, start_index), seq_set in gap_positions.items():
            if length == 1:
                length_count[length][len(seq_set)] += 1
            if length == 2:
                length_count[length][len(seq_set)] += 1
            if length == 3:
                length_count[length][len(seq_set)] += 1
            if length > 3:
                length_plus_count[len(seq_set)] += 1
            if len(seq_set) == 1:
                unique_gaps += 1
                unique_gaps_length += length
            num_of_gaps += len(seq_set)
            total_length += len(seq_set) * length

        self.av_gaps = total_length / num_of_gaps
        self.num_unique_gaps = unique_gaps
        self.avg_unique_gap = unique_gaps_length / unique_gaps
        self.gaps_1seq_len1 = length_count[1][1]
        self.gaps_2seq_len1 = length_count[1][2]
        self.gaps_all_except_1_len1 = length_count[1][self.taxa_num - 1]
        self.gaps_1seq_len2 = length_count[2][1]
        self.gaps_2seq_len2 = length_count[2][2]
        self.gaps_all_except_1_len2 = length_count[2][self.taxa_num - 1]
        self.gaps_1seq_len3 = length_count[3][1]
        self.gaps_2seq_len3 = length_count[3][2]
        self.gaps_all_except_1_len3 = length_count[3][self.taxa_num - 1]
        self.gaps_1seq_len3plus = length_plus_count[1]
        self.gaps_2seq_len3plus = length_plus_count[2]
        self.gaps_all_except_1_len3plus = length_plus_count[self.taxa_num - 1]

    def get_ordered_col_names(self) -> list[str]:
        return self.ordered_col_names


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


def calc_percentile(values, percentile: int) -> float:
    return float(np.percentile(values, percentile))
