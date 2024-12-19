import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
from collections import defaultdict, Counter

from classes.node import Node
from classes.unrooted_tree import UnrootedTree
from enums import WeightMethods

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
    sp_score_subs_norm: float
    sp_go_score_norm: float
    sp_score_gap_e_norm: float
    sp_match_ratio: float
    sp_missmatch_ratio: float
    single_char_count: int
    double_char_count: int
    rf_from_true: int
    median_bl: float
    bl_25_pct: float
    bl_75_pct: float
    var_bl: float
    skew_bl: float
    kurtosis_bl: float
    bl_std: float
    bl_max: float
    bl_min: float
    bl_sum: float
    nj_parsimony_score: int
    nj_parsimony_sd: int
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
    number_of_gap_segments: int
    number_of_mismatches: int
    henikoff_with_gaps: float
    henikoff_without_gaps: float
    clustal_mid_root: float
    clustal_differential_sum: float

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
        self.sp_score_subs_norm = 0
        self.sp_go_score_norm = 0
        self.sp_score_gap_e_norm = 0
        self.sp_match_ratio = 0
        self.sp_missmatch_ratio = 0
        self.single_char_count = 0
        self.double_char_count = 0
        self.rf_from_true = -1
        self.median_bl = -1
        self.bl_25_pct = -1
        self.bl_75_pct = -1
        self.var_bl = -1
        self.skew_bl = -1
        self.kurtosis_bl = -1
        self.bl_std = -1
        self.bl_max = -1
        self.bl_min = -1
        self.bl_sum = -1
        self.nj_parsimony_score = -1
        self.nj_parsimony_sd = -1
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
        self.number_of_gap_segments = 0
        self.number_of_mismatches = 0
        self.henikoff_with_gaps = 0
        self.henikoff_without_gaps = 0
        self.clustal_mid_root = 0
        self.clustal_differential_sum = 0
        self.ordered_col_names = [
            'code', 'sop_score', 'normalised_sop_score', 'rf_from_true', 'dpos_dist_from_true', 'taxa_num',
            'constant_sites_pct', 'n_unique_sites', 'pypythia_msa_difficulty', 'entropy_mean',
            'entropy_median', 'entropy_var', 'entropy_pct_25', 'entropy_pct_75', 'entropy_min', 'entropy_max',
            'av_gaps', 'msa_len', 'seq_max_len', 'seq_min_len', 'total_gaps', 'gaps_len_one', 'gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'avg_unique_gap', 'num_unique_gaps', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'num_cols_no_gaps',
            'num_cols_1_gap', 'num_cols_2_gaps', 'num_cols_all_gaps_except1',
            'sp_score_subs_norm', 'sp_score_gap_e_norm',
            'sp_match_ratio', 'sp_missmatch_ratio', 'single_char_count', 'double_char_count', 'bl_sum',
            'median_bl', 'bl_25_pct', 'bl_75_pct', 'var_bl', 'skew_bl', 'kurtosis_bl', 'bl_std', 'bl_max', 'bl_min',
            'k_mer_10_max', 'k_mer_10_mean', 'k_mer_10_var', 'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_norm', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_norm', 'k_mer_20_top_10_norm', 'number_of_gap_segments', 'number_of_mismatches', 'henikoff_with_gaps', 'henikoff_without_gaps', 'clustal_mid_root',
            'clustal_differential_sum'
        ]

    def set_my_sop_score(self, sop_score: float):
        self.sop_score = sop_score

    def set_my_sop_score_parts(self, seqs_count: int, alignment_length: int, sp_score_subs: float, go_score: float,
                               sp_score_gap_e: float, sp_match_count: int, sp_missmatch_count: int, sp_go_count: int):
        number_of_pairs = seqs_count * (seqs_count - 1) / 2 * alignment_length
        self.sop_score = sp_score_subs + go_score + sp_score_gap_e
        self.normalised_sop_score = self.sop_score / number_of_pairs
        self.sp_score_subs_norm = sp_score_subs / number_of_pairs
        self.sp_go_score_norm = go_score / number_of_pairs
        self.sp_score_gap_e_norm = sp_score_gap_e / number_of_pairs
        self.sp_match_ratio = sp_match_count / number_of_pairs
        self.sp_missmatch_ratio = sp_missmatch_count / number_of_pairs
        self.number_of_mismatches = sp_missmatch_count
        self.number_of_gap_segments = sp_go_count

    def set_my_w_sop(self, sop_w_options_dict: dict[str, float]):  # TODO: continue from here. use default
        self.henikoff_with_gaps = sop_w_options_dict[
            WeightMethods.HENIKOFF_WG.value] if WeightMethods.HENIKOFF_WG.value in sop_w_options_dict else 0
        self.henikoff_without_gaps = sop_w_options_dict[
            WeightMethods.HENIKOFF_WOG.value] if WeightMethods.HENIKOFF_WOG.value in sop_w_options_dict else 0
        self.clustal_mid_root = sop_w_options_dict[
            WeightMethods.CLUSTAL_MID_ROOT.value] if WeightMethods.CLUSTAL_MID_ROOT.value in sop_w_options_dict else 0
        self.clustal_differential_sum = sop_w_options_dict[
            WeightMethods.CLUSTAL_DIFFERENTIAL_SUM.value] if WeightMethods.CLUSTAL_DIFFERENTIAL_SUM.value in sop_w_options_dict else 0

    def set_my_dpos_dist_from_true(self, dpos: float):
        self.dpos_dist_from_true = dpos

    def set_my_alignment_features(self, aln: list[str]):
        self.set_length(aln)
        self.set_taxa_num(aln)
        self.set_values(aln)
        self.calc_entropy(aln)
        self.set_k_mer_features(aln)

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
        gap_positions = {}
        gaps_length_histogram = defaultdict(int)
        #
        for seq_index, record in enumerate(aln):
            total_gap_char += str(record).count('-')
            len_no_gaps = len(str(record).replace('-', ''))
            if len_no_gaps < min_length:
                min_length = len_no_gaps
            if len_no_gaps > max_length:
                max_length = len_no_gaps
            self.record_gap_lengths(record, seq_index, gap_positions, gaps_length_histogram)
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
        self.gaps_len_one = gaps_length_histogram[1]  # double counts the "same" gap in different sequences
        self.gaps_len_two = gaps_length_histogram[2]  # double counts the "same" gap in different sequences
        self.gaps_len_three = gaps_length_histogram[3]  # double counts the "same" gap in different sequences
        self.gaps_len_three_plus = sum(count for length, count in gaps_length_histogram.items() if
                                       length > 3)  # double counts the "same" gap in different sequences

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

    def record_gap_lengths(self, sequence: str, seq_index: int, gap_positions: dict, gaps_length_histogram) -> None:
        start_index = -1
        current_length = 0
        last_gap_index = -1
        single_char_count = 0
        double_char_count = 0
        for i, char in enumerate(sequence):
            if char == '-':
                if current_length == 0:
                    start_index = i
                    if start_index == last_gap_index + 2:
                        single_char_count += 1
                    elif start_index == last_gap_index + 3:
                        double_char_count += 1
                current_length += 1
            else:
                if current_length > 0:
                    if (current_length, start_index) in gap_positions and seq_index not in gap_positions[
                            (current_length, start_index)]:
                        gap_positions[(current_length, start_index)].append(seq_index)
                    else:
                        gap_positions[(current_length, start_index)] = [seq_index]
                    gaps_length_histogram[current_length] += 1
                    last_gap_index = max(i - 1, 0)
                current_length = 0

        # Record if the sequence ends with gaps
        if current_length > 0:
            if (current_length, start_index) in gap_positions and seq_index not in gap_positions[
                    (current_length, start_index)]:
                gap_positions[(current_length, start_index)].append(seq_index)
            else:
                gap_positions[(current_length, start_index)] = [seq_index]
            gaps_length_histogram[current_length] += 1
        else:
            current_index = len(sequence)
            if current_index == last_gap_index + 2:
                single_char_count += 1
            elif current_index == last_gap_index + 3:
                double_char_count += 1

        self.total_gaps = sum(count for length, count in gaps_length_histogram.items())
        self.single_char_count += single_char_count
        self.double_char_count += double_char_count

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
        if num_of_gaps != 0:
            self.av_gaps = total_length / num_of_gaps
        else:
            self.av_gaps = 0
        self.num_unique_gaps = unique_gaps
        if self.num_unique_gaps != 0:
            self.avg_unique_gap = unique_gaps_length / unique_gaps
        else:
            self.avg_unique_gap = 0
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

    def set_rf_from_true(self, my_tree: UnrootedTree, true_tree: UnrootedTree):
        self.rf_from_true = my_tree.calc_rf(true_tree)

    def set_tree_stats(self, bl_list: list[float], tree: UnrootedTree, aln: list[str], names: list[str]):
        self.median_bl = float(np.median(bl_list))
        self.bl_25_pct = calc_percentile(bl_list, 25)
        self.bl_75_pct = calc_percentile(bl_list, 75)
        self.var_bl = float(np.var(bl_list))
        self.skew_bl = float(skew(bl_list))
        self.kurtosis_bl = kurtosis(bl_list)
        self.bl_std = float(np.std(bl_list))
        self.bl_max = max(bl_list)
        self.bl_min = min(bl_list)
        self.bl_sum = sum(bl_list)
        parsimony_score_list: list[int] = calc_parsimony(tree, aln, names)
        self.nj_parsimony_score = sum(parsimony_score_list)
        self.nj_parsimony_sd = np.std(parsimony_score_list)
        # self.nj_parsimony_ci = np.ci(parsimony_score_list)

    def set_k_mer_features(self, aln: list[str]):
        seq_num: int = len(aln)
        seq_len: int = len(aln[0])
        k: int = 10
        histo: list[int] = calc_kmer_histo(aln, min(k, len(aln[0]) - 1))
        if len(histo) > 0:
            self.k_mer_10_max = int(np.max(histo))
            self.k_mer_10_mean = float(np.mean(histo))
            self.k_mer_10_var = float(np.var(histo))
            self.k_mer_10_pct_95 = int(calc_percentile(histo, 95))
            self.k_mer_10_pct_90 = int(calc_percentile(histo, 90))
            histo.sort(reverse=True)
            self.k_mer_10_top_10_norm = sum(histo[0:10]) / seq_num / seq_len
            self.k_mer_10_norm = sum(histo) / seq_num / seq_len

        k = 20
        histo_b: list[int] = calc_kmer_histo(aln, min(k, len(aln[0]) - 1))
        if len(histo_b) > 0:
            self.k_mer_20_max = int(np.max(histo_b))
            self.k_mer_20_mean = float(np.mean(histo_b))
            self.k_mer_20_var = float(np.var(histo_b))
            self.k_mer_20_pct_95 = int(calc_percentile(histo_b, 95))
            self.k_mer_20_pct_90 = int(calc_percentile(histo_b, 90))
            self.k_mer_20_top_10_norm = sum(histo_b[0:10]) / seq_num / seq_len
            self.k_mer_20_norm = sum(histo_b) / seq_num / seq_len


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
    values.sort()
    return float(np.percentile(values, percentile))


def calc_parsimony(unrooted_tree: UnrootedTree, aln: list[str], names: list[str]) -> list[int]:
    parsimony_per_col: list[int] = []
    new_node: Node = Node.create_from_children([unrooted_tree.anchor.children[0], unrooted_tree.anchor.children[1]], -1)
    new_root: Node = Node.create_from_children([new_node, unrooted_tree.anchor.children[2]], -2)
    nodes_order: list[Node] = unrooted_tree.all_nodes[:-1]
    nodes_order.sort(key=lambda x: x.id)
    nodes_order.append(new_node)
    nodes_order.append(new_root)

    seq_name_to_index_dict: dict[str, int] = {}
    for i, name in enumerate(names):
        seq_name_to_index_dict[name] = i

    for col_index in range(len(aln[0])):
        col_counter = 0
        for n in nodes_order:
            if len(n.children) == 0:
                seq_index = seq_name_to_index_dict[list(n.keys)[0]]
                char = aln[seq_index][col_index]
                n.set_parsimony_set({char})
            else:
                set_a = n.children[0].parsimony_set
                set_b = n.children[1].parsimony_set
                intersection_set = set_a.intersection(set_b)
                if len(intersection_set) > 0:
                    n.set_parsimony_set(intersection_set)
                else:
                    n.set_parsimony_set(set_a.union(set_b))
                    col_counter += 1
        parsimony_per_col.append(col_counter)
    return parsimony_per_col


def calc_kmer_histo(aln: list[str], k: int) -> list[int]:
    histo: dict[str, int] = {}
    for seq in aln:
        for i in range(len(seq) - k):
            k_mer = seq[i:i+k]
            if k_mer not in histo:
                histo[k_mer] = 0
            histo[k_mer] += 1
    return list(filter(lambda x: x > 1, histo.values()))
