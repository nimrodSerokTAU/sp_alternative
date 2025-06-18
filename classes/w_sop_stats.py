from classes.msa_basic_stats import BasicStats
from classes.rooted_tree import RootedTree
from classes.sp_score import SPScore
from classes.unrooted_tree import UnrootedTree
from enums import WeightMethods, RootingMethods


class WSopStats(BasicStats):

    sp_HENIKOFF_with_gaps: float
    sp_HENIKOFF_without_gaps: float
    sp_CLUSTAL_WEIGHTS_mid_root: float
    sp_CLUSTAL_WEIGHTS_diff_sum: float

    rooted_trees: dict[str, RootedTree]
    weight_names: list[str]
    seq_weights_options: list[list[float]]

    def __init__(self, code: str, taxa_num: int, msa_length: int):
        super().__init__(code, taxa_num, msa_length,
                         [
            'code',
            'sp_HENIKOFF_with_gaps', 'sp_HENIKOFF_without_gaps', 'sp_CLUSTAL_WEIGHTS_mid_root', 'sp_CLUSTAL_WEIGHTS_diff_sum',
                         ])
        self.sp_HENIKOFF_with_gaps = -1
        self.sp_HENIKOFF_without_gaps = -1
        self.sp_CLUSTAL_WEIGHTS_mid_root = -1
        self.sp_CLUSTAL_WEIGHTS_differential_sum = -1
        self.weight_names = []
        self.seq_weights_options = []
        self.rooted_trees = {}

    def compute_seq_w_henikoff_vars(self, sequences: list[str]) -> tuple[list[float], list[float]]:
        seq_weights_with_gap: list[float] = [0] * self.taxa_num
        seq_weights_no_gap: list[float] = [0] * self.taxa_num
        for k in range(self.msa_length):
            seq_dict: dict[str, list[int]] = {}
            for i in range(self.taxa_num):
                char = sequences[i][k]
                if not char in seq_dict:
                    seq_dict[char] = []
                seq_dict[char].append(i)
            for cluster_key in seq_dict.keys():
                w: float = 1 / len(seq_dict[cluster_key])
                for seq_inx in seq_dict[cluster_key]:
                    seq_weights_with_gap[seq_inx] += w
                    if cluster_key != '-':
                        seq_weights_no_gap[seq_inx] += w
        seq_weights_with_gap_sum: float = sum(seq_weights_with_gap)
        seq_weights_no_gap_sum: float = sum(seq_weights_no_gap)
        for seq_inx in range(self.taxa_num):
            seq_weights_with_gap[seq_inx] = seq_weights_with_gap[seq_inx] / seq_weights_with_gap_sum
            seq_weights_no_gap[seq_inx] = seq_weights_no_gap[seq_inx] / seq_weights_no_gap_sum
        return seq_weights_with_gap, seq_weights_no_gap

    def get_weight_list(self, tree: UnrootedTree, rooting_method: RootingMethods, seq_names: list[str]) -> list[float]:
        self.root_tree(tree, rooting_method)
        self.rooted_trees[rooting_method.value].calc_seq_w()
        return [self.rooted_trees[rooting_method.value].seq_weight_dict[s_name] for s_name in seq_names]

    def root_tree(self, tree: UnrootedTree, rooting_method: RootingMethods):
        self.rooted_trees[rooting_method.value] = RootedTree.root_tree(tree, rooting_method)

    def calc_seq_weights(self, additional_weights: set[WeightMethods], sequences: list[str], seq_names: list[str],
                         tree: UnrootedTree):
        if len(additional_weights) == 0:
            return None
        if WeightMethods.HENIKOFF_WG in additional_weights or WeightMethods.HENIKOFF_WOG in additional_weights:
            seq_weights_with_gap, seq_weights_no_gap = self.compute_seq_w_henikoff_vars(sequences)
            if WeightMethods.HENIKOFF_WG in additional_weights:
                self.seq_weights_options.append(seq_weights_with_gap)
                self.weight_names.append(WeightMethods.HENIKOFF_WG.value)
            if WeightMethods.HENIKOFF_WOG in additional_weights:
                self.seq_weights_options.append(seq_weights_no_gap)
                self.weight_names.append(WeightMethods.HENIKOFF_WOG.value)
            if WeightMethods.CLUSTAL_MID_ROOT in additional_weights:
                self.seq_weights_options.append(self.get_weight_list(tree, RootingMethods.LONGEST_PATH_MID, seq_names))
                self.weight_names.append(WeightMethods.CLUSTAL_MID_ROOT.value)
            if WeightMethods.CLUSTAL_DIFFERENTIAL_SUM in additional_weights:
                self.seq_weights_options.append(self.get_weight_list(tree, RootingMethods.MIN_DIFFERENTIAL_SUM, seq_names))
                self.weight_names.append(WeightMethods.CLUSTAL_DIFFERENTIAL_SUM.value)

    def set_my_w_sop(self, sop_w_options_dict: dict[str, float]):
        self.sp_HENIKOFF_with_gaps = sop_w_options_dict[WeightMethods.HENIKOFF_WG.value] if WeightMethods.HENIKOFF_WG.value in sop_w_options_dict else 0
        self.sp_HENIKOFF_without_gaps = sop_w_options_dict[WeightMethods.HENIKOFF_WOG.value] if WeightMethods.HENIKOFF_WOG.value in sop_w_options_dict else 0
        self.sp_CLUSTAL_WEIGHTS_mid_root = sop_w_options_dict[
            WeightMethods.CLUSTAL_MID_ROOT.value] if WeightMethods.CLUSTAL_MID_ROOT.value in sop_w_options_dict else 0
        self.sp_CLUSTAL_WEIGHTS_diff_sum = sop_w_options_dict[
            WeightMethods.CLUSTAL_DIFFERENTIAL_SUM.value] if WeightMethods.CLUSTAL_DIFFERENTIAL_SUM.value in sop_w_options_dict else 0

    def calc_w_sp(self, sequences: list[str], sp: SPScore):
        sop_w_options: list[float] = []
        if len(self.weight_names) > 0:
            sop_w_options = sp.compute_naive_sp_score(sequences, self.seq_weights_options)
        sop_w_options_dict: dict[str, float] = {}
        for index, weight_name in enumerate(self.weight_names):
            sop_w_options_dict[weight_name] = sop_w_options[index]
        self.set_my_w_sop(sop_w_options_dict)
        # print(sop_w_options_dict)

    def get_ordered_col_names_with_model(self, model_name: str, go_val: float, ge_val: float) -> list[str]:
        return [f'{col_name}_{model_name}_GO_{go_val}_GE_{ge_val}' for col_name in self.ordered_col_names]
