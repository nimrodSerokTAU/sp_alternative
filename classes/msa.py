from pathlib import Path

from classes.msa_stats import MSAStats
from classes.neighbor_joining import NeighborJoining
from classes.node import Node
from classes.rooted_tree import RootedTree
from classes.sp_score import SPScore
from classes.unrooted_tree import UnrootedTree
from enums import RootingMethods, WeightMethods
from utils import calc_kimura_distance_from_other


class MSA:
    dataset_name: str
    sequences: list[str]
    seq_names: list[str]
    tree: UnrootedTree
    stats: MSAStats
    rooted_trees: dict[str, RootedTree]
    weight_names: list[str]
    seq_weights_options: list[list[float]]

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.sequences = []
        self.seq_names = []
        self.stats = MSAStats(self.dataset_name)
        self.rooted_trees = {}
        self.weight_names = []
        self.seq_weights_options = []

    def add_sequence_to_me(self, sequence: str, seq_name: str):
        self.sequences.append(sequence)
        self.seq_names.append(seq_name)

    def read_me_from_fasta(self, file_path: Path):
        seq: str = ''
        seq_name: str = ''
        with open(file_path, 'r') as in_file:
            for line in in_file:
                line = line.strip()
                if len(line) == 0:
                    self.add_sequence_to_me(seq, seq_name)
                    return
                if line[0] == '>':
                    if len(seq) > 0:
                        self.add_sequence_to_me(seq, seq_name)
                        seq = ''
                    seq_name = line[1:]
                else:
                    seq += line
        if len(seq) > 0:
            self.add_sequence_to_me(seq, seq_name)

    def order_sequences(self, ordered_seq_names: list[str]):
        names_dict: dict[str, int] = {}
        for index, name in enumerate(self.seq_names):
            names_dict[name] = index
        ordered_seq: list[str] = []
        for seq_name in ordered_seq_names:
            current_index: int = names_dict[seq_name]
            ordered_seq.append(self.sequences[current_index])
        if len(ordered_seq) != len(self.sequences):
            print('we have a problem')  # TODO: throw error later
        else:
            self.sequences = ordered_seq
            self.seq_names = ordered_seq_names

    def set_my_sop_score_parts(self, sp_score_subs: float, go_score: float,
                               sp_score_gap_e: float, sp_match_count: int, sp_missmatch_count: int, sp_go_count: int):
        self.stats.set_my_sop_score_parts(seqs_count=len(self.sequences), alignment_length=len(self.sequences[0]),
                                          sp_score_subs=sp_score_subs, go_score=go_score, sp_score_gap_e=sp_score_gap_e,
                                          sp_match_count=sp_match_count, sp_missmatch_count=sp_missmatch_count,
                                          sp_go_count=sp_go_count)

    def set_w(self, sop_w_options: list[float]):
        sop_w_options_dict: dict[str, float] = {}
        for index, weight_name in enumerate(self.weight_names):
            sop_w_options_dict[weight_name] = sop_w_options[index]
        self.stats.set_my_w_sop(sop_w_options_dict)

    def set_my_sop_score(self, sop_score: float):
        self.stats.set_my_sop_score(sop_score)

    def build_nj_tree(self):
        distance_matrix: list[list[float]] = [[0] * len(self.sequences) for i in range(len(self.sequences))]
        nodes: list[Node] = []
        for i in range(len(self.sequences)):
            node = Node(i, {self.seq_names[i]}, [], 0)
            node.fill_newick()
            nodes.append(node)
            for j in range(i, len(self.sequences)):
                kimura_distance: float = calc_kimura_distance_from_other(self.sequences[i], self.sequences[j])
                distance_matrix[i][j] = kimura_distance
                distance_matrix[j][i] = kimura_distance
        nj = NeighborJoining(distance_matrix, nodes)
        self.tree = nj.tree_res
        self.stats.set_tree_stats(self.tree.get_branches_lengths_list(), self.tree, self.sequences, self.seq_names)

    def set_tree(self, tree: UnrootedTree):
        self.tree = tree

    def set_rf_from_true(self, true_tree: UnrootedTree):
        self.stats.set_rf_from_true(self.tree, true_tree)

    def set_tree_stats(self):
        self.stats.set_tree_stats(self.tree.get_branches_lengths_list())

    def set_my_alignment_features(self):
        self.stats.set_my_alignment_features(self.sequences)

    def root_tree(self, rooting_method: RootingMethods):
        self.rooted_trees[rooting_method.value] = RootedTree.root_tree(self.tree, rooting_method)

    def get_weight_list(self, rooting_method: RootingMethods) -> list[float]:
        self.root_tree(rooting_method)
        self.rooted_trees[rooting_method.value].calc_seq_w()
        return [self.rooted_trees[rooting_method.value].seq_weight_dict[s_name] for s_name in self.seq_names]

    def compute_seq_w_henikoff_vars(self) -> tuple[list[float], list[float]]:
        seq_len: int = len(self.sequences[0])
        seq_num: int = len(self.sequences)
        seq_weights_with_gap: list[float] = [0] * seq_num
        seq_weights_no_gap: list[float] = [0] * seq_num
        for k in range(seq_len):
            seq_dict: dict[str, list[int]] = {}
            for i in range(seq_num):
                char = self.sequences[i][k]
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
        for seq_inx in range(seq_num):
            seq_weights_with_gap[seq_inx] = seq_weights_with_gap[seq_inx] / seq_weights_with_gap_sum
            seq_weights_no_gap[seq_inx] = seq_weights_no_gap[seq_inx] / seq_weights_no_gap_sum
        return seq_weights_with_gap, seq_weights_no_gap

    def calc_seq_weights(self, additional_weights: set[WeightMethods]):
        if len(additional_weights) == 0:
            return None
        if WeightMethods.HENIKOFF_WG in additional_weights or WeightMethods.HENIKOFF_WOG in additional_weights:
            seq_weights_with_gap, seq_weights_no_gap = self.compute_seq_w_henikoff_vars()
            if WeightMethods.HENIKOFF_WG in additional_weights:
                self.seq_weights_options.append(seq_weights_with_gap)
                self.weight_names.append(WeightMethods.HENIKOFF_WG.value)
            if WeightMethods.HENIKOFF_WOG in additional_weights:
                self.seq_weights_options.append(seq_weights_no_gap)
                self.weight_names.append(WeightMethods.HENIKOFF_WOG.value)
            if WeightMethods.CLUSTAL_MID_ROOT in additional_weights:
                self.seq_weights_options.append(self.get_weight_list(RootingMethods.LONGEST_PATH_MID))
                self.weight_names.append(WeightMethods.CLUSTAL_MID_ROOT.value)
            if WeightMethods.CLUSTAL_DIFFERENTIAL_SUM in additional_weights:
                self.seq_weights_options.append(self.get_weight_list(RootingMethods.MIN_DIFFERENTIAL_SUM))
                self.weight_names.append(WeightMethods.CLUSTAL_DIFFERENTIAL_SUM.value)