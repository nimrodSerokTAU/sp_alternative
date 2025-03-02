from random import shuffle
from pathlib import Path
from random import randrange

from classes.config import Configuration
from classes.global_alignment import GlobalAlign
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

    def set_sequences_to_me(self, sequences: list[str], seq_names: list[str]):
        self.sequences = sequences.copy()
        self.seq_names = seq_names.copy()

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

    def set_my_sop_score_parts(self, sp_score_subs: float, go_score: float, sp_score_gap_e: float, sp_match_count: int,
                               sp_missmatch_count: int, sp_go_count: int, sp_ge_count: int):
        self.stats.set_my_sop_score_parts(seqs_count=len(self.sequences), alignment_length=len(self.sequences[0]),
                                          sp_score_subs=sp_score_subs, go_score=go_score, sp_score_gap_e=sp_score_gap_e,
                                          sp_match_count=sp_match_count, sp_missmatch_count=sp_missmatch_count,
                                          sp_go_count=sp_go_count, sp_ge_count=sp_ge_count)

    def set_w(self, sop_w_options: list[float]):
        sop_w_options_dict: dict[str, float] = {}
        for index, weight_name in enumerate(self.weight_names):
            sop_w_options_dict[weight_name] = sop_w_options[index]
        self.stats.set_my_w_sop(sop_w_options_dict)
        print(sop_w_options_dict)

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

    def set_my_features(self, weights: set[WeightMethods], sop_w_options: list[float], true_tree: UnrootedTree,
                        dpos: float):
        self.set_my_alignment_features()
        self.build_nj_tree()
        self.calc_seq_weights(weights)
        if len(sop_w_options) > 0:
            self.set_w(sop_w_options)
        self.set_rf_from_true(true_tree)
        self.stats.set_my_dpos_dist_from_true(dpos)

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

    def create_alternative_msas_by_realign(self, config: Configuration) -> list[list[str]]:
        get_seq_inx_to_realign_a: int = randrange(len(self.seq_names) - 1)
        get_seq_inx_to_realign_b: int = randrange(get_seq_inx_to_realign_a + 1, len(self.seq_names))
        seq_a: str = self.sequences[get_seq_inx_to_realign_a]
        seq_b: str = self.sequences[get_seq_inx_to_realign_b]
        seq_a = seq_a.replace('-','')
        seq_b = seq_b.replace('-','')
        ga = GlobalAlign(seq_a, seq_b, config)
        res_seq = list(map(lambda x: {'seq_a': x.profile_a, 'seq_b': x.profile_b}, ga.aligned_sequences))
        res: list[list[str]] = []
        for option in res_seq:
            inx_on_a: int = 0
            inx_on_msa: int  = 0
            new_cols_on_msa: list[int] = []
            while inx_on_a < len(option['seq_a']):
                if option['seq_a'][inx_on_a] == self.sequences[get_seq_inx_to_realign_a][inx_on_msa]:
                    inx_on_msa += 1
                elif self.sequences[get_seq_inx_to_realign_a][inx_on_msa] == '-':
                    option['seq_a'].insert(inx_on_a, '-')
                    option['seq_b'].insert(inx_on_a, '-')
                    inx_on_msa += 1
                elif option['seq_a'][inx_on_a] == '-':
                    new_cols_on_msa.append(inx_on_msa)
                inx_on_a += 1
            seq_a = ''.join(option['seq_a'])
            seq_b = ''.join(option['seq_b'])
            new_msa_seqs: list[str] = []
            for s in self.sequences:
                seq = list(s)
                for gap_inx in new_cols_on_msa:
                    seq.insert(gap_inx, '-')
                new_msa_seqs.append(''.join(seq))
            new_msa_seqs[get_seq_inx_to_realign_a] = seq_a
            new_msa_seqs[get_seq_inx_to_realign_b] = seq_b
            res.append(new_msa_seqs)
        return res


    def create_alternative_msas_by_moving_one_part(self) -> list[list[str]]:
        res_msas: list[list[str]] = []
        partial_seq = {'start': -1, 'length': 0}
        for seq_inx_to_update in range(len(self.seq_names)):
            chars: list[dict] = []
            seq: str = self.sequences[seq_inx_to_update]
            is_gap: bool = True
            current_seq = partial_seq.copy()
            if seq[0] != '-':
                current_seq['start'] = 0
                is_gap = False
            for index, c in enumerate(seq):
                if c == '-':
                    if not is_gap:
                        current_seq['length'] = index - current_seq['start']
                        chars.append(current_seq)
                        is_gap = True
                else:
                    if is_gap:
                        current_seq = partial_seq.copy()
                        current_seq['start'] = index
                        is_gap = False
            if len(chars) > 1:
                for cp in chars:
                    new_seq_as_list: list[str] = list(seq)
                    index_to_insert: int = cp['start']
                    del new_seq_as_list[index_to_insert + cp['length']]
                    new_seq_as_list.insert(index_to_insert, '-')
                    new_seq: str = ''.join(new_seq_as_list)
                    new_msa_seqs = self.sequences.copy()
                    new_msa_seqs[seq_inx_to_update] = new_seq
                    res_msas.append(new_msa_seqs)
        return res_msas

    def print_me_to_fasta_file(self, dir_path: Path):
        output_file_path = Path(f'{str(dir_path)}/{self.dataset_name}.fas')
        with (open(output_file_path, 'w') as outfile):
            for i, seq in enumerate(self.sequences):
                print(f'>{self.seq_names[i]}', file=outfile)
                print(seq, file=outfile)







