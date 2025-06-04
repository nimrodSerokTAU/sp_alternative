from pathlib import Path
from random import randrange
from typing import Self

from classes.config import Configuration
from classes.dist_labels_stats import DistanceLabelsStats
from classes.entropy_stats import EntropyStats
from classes.gaps_stats import GapStats
from classes.global_alignment import GlobalAlign
from classes.kmer_stats import KMerStats
from classes.msa_basic_stats import BasicStats
from classes.neighbor_joining import NeighborJoining
from classes.node import Node
from classes.sop_stats import SopStats
from classes.sp_score import SPScore
from classes.tree_stats import TreeStats
from classes.unrooted_tree import UnrootedTree
from classes.w_sop_stats import WSopStats
from enums import StatsOutput
from utils import calc_kimura_distance_from_other


class MSA:
    dataset_name: str
    sequences: list[str]
    seq_names: list[str]
    tree: UnrootedTree

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.sequences = []
        self.seq_names = []

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

    def set_tree(self, tree: UnrootedTree):
        self.tree = tree

    def get_msa_len(self) -> int:
        return len(self.sequences[0])

    def get_taxa_num(self) -> int:
        return len(self.sequences)

    def create_alternative_msas_by_realign(self, config: Configuration) -> list[list[str]]:
        get_seq_inx_to_realign_a: int = randrange(len(self.seq_names) - 1)
        get_seq_inx_to_realign_b: int = randrange(get_seq_inx_to_realign_a + 1, len(self.seq_names))
        seq_a: str = self.sequences[get_seq_inx_to_realign_a]
        seq_b: str = self.sequences[get_seq_inx_to_realign_b]
        seq_a = seq_a.replace('-','')
        seq_b = seq_b.replace('-','')
        ga = GlobalAlign(seq_a, seq_b, config.models[0])
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

    def calc_and_print_stats(self, true_msa: Self, config: Configuration, sp: SPScore, output_dir_path: Path,
                             true_tree: UnrootedTree, is_init_file: bool):
        basic_stats = BasicStats(self.dataset_name, self.get_taxa_num(), self.get_msa_len(),
                                 ['code', 'taxa_num', 'msa_len'])
        self.print_stats_file(basic_stats.get_my_features_as_list(), output_dir_path,'basic_stats',
                              is_init_file, basic_stats.get_ordered_col_names())
        dist_labels_stats = DistanceLabelsStats(self.dataset_name, self.get_taxa_num(), self.get_msa_len())

        if len({StatsOutput.ALL, StatsOutput.DISTANCE_LABELS }.intersection(config.stats_output)) > 0:
            dist_labels_stats.set_my_dpos_dist_from_true(true_msa.sequences, self.sequences)
            self.print_stats_file(dist_labels_stats.get_my_features_as_list(), output_dir_path, StatsOutput.DISTANCE_LABELS.value,
                                  is_init_file, dist_labels_stats.get_ordered_col_names())

        if len({StatsOutput.ALL, StatsOutput.ENTROPY}.intersection(config.stats_output)) > 0:
            entropy_stats = EntropyStats(self.dataset_name, self.get_taxa_num(), self.get_msa_len())
            entropy_stats.calc_entropy(self.sequences)
            self.print_stats_file(entropy_stats.get_my_features_as_list(), output_dir_path, StatsOutput.ENTROPY.value,
                                  is_init_file, dist_labels_stats.get_ordered_col_names())

        if len({StatsOutput.ALL, StatsOutput.GAPS}.intersection(config.stats_output)) > 0:
            gaps_stats = GapStats(self.dataset_name, self.get_taxa_num(), self.get_msa_len())
            gaps_stats.calc_gaps_values(self.sequences)
            self.print_stats_file(gaps_stats.get_my_features_as_list(), output_dir_path, StatsOutput.GAPS.value,
                                  is_init_file, dist_labels_stats.get_ordered_col_names())

        if len({StatsOutput.ALL, StatsOutput.K_MER}.intersection(config.stats_output)) > 0:
            k_mer_stats = KMerStats(self.dataset_name, self.get_taxa_num(), self.get_msa_len())
            k_mer_stats.set_k_mer_features(self.sequences)
            self.print_stats_file(k_mer_stats.get_my_features_as_list(), output_dir_path, StatsOutput.K_MER.value,
                                  is_init_file, dist_labels_stats.get_ordered_col_names())

        if len({StatsOutput.ALL, StatsOutput.TREE}.intersection(config.stats_output)) > 0:
            self.build_nj_tree()
            tree_stats = TreeStats(self.dataset_name, self.get_taxa_num(), self.get_msa_len())
            tree_stats.set_tree_stats(self.tree.get_branches_lengths_list(), self.tree, self.sequences, self.seq_names)
            self.print_stats_file(tree_stats.get_my_features_as_list(), output_dir_path, StatsOutput.TREE.value,
                                  is_init_file, dist_labels_stats.get_ordered_col_names())
            if len({StatsOutput.ALL, StatsOutput.RF_LABEL}.intersection(config.stats_output)) > 0:
                dist_labels_stats.set_rf_from_true(self.tree, true_tree)
                data_to_print, col_names = dist_labels_stats.get_print_rf()
                self.print_stats_file(data_to_print, output_dir_path, StatsOutput.RF_LABEL.value,
                                      is_init_file, col_names)

        if len({StatsOutput.ALL, StatsOutput.SP}.intersection(config.stats_output)) > 0:

            # if len(self.weight_names) > 0:
            #     sop_w_options = sp.compute_naive_sp_score(self.sequences, self.seq_weights_options)
            #     if len(sop_w_options) > 0:
            #         self.set_w(sop_w_options)

            sop_stats = SopStats(self.dataset_name, self.get_taxa_num(), self.get_msa_len())
            sop_stats.set_my_sop_score_parts(sp, self.sequences)
            self.print_stats_file(sop_stats.get_my_features_as_list(), output_dir_path, StatsOutput.SP.value,
                                  is_init_file, dist_labels_stats.get_ordered_col_names())

        # TODO: different sop configs

        if len({StatsOutput.ALL, StatsOutput.W_SP}.intersection(config.stats_output)) > 0:
            if len({StatsOutput.ALL, StatsOutput.TREE}.intersection(config.stats_output)) == 0:
                self.build_nj_tree()
            w_sop_stats = WSopStats(self.dataset_name, self.get_taxa_num(), self.get_msa_len())
            w_sop_stats.calc_seq_weights(config.additional_weights, self.sequences, self.seq_names, self.tree)
            w_sop_stats.calc_w_sp(self.sequences, sp)
            self.print_stats_file(w_sop_stats.get_my_features_as_list(), output_dir_path, StatsOutput.W_SP.value,
                                  is_init_file, dist_labels_stats.get_ordered_col_names())


    @staticmethod
    def print_stats_file(dist_labels_stats, output_dir_path, file_name: str, is_init_file: bool,
                         col_names: list[str]):
        output_file_path = Path(f'{str(output_dir_path)}/{file_name}.csv')
        if is_init_file:
            with (open(output_file_path, 'w') as outfile):
                print(','.join(col_names), file=outfile)
                print(str(dist_labels_stats)[1:-1], file=outfile)
        else:
            with (open(output_file_path, 'a') as outfile):
                print(str(dist_labels_stats)[1:-1], file=outfile)


