from pathlib import Path

from classes.msa_stats import MSAStats
from classes.neighbor_joining import NeighborJoining
from classes.node import Node
from classes.sp_score import SPScore
from classes.unrooted_tree import UnrootedTree


class MSA:
    dataset_name: str
    sequences: list[str]
    seq_names: list[str]
    tree: UnrootedTree
    stats: MSAStats

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.sequences = []
        self.seq_names = []
        self.stats = MSAStats(self.dataset_name)

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
                               sp_score_gap_e: float, sp_match_count: int, sp_missmatch_count: int):
        self.stats.set_my_sop_score_parts(seqs_count=len(self.sequences), alignment_length=len(self.sequences[0]),
                                          sp_score_subs=sp_score_subs, go_score=go_score, sp_score_gap_e=sp_score_gap_e,
                                          sp_match_count=sp_match_count, sp_missmatch_count=sp_missmatch_count)

    def build_nj_tree(self, sp_score: SPScore) -> UnrootedTree:
        distance_matrix: list[list[float]] = [[0] * len(self.sequences) for i in range(len(self.sequences))]
        nodes: list[Node] = []
        for i in range(len(self.sequences)):
            node = Node(i, [self.seq_names[i]], '', None, None)
            nodes.append(node)
            for j in range(i, len(self.sequences)):
                sop = sp_score.compute_efficient_sp(profile=[self.sequences[i], self.sequences[j]])
                distance_matrix[i][j] = sop
                distance_matrix[j][i] = sop
        nj = NeighborJoining(distance_matrix, nodes)
        return nj.unrooted_nodes

    def set_true_tree(self, tree: UnrootedTree):
        self.tree = tree
