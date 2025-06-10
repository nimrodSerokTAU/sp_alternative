import numpy as np
from scipy.stats import skew, kurtosis

from classes.msa_basic_stats import BasicStats
from classes.node import Node
from classes.unrooted_tree import UnrootedTree
from utils import calc_percentile


class TreeStats(BasicStats):
    code: str
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
    nj_parsimony_sd: float

    ordered_col_names: list[str]

    def __init__(self, code: str, taxa_num: int, msa_length: int):
        super().__init__(code, taxa_num, msa_length,
                         [
            'code',
            'bl_sum', 'median_bl', 'bl_25_pct', 'bl_75_pct', 'var_bl', 'skew_bl', 'kurtosis_bl', 'bl_std',
            'bl_max', 'bl_min', 'nj_parsimony_score', 'nj_parsimony_sd'
                         ])
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

    def set_tree_stats(self, bl_list: list[float], tree: UnrootedTree, aln: list[str], names: list[str]):
        self.median_bl = float(np.median(bl_list))
        self.bl_25_pct = calc_percentile(bl_list, 25)
        self.bl_75_pct = calc_percentile(bl_list, 75)
        self.var_bl = float(np.var(bl_list))
        self.skew_bl = float(skew(bl_list))
        self.kurtosis_bl = float(kurtosis(bl_list))
        self.bl_std = float(np.std(bl_list))
        self.bl_max = max(bl_list)
        self.bl_min = min(bl_list)
        self.bl_sum = sum(bl_list)
        parsimony_score_list: list[int] = calc_parsimony(tree, aln, names)
        self.nj_parsimony_score = sum(parsimony_score_list)
        self.nj_parsimony_sd = float(np.std(parsimony_score_list))


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
