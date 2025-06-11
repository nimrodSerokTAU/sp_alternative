import numpy as np
from classes.msa_basic_stats import BasicStats
from classes.node import Node
from classes.unrooted_tree import UnrootedTree
from utils import calc_percentile


class TreeStats(BasicStats):
    code: str

    bl_mean: float
    bl_25_pct: float
    bl_75_pct: float
    bl_max: float
    bl_min: float
    bl_sum: float

    parsimony_mean: float
    parsimony_sum: float
    parsimony_max: float
    parsimony_min: float
    parsimony_25_pct: float
    parsimony_75_pct: float

    ordered_col_names: list[str]

    def __init__(self, code: str, taxa_num: int, msa_length: int):
        super().__init__(code, taxa_num, msa_length,
                         [
            'code',
            'bl_sum', 'bl_mean', 'bl_25_pct', 'bl_75_pct', 'bl_max', 'bl_min',
            'parsimony_mean', 'parsimony_sum', 'parsimony_max', 'parsimony_min', 'parsimony_25_pct', 'parsimony_75_pct'
                         ])
        self.bl_mean = -1
        self.bl_25_pct = -1
        self.bl_75_pct = -1
        self.bl_max = -1
        self.bl_min = -1
        self.bl_sum = -1
        self.parsimony_mean = -1
        self.parsimony_sum = -1
        self.parsimony_max = -1
        self.parsimony_min = -1
        self.parsimony_25_pct = -1
        self.parsimony_75_pct = -1

    def set_tree_stats(self, bl_list: list[float], tree: UnrootedTree, aln: list[str], names: list[str]):
        self.bl_mean = float(np.mean(bl_list))
        self.bl_sum = sum(bl_list)
        self.bl_max = max(bl_list)
        self.bl_min = min(bl_list)
        self.bl_25_pct = calc_percentile(bl_list, 25)
        self.bl_75_pct = calc_percentile(bl_list, 75)

        parsimony_score_list: list[int] = calc_parsimony(tree, aln, names)
        self.parsimony_mean = float(np.mean(parsimony_score_list))
        self.parsimony_sum = float(sum(parsimony_score_list))
        self.parsimony_max = max(parsimony_score_list)
        self.parsimony_min = min(parsimony_score_list)
        self.parsimony_25_pct = calc_percentile(parsimony_score_list, 25)
        self.parsimony_75_pct = calc_percentile(parsimony_score_list, 75)


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
