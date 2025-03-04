import ast
import os
from pathlib import Path
from typing import Self

class RFNode:
    id: int
    feature_name: str
    is_smaller: bool
    value: float
    gen: int
    children: list['Self']
    father: 'Self'
    is_truncated: bool
    is_valid: bool
    raw_feature: str


    def __init__(self, _id: int, line: str):
        self.id = _id
        self.children = []
        self.father = None
        self.is_valid = False
        clean_line: str = line.replace('-','')
        line_parts = clean_line.split('|')
        self.gen = len(line_parts) - 1
        parts = line_parts[-1].split(' ')
        self.raw_feature = line_parts[-1]
        self.feature_name = parts[1]
        if self.feature_name == 'value:':
            self.value = ast.literal_eval(parts[2])[0]
            self.is_truncated = False
        elif self.feature_name == 'truncated':
            self.is_truncated = True
        else:
            self.value = float(parts[-1].split('\n')[0])
            self.is_smaller = '<' in parts[2]
            self.is_truncated = False

    def remove_child(self, child: 'Self'):
        self.children.remove(child)

    def get_key(self):
        return f'gen_{self.gen}_{self.raw_feature}'

def read_from_file(file_path: Path) -> tuple[list[RFNode], list[RFNode]]:
    all_rf_nodes: list[RFNode] = []
    last_open_per_gen: list[RFNode | None] = [None]
    leafs: list[RFNode] = []
    with open(file_path, 'r') as in_file:
        for line in in_file:
            if len(line) == 0:
                return all_rf_nodes, leafs
            else:
                # print(line)
                rf = RFNode(len(all_rf_nodes), line)
                last_open_per_gen = add_node_to_rf_nodes(all_rf_nodes, last_open_per_gen, leafs, rf)

    return all_rf_nodes, leafs


def add_node_to_rf_nodes(all_rf_nodes: list[RFNode], last_open_per_gen: list[RFNode], leafs: list[RFNode], rf: RFNode) -> list[RFNode]:
    if not rf.is_truncated:
        if not rf.feature_name == 'value:' or rf.value <= 0.05:
            last_open_per_gen = last_open_per_gen[0: rf.gen]
            if len(last_open_per_gen) > 1:
                last_open_per_gen[-1].children.append(rf)
                rf.father = last_open_per_gen[-1]
            last_open_per_gen.append(rf)
            all_rf_nodes.append(rf)
            if rf.feature_name == 'value:' and rf.value <= 0.05:
                leafs.append(rf)
    return last_open_per_gen


def clean_tree(all_rf_nodes: list[RFNode], leafs: list[RFNode]):
    root: RFNode = all_rf_nodes[0]
    for leaf in leafs:
        leaf.is_valid = True
        mark_father_as_valid(leaf)
    all_rf_nodes.reverse()
    for n in all_rf_nodes:
        if not n.is_valid:
            n.father.remove_child(n)
    return root


def mark_father_as_valid(leaf: RFNode):
    if leaf.father is not None and not leaf.father.is_valid:
        # if leaf.father.id == 2010:
        #     stop = True
        leaf.father.is_valid = True
        mark_father_as_valid(leaf.father)

def get_statistics(roots: list[RFNode]) -> dict[str, int]:
    histo: dict[str, int] = {}
    for tree in roots:
        add_to_histo(tree, histo)
    return histo

def add_to_histo(node: RFNode, histo: dict[str, int]):
    key = node.get_key()
    if key not in histo:
        histo[key] = 0
    histo[key] += 1
    if node.gen <= 2:
        for c in node.children:
            add_to_histo(c, histo)

def analyze_trees(trees_dir_path: Path):
    roots: list[RFNode] = []
    for file_name in os.listdir(trees_dir_path):
        all_rf_nodes, leafs = read_from_file(Path(os.path.join(str(trees_dir_path), file_name)))
        root: RFNode = clean_tree(all_rf_nodes, leafs)
        roots.append(root)
    histo: dict[str, int] = get_statistics(roots)
    print(histo)



analyze_trees('C:/Users/Nimrod.Serok/Nimrod/PhDB/sp_alt/Random_Forest/random_forest_trees_squared_error_True_9_features')



