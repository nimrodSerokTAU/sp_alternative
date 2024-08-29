from pathlib import Path
from classes.node import Node


class UnrootedTree:
    top_nodes: [Node, Node, Node]
    newick: str

    def __init__(self, unrooted_nodes: [Node, Node, Node]):
        self.top_nodes = unrooted_nodes
        # self.newick = newick

    @classmethod
    def create_from_newick_file(cls, path: Path):
        newick_str: str = read_newick_from_file(path)
        roots: list[Node] = create_a_tree_from_newick(newick_str)
        if len(roots) == 3:
            return cls(unrooted_nodes=roots)
        if len(roots) == 2:
            res: list[Node] = []
            roots.sort(key=lambda x: x.branch_length)
            roots.sort(key=lambda x: len(x.children), reverse=True)
            if len(roots[0].children) == 2:
                res += roots[0].children
                res.append(roots[1])
                return cls(unrooted_nodes=res)


def read_newick_from_file(input_file_path: Path) -> str:
    with open(input_file_path, 'r') as in_file:
        for line in in_file:
            return line.strip()


def create_a_tree_from_newick(newick: str) -> list[Node]:
    all_nodes: list[Node] = []
    open_nodes_per_level: dict[int, list[Node]] = {}
    level = 0
    current_key = ''
    branch_length = ''
    is_reading_br_len = False
    for c in newick:
        if c == '(':
            current_key = ''
            level += 1
            if level not in open_nodes_per_level:
                open_nodes_per_level[level] = []
        elif c == ':':
            is_reading_br_len = True
        elif c == ')' or c == ',':
            if is_reading_br_len:
                if len(current_key) > 0:
                    current_node = Node(node_id=len(all_nodes), keys=[current_key], children=[],
                                        branch_length=float(branch_length))
                else:
                    node_keys = []
                    for child in open_nodes_per_level[level + 1]:
                        node_keys += child.keys
                    current_node = Node(node_id=len(all_nodes), keys=node_keys, children=open_nodes_per_level[level + 1].copy(),
                                        branch_length=float(branch_length))
                    for child in open_nodes_per_level[level + 1]:
                        child.set_a_father(current_node)
                    open_nodes_per_level[level + 1] = []
                open_nodes_per_level[level].append(current_node)
                all_nodes.append(current_node)
                is_reading_br_len = False
                current_key = ''
                branch_length = ''
            if c == ')':
                level -= 1
        elif c == ';':
            return open_nodes_per_level[1]
        else:
            if is_reading_br_len:
                branch_length += c
                continue
            current_key += c
