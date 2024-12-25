from typing import Self


class Node:
    id: int
    keys: set[str]
    # father: Self
    father: 'Node'
    branch_length: float
    # children: list[Self]
    children: list['Node']
    newick_part: str
    parsimony_set: set[str]
    children_bl_sum: float
    bl_sum_on_differentiator: float
    bl_sum_on_non_differentiator: float
    rank_from_root: int
    w_from_root: list[float]
    weight: float

    def __init__(self, node_id: int, keys: set[str], children: list['Node'], children_bl_sum: float,
                 branch_length: float = 0):
        self.id = node_id
        self.keys = keys
        self.father = None
        self.branch_length = branch_length
        self.children = children
        self.parsimony_set = set()
        self.children_bl_sum = children_bl_sum
        self.rank_from_root = -1
        self.w_from_root = []

    @classmethod
    def create_from_children(cls, children_list: list['Node'], inx: int | None):
        keys: set[str] = set()
        children_bl_sum = 0
        for child in children_list:
            keys = keys.union(child.keys)
            children_bl_sum += child.children_bl_sum + child.branch_length
        return cls(node_id=inx,
                   keys=keys,
                   children=children_list,
                   children_bl_sum=children_bl_sum)

    def add_child_to_me(self, child_node):
        self.children.append(child_node)
        child_node.set_a_father(self)

    def set_a_father(self, other_node: 'Node'):
        self.father = other_node

    def set_branch_length(self, branch_length: float):
        self.branch_length = branch_length

    def get_keys_rooted_string(self) -> str:
        sorted_keys = list(self.keys)
        sorted_keys.sort()
        return ','.join(sorted_keys)

    def get_keys_unrooted_string(self, tree_keys: set[str], differentiator_key: str) -> str | None:
        other_side = tree_keys.difference(self.keys)
        if len(self.keys) > len(other_side) or (len(self.keys) == len(other_side) and differentiator_key in other_side):
            sorted_keys = list(other_side)
        else:
            sorted_keys = list(self.keys)
        if len(sorted_keys) < 2:
            return None
        sorted_keys.sort()
        return ','.join(sorted_keys)

    def fill_newick(self):
        if len(self.children) == 0:
            self.newick_part = f'{list(self.keys)[0]}:{self.branch_length}'
        elif len(self.children) == 2:
            self.newick_part = f'({self.children[0].newick_part},{self.children[1].newick_part}):{self.branch_length}'

    def set_parsimony_set(self, new_set: set[str]):
        self.parsimony_set = new_set

    def get_adj(self) -> list[dict]:
        res: list[dict] = []
        for child in self.children:
            res.append({'node': child, 'dist': child.branch_length})
        if self.father:
            res.append({'node': self.father, 'dist': self.branch_length})
        return res

    def update_children_only(self, children_list: list[Self]):
        self.children = children_list
        for child in children_list:
            child.set_a_father(self)

    def set_rank_from_root(self, rank: int):
        self.rank_from_root = rank

    def set_w_from_root(self, w_list: list[float]):
        self.w_from_root = w_list

    def update_data_from_children(self):
        if len(self.children):
            self.keys: set[str] = set()
            self.children_bl_sum = 0
            for child in self.children:
                self.keys = self.keys.union(child.keys)
                self.children_bl_sum += child.children_bl_sum + child.branch_length

    def set_weight_from_root(self):
        w: float = 0
        rev_w = self.w_from_root.copy()
        rev_w.reverse()
        for index, bw in enumerate(rev_w):
            w += bw / (index + 1)
        self.weight = w

