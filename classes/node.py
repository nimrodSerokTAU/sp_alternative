# from typing import Self
import copy


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

    # def __init__(self, node_id: int, keys: set[str], children: list[Self], branch_length: float = 0):
    def __init__(self, node_id: int, keys: set[str], children: list['Node'], branch_length: float = 0):
        self.id = node_id
        self.keys = keys
        self.father = None
        self.branch_length = branch_length
        self.children = children
        self.parsimony_set = set()

    @classmethod
    def create_from_children(cls, children_list: list['Node'], inx: int | None):
    # def create_from_children(cls, children_list: list[Self], inx: int | None):
        keys: set[str] = set()
        for child in children_list:
            keys = keys.union(child.keys)
        return cls(node_id=inx,
                   keys=keys,
                   children=children_list)

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

