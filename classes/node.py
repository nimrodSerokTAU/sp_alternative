from typing import Self


class Node:
    id: int
    keys: list[str]
    father: Self
    branch_length: float
    children: list[Self]

    def __init__(self, node_id: int, keys: list[str], children: list[Self] | None, branch_length: float = 0):
        self.id = node_id
        self.keys = keys
        self.father = None
        self.branch_length = branch_length
        self.children = children

    @classmethod
    def create_from_children(cls, children_list: list[Self], inx: int | None):
        keys: list[str] = []
        for child in children_list:
            keys += child.keys
        return cls(node_id=inx,
                   keys=keys,
                   children=children_list)

    def get_internal_edges_set(self) -> set[str]:
        return set(self.keys)

    def calc_rf(self, other_tree):
        return len(self.get_internal_edges_set() ^ other_tree.get_internal_edges_set())

    def add_child_to_me(self, child_node):
        self.children.append(child_node)
        child_node.set_a_father(self)

    def align_down(self):
        current_inx = 0
        for child in self.children:
            current_inx += len(child.keys)
        for child in self.children:
            if hasattr(child, 'children'):
                child.align_down()

    def set_a_father(self, other_node: 'Node'):
        self.father = other_node

    def set_branch_length(self, branch_length: float):
        self.branch_length = branch_length
