import os
import random
from pathlib import Path

from ete3 import Tree, TreeNode


def clean_nodes(newick_str: str, leaf_names_to_keep: list[str]) -> tuple[str, list[str]]:
    t = Tree(newick_str)
    t.prune(leaf_names_to_keep, preserve_branch_length=True)
    return t.write(), t.get_leaf_names()


def get_length_from_root(node: TreeNode) -> float:
    return sum([n._dist for n in node.get_ancestors()])

def get_length_from_root_from_newick_and_name(newick: str, name: str) -> float:
    t = Tree(newick)
    node: TreeNode = t.search_nodes(name=name)[0]
    return get_length_from_root(node)



def read_file_and_get_names(fila_name) -> tuple[str, list[str]]:
    project_path: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    file_path: str = f'{project_path}/trees/{fila_name}'
    with open(file_path) as infile:
        for line in infile:
            t = Tree(line)
            break
    leaf_names: list[str] = t.get_leaf_names()
    return t.write(), leaf_names


def delete_rand_items(items, n) -> list:
    to_delete = set(random.sample(range(len(items)),n))
    return [x for i,x in enumerate(items) if not i in to_delete]


def remove_some_leaves(fila_name: str, wanted_leaf_number: int) -> tuple[str, list[str]]:
    newick_str, leaf_names = read_file_and_get_names(fila_name)
    leaf_names_to_keep: list[str] = delete_rand_items(leaf_names, len(leaf_names) - wanted_leaf_number)
    res_tree, leaf_names = clean_nodes(newick_str, leaf_names_to_keep)
    return res_tree, leaf_names
