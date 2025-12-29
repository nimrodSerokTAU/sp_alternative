import random
import sys
import os
import random
from pathlib import Path
from ete3 import Tree, TreeNode
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment

# from clean_trees import clean_nodes, read_file_and_get_names, remove_some_leaves, \
#     get_length_from_root_from_newick_and_name

def read_file_and_get_names(fila_name) -> tuple[str, list[str]]:
    project_path: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    file_path: str = f'{fila_name}'
    with open(file_path) as infile:
        for line in infile:
            t = Tree(line)
            break
    leaf_names: list[str] = t.get_leaf_names()
    return t.write(), leaf_names

def clean_nodes(newick_str: str, leaf_names_to_keep: list[str]) -> tuple[str, list[str]]:
    t = Tree(newick_str)
    t.prune(leaf_names_to_keep, preserve_branch_length=True)

    for node in t.traverse():
        if not node.is_leaf():
            node.support = 0.0

    return t.write(format=5), t.get_leaf_names()

def save_tree(newick_str: str, output_tree_path: str) -> None:
    with open(output_tree_path, "w") as f:
        f.write(newick_str)

# def msa_with_chosen_species(final_leaves, input_msa, output_msa):
#     marker = 0
#     with open(input_msa, 'r') as input, open(output_msa, 'w') as output:
#         lines = input.readlines()
#         for line in lines:
#             if marker == 1 and not line.startswith('>'):
#                 output.write(line)
#             if line.startswith('>'):
#                 marker = 0
#                 species = line.split('>')[1].strip()
#                 if species in final_leaves:
#                     output.write(line)
#                     marker = 1

def filter_msa_by_species_and_remove_gaps(final_leaves: list[str], input_msa: str, output_msa: str) -> None:
    alignment = AlignIO.read(input_msa, "fasta")

    filtered = MultipleSeqAlignment([
        record for record in alignment
        if record.id in final_leaves
    ])

    AlignIO.write(filtered, output_msa, "fasta")

    if not filtered:
        raise ValueError("No matching sequences found.")

    no_gap_columns = []
    for i in range(filtered.get_alignment_length()):
        column = filtered[:, i]
        if not all(res == '-' for res in column):
            no_gap_columns.append(i)

    cleaned = MultipleSeqAlignment([])
    for record in filtered:
        cleaned_seq = ''.join(record.seq[i] for i in no_gap_columns)
        record.seq = record.seq.__class__(cleaned_seq)
        cleaned.append(record)

    AlignIO.write(cleaned, output_msa, "fasta")

def remove_gap_columns(msa_path):
    alignment = AlignIO.read(msa_path, "fasta")


if __name__ == '__main__':
    code = sys.argv[1]
    # code = '19'
    # folder = '/Users/kpolonsky/Downloads/OrthoMaM.v12'
    folder = '/groups/pupko/kseniap/DataSets/OrthoMaM_v12a_new'
    input_tree = f'{folder}/trees_AA_BL_no_supports/{code}_no_supports.tree'
    output_tree = f'{folder}/TRUE_TREES/{code}_true.tree'
    input_msa = f'{folder}/OrthoMaM_reference_MSAs/{code}_cleaned.fasta'
    output_msa = f'{folder}/REFERENCE_MSAs/{code}_ref.fasta'

    newick_str, leaf_names = read_file_and_get_names(input_tree)
    wanted_num_leaves = random.choice([20, 30, 40, 50, 60])
    print(f"{code}: the remaining number of leaves is {wanted_num_leaves}\n")
    selected_leaves = random.sample(leaf_names, wanted_num_leaves)
    res_tree, final_leaves = clean_nodes(newick_str, selected_leaves)
    assert set(selected_leaves) == set(final_leaves), "Leaf sets do not match"
    save_tree(res_tree, output_tree)
    # msa_with_chosen_species(final_leaves, input_msa, output_msa)
    # remove_gap_columns(output_msa)
    filter_msa_by_species_and_remove_gaps(final_leaves, input_msa, output_msa)
