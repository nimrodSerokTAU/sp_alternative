import random
from datetime import timedelta
import time
from pathlib import Path
import os


from classes.msa import MSA
from multi_msa_service import get_file_names_ordered


def remove_columns(strings: list[str], cols_set: set[int]) -> list[str]:
    return [
        ''.join(ch for i, ch in enumerate(s) if i not in cols_set)
        for s in strings
    ]


def select_strings(sequences: list[str], seq_names: list[str], selected_names: list[str]) -> list[str]:
    name_to_string = dict(zip(seq_names, sequences))
    return [name_to_string[name] for name in selected_names if name in name_to_string]


def clean_msa_from_spaces_col(sequences: list[str]):
    empty_col_indices: set[int] = set()
    for col_inx in range(len(sequences[0])):
        is_empty: bool = True
        for row in sequences:
            if row[col_inx] != '-':
                is_empty = False
                break
        if is_empty:
            empty_col_indices.add(col_inx)
    new_sequences: list[str] = remove_columns(sequences, empty_col_indices)
    return new_sequences


def sample_and_clean_msa(msa: MSA, sample_seq_names: list[str]):
    sampled_taxa: list[str] = select_strings(msa.sequences, msa.seq_names, sample_seq_names)
    cleaned_sampled_taxa: list[str] = clean_msa_from_spaces_col(sampled_taxa)
    msa.sequences = cleaned_sampled_taxa
    msa.seq_names = sample_seq_names


def sample_seqs_and_clean_msas(input_files_dir_name: str, k: int):
    project_path: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    # project_path: Path = script_path.parent.absolute()
    input_dir: Path = Path(os.path.join(str(project_path), input_files_dir_name))
    output_dir_path = Path(os.path.join(str(project_path), input_files_dir_name, f'output_{k}/'))
    true_file_name, true_tree_file_name, inferred_file_names = get_file_names_ordered(os.listdir(input_dir))
    true_msa = MSA("true_msa")
    if not true_file_name:
        return
    true_msa.read_me_from_fasta(Path(os.path.join(str(input_dir), true_file_name)))

    sample_seq_names: list[str] = random.sample(true_msa.seq_names, k)
    sample_and_clean_msa(true_msa, sample_seq_names)

    true_msa.print_me_to_fasta_file(output_dir_path)

    for inferred_file_name in inferred_file_names:
        print(inferred_file_name)
        inferred_msa = MSA(inferred_file_name)
        inferred_msa.read_me_from_fasta(Path(os.path.join(str(input_files_dir_name), inferred_file_name)))
        sample_and_clean_msa(inferred_msa, sample_seq_names)
        inferred_msa.print_me_to_fasta_file(output_dir_path)

    print(f"Done")


sample_seqs_and_clean_msas('D:/PhDB/papers/first/large_trees/1000L1_0.5', 60)
