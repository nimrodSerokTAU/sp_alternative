import os
from pathlib import Path

from classes.msa import MSA
from classes.sp_score import SPScore
from dpos import compute_dpos_distance


def get_file_names_ordered(file_names: list[str]) -> tuple[str, list[str]]:
    true_file_name: str
    other_file_names: list[str] = []
    for file_name in file_names:
        ext: str = file_name.split('.')[-1]
        if ext == 'fas':  # TODO: define identification
            true_file_name = file_name
        else:
            other_file_names.append(file_name)
    return true_file_name, other_file_names  # TODO: can protect


def calc_multiple_msa_sp_scores(is_naive: bool):
    project_path: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    # project_path: Path = script_path.parent.absolute()
    comparison_dir: Path = Path(os.path.join(str(project_path), 'comparison_files'))
    sp: SPScore = SPScore(-10, -0.5, 0)  # TODO: get default values
    output_file_path = Path(os.path.join(str(project_path), 'output/comparison_results.csv'))
    with (open(output_file_path, 'w') as outfile):
        print('name,sp_score,normalized_sp_score,dpos', file=outfile)
        for dir_name in os.listdir(comparison_dir):
            dir_path: Path = Path(os.path.join(str(comparison_dir), dir_name))
            true_file_name, inferred_file_names = get_file_names_ordered(os.listdir(dir_path))
            true_msa = MSA(dir_name)
            true_msa.read_me_from_fasta(Path(os.path.join(str(dir_path), true_file_name)))
            true_msa.set_my_sop_score(sp.compute_naive_sp_score(true_msa.sequences))
            for inferred_file_name in inferred_file_names:
                inferred_msa = MSA(dir_name)
                inferred_msa.read_me_from_fasta(Path(os.path.join(str(dir_path), inferred_file_name)))
                if is_naive:
                    inferred_msa.set_my_sop_score(sp.compute_naive_sp_score(inferred_msa.sequences))
                else:
                    inferred_msa.set_my_sop_score(sp.compute_naive_sp_score(inferred_msa.sequences))
                inferred_msa.order_sequences(true_msa.seq_names)
                dpos: float = compute_dpos_distance(true_msa.sequences, inferred_msa.sequences)  # TODO: handle this
                print(inferred_msa.get_my_features(dpos, true_msa.sop_score), file=outfile)

    print('done')

# TODO: use seq to order - each sequence has a different number of chars
# can use parser
# TODO: run on Ksenia's
