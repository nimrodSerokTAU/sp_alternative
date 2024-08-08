import os
from pathlib import Path

from classes.sp_score import SPScore
from utils import get_msa_from_fas, get_msa_from_aln


def calc_multiple_msa_sp_scores(is_naive: bool):
    project_path: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    # project_path: Path = script_path.parent.absolute()
    comparison_dir: Path = Path(os.path.join(str(project_path), 'comparison_files'))
    sp: SPScore = SPScore(-1, -5, 0)
    output_file_path = Path(os.path.join(str(project_path), 'output/comparison_results.csv'))
    with open(output_file_path, 'w') as outfile:
        print('name,sp_score_true,sp_score_inferred,dpos', file=outfile)
        for dir_name in os.listdir(comparison_dir):
            dir_path: Path = Path(os.path.join(str(comparison_dir), dir_name))
            fas_file_path: Path | None = None
            aln_file_path: Path | None = None
            for file_name in os.listdir(dir_path):
                ext: str = file_name.split('.')[-1]
                if ext == 'fas':
                    fas_file_path: Path = Path(os.path.join(str(dir_path), file_name))
                elif ext == 'aln':
                    aln_file_path: Path = Path(os.path.join(str(dir_path), file_name))
            true_msa = get_msa_from_fas(fas_file_path)
            inferred_msa = get_msa_from_aln(aln_file_path)
            sp_score_true: int = 0
            sp_score_inferred: int = 0
            if is_naive:
                sp_score_true = sp.compute_naive_sp_score(true_msa)
                sp_score_inferred = sp.compute_naive_sp_score(inferred_msa)
            else:
                sp_score_true = sp.compute_naive_sp_score(true_msa)
                sp_score_inferred = sp.compute_naive_sp_score(inferred_msa)
            dpos: float = 1  # compute_dpos_distance(true_msa, inferred_msa)  # TODO: handle this
            print(f'{dir_name},{sp_score_true},{sp_score_inferred},{dpos}', file=outfile)
            print(str(sp_score_true))
            print(str(sp_score_inferred))

    print('done')
