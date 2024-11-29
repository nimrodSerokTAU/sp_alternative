import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats

from classes.msa import MSA, MSAStats
from classes.config import Configuration
from classes.sp_score import SPScore
from classes.unrooted_tree import UnrootedTree
from dpos import compute_dpos_distance
from enums import SopCalcTypes


def get_file_names_ordered(file_names: list[str]) -> tuple[str | None, str | None, list[str]]:
    true_file_name: str | None = None
    true_tree_file_name: str | None = None
    other_file_names: list[str] = []
    for file_name in file_names:
        ext: str = file_name.split('.')[-1]
        if ext == 'fas':  # TODO: define identification
            true_file_name = file_name
        elif ext == 'txt':  # TODO: define identification
            true_tree_file_name = file_name
        else:
            other_file_names.append(file_name)
    return true_file_name, true_tree_file_name, other_file_names  # TODO: can protect


def print_comparison_file(output_file_path: Path, all_msa_stats: list[MSAStats], pearsonr: float, spearmanr: float,
                          sop_over_count: float, dir_name: str = ''):
    if dir_name != '':
        output_file_path = Path(f'{str(output_file_path)}_{dir_name}')
    with (open(output_file_path, 'w') as outfile):
        print(f'pearson r:{pearsonr}, spearman r:{spearmanr}, sop over 1 count: {sop_over_count}', file=outfile)
        print(','.join(all_msa_stats[0].ordered_col_names), file=outfile)
        for msa_stats in all_msa_stats:
            print(msa_stats.get_my_features(), file=outfile)


def analyze_msa_stats(all_msa_stats: list[MSAStats]) -> tuple[float, float, int]:
    if len(all_msa_stats) < 2:
        return 1, 1, int(all_msa_stats[0].normalised_sop_score > 1)
    data: list[list] = []
    sop_over_count: int = 0
    for msa_stats in all_msa_stats:
        data.append(msa_stats.get_my_features_as_list())
        if msa_stats.normalised_sop_score > 1:
            sop_over_count += 1

    df: pd.DataFrame = pd.DataFrame(data, columns=all_msa_stats[0].get_ordered_col_names())
    x = df['normalised_sop_score']
    y = df['dpos_dist_from_true']
    pearsonr: float = scipy.stats.pearsonr(x, y)
    spearmanr: float = scipy.stats.spearmanr(x, y)
    return pearsonr.statistic, spearmanr.statistic, sop_over_count


def calc_multiple_msa_sp_scores(config: Configuration):
    all_msa_stats: list[MSAStats] = []
    project_path: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    # project_path: Path = script_path.parent.absolute()
    comparison_dir: Path = Path(os.path.join(str(project_path), config.input_files_dir_name))
    sp: SPScore = SPScore(config)
    output_file_path = Path(os.path.join(str(project_path), 'output/comparison_results.csv'))
    pearsonr = spearmanr = sop_over_count = 0
    for dir_name in os.listdir(comparison_dir):
        dir_path: Path = Path(os.path.join(str(comparison_dir), dir_name))
        true_file_name, true_tree_file_name, inferred_file_names = get_file_names_ordered(os.listdir(dir_path))
        true_msa = MSA(dir_name)
        if true_tree_file_name:
            true_msa.set_tree(UnrootedTree.create_from_newick_file(Path(os.path.join(str(dir_path), true_tree_file_name))))
        if true_file_name:
            true_msa.read_me_from_fasta(Path(os.path.join(str(dir_path), true_file_name)))
        true_msa.set_my_sop_score(sp.compute_efficient_sp(true_msa.sequences))
        for inferred_file_name in inferred_file_names:
            msa_name = inferred_file_name if config.is_analyze_per_dir else dir_name
            print(msa_name)
            inferred_msa = MSA(msa_name)
            inferred_msa.read_me_from_fasta(Path(os.path.join(str(dir_path), inferred_file_name)))
            if config.sop_clac_type == SopCalcTypes.NAIVE:
                inferred_msa.set_my_sop_score(sp.compute_naive_sp_score(inferred_msa.sequences))
            else:
                sp_score_subs, go_score, sp_score_gap_e, sp_match_count, sp_missmatch_count, go_count = sp.compute_efficient_sp_parts(inferred_msa.sequences)
                inferred_msa.set_my_sop_score_parts(sp_score_subs, go_score, sp_score_gap_e, sp_match_count,
                                                    sp_missmatch_count, go_count)
            inferred_msa.order_sequences(true_msa.seq_names)
            dpos: float = compute_dpos_distance(true_msa.sequences, inferred_msa.sequences)
            inferred_msa.stats.set_my_dpos_dist_from_true(dpos)
            inferred_msa.set_my_alignment_features()
            inferred_msa.build_nj_tree()
            inferred_msa.set_rf_from_true(true_msa.tree)
            all_msa_stats.append(inferred_msa.stats)
        if config.is_analyze_per_dir:
            if config.is_compute_correlation:
                pearsonr, spearmanr, sop_over_count = analyze_msa_stats(all_msa_stats)
            print_comparison_file(output_file_path, all_msa_stats, pearsonr, spearmanr, sop_over_count, dir_name)
            all_msa_stats = []
    if not config.is_analyze_per_dir:
        if config.is_compute_correlation:
            pearsonr, spearmanr, sop_over_count = analyze_msa_stats(all_msa_stats)
        print_comparison_file(output_file_path, all_msa_stats, pearsonr, spearmanr, sop_over_count)
    print('done')

