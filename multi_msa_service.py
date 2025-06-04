import os
from pathlib import Path

from classes.msa import MSA
from classes.config import Configuration
from classes.sp_score import SPScore
from classes.unrooted_tree import UnrootedTree
from enums import StatsOutput


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


def multiple_msa_calc_features_and_labels(config: Configuration):
    all_msa_stats: dict[str, dict] = {}
    for stats_file_name in config.stats_output:
        all_msa_stats[stats_file_name.value] = {}
    project_path: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    # project_path: Path = script_path.parent.absolute()
    comparison_dir: Path = Path(os.path.join(str(project_path), config.input_files_dir_name))
    sp_models: list[SPScore] = [SPScore(i) for i in config.models]
    output_dir_path = Path(os.path.join(str(project_path), 'output/'))
    for dir_name in os.listdir(comparison_dir):
        dir_path: Path = Path(os.path.join(str(comparison_dir), dir_name))
        true_file_name, true_tree_file_name, inferred_file_names = get_file_names_ordered(os.listdir(dir_path))
        true_msa = MSA(dir_name)
        if true_file_name:
            true_msa.read_me_from_fasta(Path(os.path.join(str(dir_path), true_file_name)))
        if len({StatsOutput.ALL, StatsOutput.RF_LABEL}.intersection(config.stats_output)) > 0:
            if true_tree_file_name:
                true_msa.set_tree(UnrootedTree.create_from_newick_file(Path(os.path.join(str(dir_path), true_tree_file_name))))
            else:
                true_msa.build_nj_tree()

        # alternative_true: list[list[str]] = true_msa.create_alternative_msas_by_moving_smallest()
        # for i, m in enumerate(alternative_true):
        #     inf_alt_msa = MSA(f'true_alt_{i}')
        #     inf_alt_msa.set_sequences_to_me(m, true_msa.seq_names)
        #     add_msa_to_stats(all_msa_stats, true_msa, true_msa, config, sp)
        is_init_files: bool = True
        for inferred_file_name in inferred_file_names:
            msa_name = inferred_file_name # if config.is_analyze_per_dir else dir_name
            print(msa_name)
            inferred_msa = MSA(msa_name)
            inferred_msa.read_me_from_fasta(Path(os.path.join(str(dir_path), inferred_file_name)))
            inferred_msa.order_sequences(true_msa.seq_names)
            # alternative_inferred: list[list[str]] = inferred_msa.create_alternative_msas_by_moving_one_part()
            inferred_msa.calc_and_print_stats(true_msa, config, sp_models, output_dir_path, true_msa.tree, is_init_files)
            is_init_files = False
            # for i, m in enumerate(alternative_inferred):
            #     inf_alt_msa = MSA(f'{msa_name}_alt_{i}')
            #     inf_alt_msa.set_sequences_to_me(m, inferred_msa.seq_names)
            #     add_msa_to_stats(all_msa_stats, inferred_msa, true_msa, config, sp)
        if true_msa is not None:
            true_msa.calc_and_print_stats(true_msa, config, sp_models, output_dir_path, true_msa.tree, is_init_files)
    print('done')
    # TODO: handle alternative_inferred
