import os
import random
import statistics
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats

from classes.msa import MSA
from distance_calc import compute_distance
from classes.config import Configuration
from classes.sp_score import SPScore
from classes.unrooted_tree import UnrootedTree
from enums import SopCalcTypes, RootingMethods, DistanceType

folder = '/groups/pupko/kseniap/'
# output_dir = f'{folder}/OrthoMaM/dpos_res_v2/'
# output_dir = f"{folder}/DataSets/OrthoMaM_v12a_new/Guidance_Filtering_Paper/pairwise_dpos_res/"
output_dir = f"{folder}/DataSets/OrthoMaM_v12a_new/SIM_DISTANT/pairwise_dpos_res/"

if __name__ == '__main__':
    code = sys.argv[1]
    dir_path = f'{folder}/DataSets/OrthoMaM_v12a_new/SIM_DISTANT/ALL_MSAs/{code}/{code}/'
    # dir_path = '/Users/kpolonsky/Downloads/TEST_541_OrthoMaM/MSAs_ALL_500K/ACVR1/'
    dpos_dict = {}
    dpos_array = []
    muscle_files, baliphy_files, prank_files, mafft_files = [], [], [], []

    files = os.listdir(dir_path)
    for filename in files:
        if 'true_tree' in filename or '.DS_Store' in filename or 'TRUE.fas' in filename or 'orig' in filename:
            continue

        filename_lower = filename.lower()

        if "muscle" in filename_lower:
            muscle_files.append(filename)
        elif "baliphy" in filename_lower or 'bali_phy' in filename_lower:
            baliphy_files.append(filename)
        elif "prank" in filename_lower:
            prank_files.append(filename)
        else:
            mafft_files.append(filename)

    muscle_files = random.sample(muscle_files, min(100, len(muscle_files)))
    baliphy_files = random.sample(baliphy_files, min(100, len(baliphy_files)))
    prank_files = random.sample(prank_files, min(100, len(prank_files)))
    mafft_files = random.sample(mafft_files, min(100, len(mafft_files)))

    four_sets = [muscle_files, baliphy_files, prank_files, mafft_files]
    for i in range(len(four_sets)):
        for inferred_file_name1 in four_sets[i]:
            if ".DS_Store" in inferred_file_name1 or "true_tree.txt" in inferred_file_name1:
                continue
            for inferred_file_name2 in four_sets[i]:
                    if inferred_file_name1 == inferred_file_name2:
                        continue
                    if ".DS_Store" in inferred_file_name2 or "true_tree.txt" in inferred_file_name2:
                        continue
                    try:
                        print(inferred_file_name1 + " and " + inferred_file_name2)

                        inferred_msa1 = MSA(inferred_file_name1)
                        inferred_msa1.read_me_from_fasta(Path(os.path.join(str(dir_path), inferred_file_name1)))

                        inferred_msa2 = MSA(inferred_file_name2)
                        inferred_msa2.read_me_from_fasta(Path(os.path.join(str(dir_path), inferred_file_name2)))

                        inferred_msa2.order_sequences(inferred_msa1.seq_names)
                        # dpos: float = compute_dpos_distance(inferred_msa1.sequences, inferred_msa2.sequences)
                        dpos_from_true: float = compute_distance(inferred_msa1.sequences, inferred_msa2.sequences, DistanceType.D_POS)
                        dseq_from_true: float = compute_distance(inferred_msa1.sequences, inferred_msa2.sequences, DistanceType.D_SEQ)
                        ssp_from_true: float = compute_distance(inferred_msa1.sequences, inferred_msa2.sequences, DistanceType.D_SSP)

                        if isinstance(dpos_from_true, float):
                            # dpos_dict[(inferred_file_name1, inferred_file_name2)] = dpos
                            # dpos_array.append(float(dpos))
                            dpos_dict[(inferred_file_name1, inferred_file_name2)] = dpos_from_true
                            dpos_array.append(float(dpos_from_true))
                        else:
                            print(f"dpos for {inferred_file_name1}, {inferred_file_name2} is not float\n")

                    except Exception as e:
                        print(f"Exception {e} for {inferred_file_name1}, {inferred_file_name2}\n")
                    # break

        print(statistics.mean(dpos_array))
    # print(dpos_dict)
    df = pd.DataFrame(list(dpos_dict.items()), columns=['MSAs', 'dpos'])
    df.to_csv(f'{output_dir}/dpos_{code}.csv', index=False)
    # df.to_csv(f'/Users/kpolonsky/Downloads/test/dpos_ACMSD.csv', index=False)

