print("start1\n")
import os
import shutil
import sys
from classes.msa_align_by_pattern import MsaAlignByPattern
folder = '/groups/pupko/kseniap/'
orig_msa_folder = 'BaliBase4/BaliBase4/RV10/'
baliphy_msa_folder = 'BaliBase4/RV10/Bali-Phy_BaliBase_MSAs_no_ancestors/'
corrected_bali_msa_folder = 'BaliBase4/RV10/Bali-Phy_BaliBase_MSAs_no_ancestors_corr/'
# orig_msa_folder = 'BaliBase4/RV11-50/ALL_Unaligned_fasta_truncated/'
# baliphy_msa_folder = 'BaliBase4/RV11-50/BALIPHY_MSAs_no_ancestors/'
# corrected_bali_msa_folder = 'BaliBase4/RV11-50/BALIPHY_MSAs_no_ancestors_corr/'
# orig_msa_folder = 'DataSets/OrthoMaM_v12a_new/SIM_DISTANT/TRUE_SEQ/'
# baliphy_msa_folder = 'DataSets/OrthoMaM_v12a_new/SIM_DISTANT/BALIPHY_MSAs_no_ancestors/'
# corrected_bali_msa_folder = 'DataSets/OrthoMaM_v12a_new/SIM_DISTANT/BALIPHY_MSAs_no_ancestors_corr/'

# orig_msa_folder = 'DataSets/OrthoMaM_v12a_new/SIM_RANDOM/TRUE_SEQ/'
# baliphy_msa_folder = 'DataSets/OrthoMaM_v12a_new/SIM_RANDOM/BALIPHY_MSAs_no_ancestors/'
# corrected_bali_msa_folder = 'DataSets/OrthoMaM_v12a_new/SIM_RANDOM/BALIPHY_MSAs_no_ancestors_corr/'

# folder = '/Users/kpolonsky/Downloads/'
# orig_msa_folder = 'BaliBase4/BaliBase4/ALL_Unaligned_fasta_truncated/'
# baliphy_msa_folder = 'BaliBase4/RV11-50/BALIPHY_MSAs_no_ancestors/'
# corrected_bali_msa_folder = 'BaliBase4/RV11-50/BALIPHY_MSAs_no_ancestors_corr/'

# folder = '/Users/kpolonsky/Downloads/'
# orig_msa_folder = 'BaliBase4/BaliBase4/RV10/'
# baliphy_msa_folder = 'BaliBase4/RV10/Bali-Phy_BaliBase_MSAs_no_ancestors/'
# corrected_bali_msa_folder = 'BaliBase4/RV10/Bali-Phy_BaliBase_MSAs_no_ancestors_corr/'

if __name__ == '__main__':
    # code = sys.argv[1]
    print("start2\n")
    if len(sys.argv) > 1:
        code = sys.argv[1]
        print(code)
        original_file_path = f"{folder}/{orig_msa_folder}/{code}.tfa"
        # original_file_path = f"{folder}/{orig_msa_folder}/{code}.faa"
        path = f'{folder}/{baliphy_msa_folder}/{code}/'
        if not os.path.exists(original_file_path):
            print(f"reference MSA {original_file_path} does not exist\n")
        else:
            for file in os.listdir(f'{path}'):
                try:
                    # if "true_tree.txt" in file:
                    #     continue
                    # if "TRUE.fas" in file:
                    #     continue
                    # if ".DS_Store" in file:
                    #     continue
                    # if '.400.fasta' not in file:
                    #     continue
                    pattern_file_path = f"{path}/{file}"
                    print(pattern_file_path)
                    output_dir_path = f"{folder}/{corrected_bali_msa_folder}/{code}/"
                    print(output_dir_path)
                    if not os.path.exists(output_dir_path):
                        os.makedirs(output_dir_path, exist_ok=True)
                    if '.fasta' in file:
                        dataset_name = file.split('.fasta')[0]
                        print(dataset_name)
                    else:
                        dataset_name = file.split('.fas')[0]
                        print(dataset_name)
                    msa_aligner = MsaAlignByPattern(dataset_name=dataset_name, original_file_path=original_file_path, pattern_file_path=pattern_file_path, output_dir_path=output_dir_path)
                    # if 'MSA.BALIPHY.aln.best' in dataset_name:
                    #     shutil.move(f'{output_dir_path}/{dataset_name}.fasta', f'{output_dir_path}/{dataset_name}.fas')
                    print("Alignment completed and saved to:", output_dir_path)

                except Exception as e:
                    print(f"Error processing file {file} in {path}: {e}\n")
    else:
        print("No Code provided as parameter\n")