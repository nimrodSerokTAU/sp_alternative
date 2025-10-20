# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from classes.msa import MSA

class MsaAlignByPattern(MSA):
    pattern_msa: MSA

    def __init__(self, dataset_name: str, original_file_path: str, pattern_file_path: str, output_dir_path: str):
        super().__init__(dataset_name)
        self.pattern_msa = MSA(dataset_name)
        self.read_data(original_file_path, pattern_file_path)
        self.align_me()
        self.print_me_to_fasta_file(Path(output_dir_path))

    def read_data(self, original_file_path: str, pattern_file_path: str):
        self.read_me_from_fasta(Path(original_file_path))
        self.pattern_msa.read_me_from_fasta(Path(pattern_file_path))

    def align_me(self):
        self.pattern_msa.order_sequences(self.seq_names)
        for seq_i in range(len(self.sequences)):
            orig_seq: str = self.sequences[seq_i].replace('-', '')
            orig_seq_i: int = 0
            pattern: str = self.pattern_msa.sequences[seq_i]
            aligned_seq: list[str] = []
            for char in pattern:
                if char == '-':
                    aligned_seq.append('-')
                else:
                    aligned_seq.append(orig_seq[orig_seq_i])
                    orig_seq_i += 1
            self.sequences[seq_i] = ''.join(aligned_seq)

# if __name__ == '__main__':
#     folder = '/groups/pupko/kseniap/'
#     print("start\n")
#     if len(sys.argv) > 1:
#         code = sys.argv[1]
#         print(code)
#         original_file_path = f"{folder}/BaliBase4/BaliBase4/RV10/{code}.tfa"
#         path = f'{folder}/BaliBase4/RV10/Bali-Phy_BaliBase_MSAs_no_ancestors/{code}'
#         if not os.path.exists(original_file_path):
#             print(f"reference MSA {original_file_path} does not exist\n")
#         else:
#             for file in os.listdir(f'{path}'):
#                 try:
#                     if "true_tree.txt" in file:
#                         continue
#                     if "TRUE.fas" in file:
#                         continue
#                     if ".DS_Store" in file:
#                         continue
#                     pattern_file_path = f"{path}/{file}"
#                     print(pattern_file_path)
#                     output_dir_path = f"{folder}/BaliBase4/RV10/Bali-Phy_BaliBase_MSAs_no_ancestors_corr/{code}/"
#                     print(output_dir_path)
#                     if not os.path.exists(output_dir_path):
#                         os.makedirs(output_dir_path, exist_ok=True)
#                     if '.fas' in file:
#                         dataset_name = file.split('.fas')[0]
#                         print(dataset_name)
#                     else:
#                         dataset_name = file.split('.fasta')[0]
#                         print(dataset_name)
#                     msa_aligner = MsaAlignByPattern(dataset_name=dataset_name, original_file_path=original_file_path, pattern_file_path=pattern_file_path, output_dir_path=output_dir_path)
#                     if 'MSA.BALIPHY.aln.best' in dataset_name:
#                         shutil.move(f'{output_dir_path}/{dataset_name}.fasta', f'{output_dir_path}/{dataset_name}.fas')
#                     print("Alignment completed and saved to:", output_dir_path)
#
#                 except Exception as e:
#                     print(f"Error processing file {file} in {path}: {e}\n")
#     else:
#         print("No Code provided as parameter\n")