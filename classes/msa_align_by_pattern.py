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
