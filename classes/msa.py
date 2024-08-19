from pathlib import Path

from classes.msa_stats import MSAStats


class MSA:
    dataset_name: str
    sequences: list[str]
    seq_names: list[str]
    sop_score: float
    normalised_sop_score: float
    dpos_dist_from_true: float
    stats: MSAStats

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.sequences = []
        self.seq_names = []
        self.stats = MSAStats(self.dataset_name)

    def add_sequence_to_me(self, sequence: str, seq_name: str):
        self.sequences.append(sequence)
        self.seq_names.append(seq_name)

    def read_me_from_fasta(self, file_path: Path):
        seq: str = ''
        seq_name: str = ''
        with open(file_path, 'r') as in_file:
            for line in in_file:
                line = line.strip()
                if len(line) == 0:
                    self.add_sequence_to_me(seq, seq_name)
                    return
                if line[0] == '>':
                    if len(seq) > 0:
                        self.add_sequence_to_me(seq, seq_name)
                        seq = ''
                    seq_name = line[1:]
                else:
                    seq += line
        if len(seq) > 0:
            self.add_sequence_to_me(seq, seq_name)

    def order_sequences(self, ordered_seq_names: list[str]):
        names_dict: dict[str, int] = {}
        for index, name in enumerate(self.seq_names):
            names_dict[name] = index
        ordered_seq: list[str] = []
        for seq_name in ordered_seq_names:
            current_index: int = names_dict[seq_name]
            if seq_name == 'Dipodomys':
                stop = True
            ordered_seq.append(self.sequences[current_index])
        if len(ordered_seq) != len(self.sequences):
            print('we have a problem')  # TODO: throw error later
        else:
            self.sequences = ordered_seq
            self.seq_names = ordered_seq_names

