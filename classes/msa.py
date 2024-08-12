from pathlib import Path


class MSAStats:
    dataset_name: str
    sop_score: float
    normalised_sop_score: float
    dpos_dist_from_true: float

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def set_my_sop_score(self, sop_score: float):
        self.sop_score = sop_score

    def set_my_normalised_sop(self, true_sop: float):
        self.normalised_sop_score = self.sop_score / true_sop

    def set_my_dpos_dist_from_true(self, dpos: float):
        self.dpos_dist_from_true = dpos

    def get_my_features(self) -> str:
        return f'{self.dataset_name},{self.sop_score},{round(self.normalised_sop_score, 4)},{round(self.dpos_dist_from_true, 4)}'

    def get_my_features_as_list(self) -> list:
        return [self.dataset_name, self.sop_score, self.normalised_sop_score, self.dpos_dist_from_true]


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

