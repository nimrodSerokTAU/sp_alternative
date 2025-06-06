from collections import defaultdict, Counter

from classes.msa_basic_stats import BasicStats


class GapStats(BasicStats):
    code: str
    msa_len: int
    taxa_num: int

    num_gap_segments: int
    gaps_len_one: int
    gaps_len_two: int
    gaps_len_three: int
    gaps_len_three_plus: int
    avg_unique_gap: float
    num_unique_gaps: int
    gaps_1seq_len1: int
    gaps_2seq_len1: int
    gaps_all_except_1_len1: int
    gaps_1seq_len2: int
    gaps_2seq_len2: int
    gaps_all_except_1_len2: int
    gaps_1seq_len3: int
    gaps_2seq_len3: int
    gaps_all_except_1_len3: int
    gaps_1seq_len3plus: int
    gaps_2seq_len3plus: int
    gaps_all_except_1_len3plus: int
    num_cols_no_gaps: int
    num_cols_1_gap: int
    num_cols_2_gaps: int
    num_cols_all_gaps_except1: int
    av_gaps: float
    single_char_count: int
    double_char_count: int
    seq_max_len: int  # TODO: rename
    seq_min_len: int  # TODO: rename

    ordered_col_names: list[str]

    def __init__(self, code: str, taxa_num: int, msa_len: int):
        super().__init__(code, taxa_num, msa_len,
                         [
                              'code',
            'av_gaps', 'num_gap_segments', 'gaps_len_one', 'gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'avg_unique_gap', 'num_unique_gaps', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'num_cols_no_gaps',
            'num_cols_1_gap', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'single_char_count', 'double_char_count',
                             'seq_max_len', 'seq_min_len'
                         ])
        self.av_gaps = 0
        self.num_gap_segments = 0
        self.gaps_len_one = 0
        self.gaps_len_two = 0
        self.gaps_len_three = 0
        self.gaps_len_three_plus = 0
        self.avg_unique_gap = 0
        self.num_unique_gaps = 0
        self.gaps_1seq_len1 = 0
        self.gaps_2seq_len1 = 0
        self.gaps_all_except_1_len1 = 0
        self.gaps_1seq_len2 = 0
        self.gaps_2seq_len2 = 0
        self.gaps_all_except_1_len2 = 0
        self.gaps_1seq_len3 = 0
        self.gaps_2seq_len3 = 0
        self.gaps_all_except_1_len3 = 0
        self.gaps_1seq_len3plus = 0
        self.gaps_2seq_len3plus = 0
        self.gaps_all_except_1_len3plus = 0
        self.num_cols_no_gaps = 0
        self.num_cols_1_gap = 0
        self.num_cols_2_gaps = 0
        self.num_cols_all_gaps_except1 = 0
        self.single_char_count = 0
        self.double_char_count = 0
        self.seq_max_len = -1
        self.seq_min_len = -1


    def calc_gaps_values(self, aln: list[str]) -> None:
        min_length = 1000000000
        max_length = -1
        total_gap_char = 0
        gap_positions = {}
        gaps_length_histogram = defaultdict(int)
        #
        for seq_index, record in enumerate(aln):
            total_gap_char += str(record).count('-')
            len_no_gaps = len(str(record).replace('-', ''))
            if len_no_gaps < min_length:
                min_length = len_no_gaps
            if len_no_gaps > max_length:
                max_length = len_no_gaps
            self.record_gap_lengths(record, seq_index, gap_positions, gaps_length_histogram)
        self.calculate_counts(gap_positions)

        # per column
        for pos in range(self.msa_len):
            # Extract the column by iterating over all sequences
            column = [record[pos] for record in aln]

            num_gaps = column.count('-')
            # Check if any sequence in this column contains a gap
            if num_gaps == 0:
                self.num_cols_no_gaps += 1
            elif num_gaps == 1:
                self.num_cols_1_gap += 1
            elif num_gaps == 2:
                self.num_cols_2_gaps += 1
            elif num_gaps == (self.taxa_num - 1):
                self.num_cols_all_gaps_except1 += 1
        self.seq_min_len = min_length
        self.seq_max_len = max_length
        self.gaps_len_one = gaps_length_histogram[1]  # double counts the "same" gap in different sequences
        self.gaps_len_two = gaps_length_histogram[2]  # double counts the "same" gap in different sequences
        self.gaps_len_three = gaps_length_histogram[3]  # double counts the "same" gap in different sequences
        self.gaps_len_three_plus = sum(count for length, count in gaps_length_histogram.items() if
                                       length > 3)  # double counts the "same" gap in different sequences


    def record_gap_lengths(self, sequence: str, seq_index: int, gap_positions: dict, gaps_length_histogram) -> None:
        start_index = -1
        current_length = 0
        last_gap_index = -1
        single_char_count = 0
        double_char_count = 0
        for i, char in enumerate(sequence):
            if char == '-':
                if current_length == 0:
                    start_index = i
                    if start_index == last_gap_index + 2:
                        single_char_count += 1
                    elif start_index == last_gap_index + 3:
                        double_char_count += 1
                current_length += 1
            else:
                if current_length > 0:
                    if (current_length, start_index) in gap_positions and seq_index not in gap_positions[
                            (current_length, start_index)]:
                        gap_positions[(current_length, start_index)].append(seq_index)
                    else:
                        gap_positions[(current_length, start_index)] = [seq_index]
                    gaps_length_histogram[current_length] += 1
                    last_gap_index = max(i - 1, 0)
                current_length = 0

        # Record if the sequence ends with gaps
        if current_length > 0:
            if (current_length, start_index) in gap_positions and seq_index not in gap_positions[
                    (current_length, start_index)]:
                gap_positions[(current_length, start_index)].append(seq_index)
            else:
                gap_positions[(current_length, start_index)] = [seq_index]
            gaps_length_histogram[current_length] += 1
        else:
            current_index = len(sequence)
            if current_index == last_gap_index + 2:
                single_char_count += 1
            elif current_index == last_gap_index + 3:
                double_char_count += 1

        self.num_gap_segments = sum(count for length, count in gaps_length_histogram.items())
        self.single_char_count += single_char_count
        self.double_char_count += double_char_count

    def calculate_counts(self, gap_positions: dict) -> None:
        length_count = {1: Counter(), 2: Counter(), 3: Counter()}
        length_plus_count = Counter()
        total_length = 0
        unique_gaps = 0
        unique_gaps_length = 0

        for (length, start_index), seq_set in gap_positions.items():
            if length == 1:
                length_count[length][len(seq_set)] += 1
            if length == 2:
                length_count[length][len(seq_set)] += 1
            if length == 3:
                length_count[length][len(seq_set)] += 1
            if length > 3:
                length_plus_count[len(seq_set)] += 1
            if len(seq_set) == 1:
                unique_gaps += 1
                unique_gaps_length += length
            total_length += len(seq_set) * length

        self.av_gaps = total_length / self.num_gap_segments
        self.num_unique_gaps = unique_gaps
        self.avg_unique_gap = unique_gaps_length / max(unique_gaps, 1)
        self.gaps_1seq_len1 = length_count[1][1]
        self.gaps_2seq_len1 = length_count[1][2]
        self.gaps_all_except_1_len1 = length_count[1][self.taxa_num - 1]
        self.gaps_1seq_len2 = length_count[2][1]
        self.gaps_2seq_len2 = length_count[2][2]
        self.gaps_all_except_1_len2 = length_count[2][self.taxa_num - 1]
        self.gaps_1seq_len3 = length_count[3][1]
        self.gaps_2seq_len3 = length_count[3][2]
        self.gaps_all_except_1_len3 = length_count[3][self.taxa_num - 1]
        self.gaps_1seq_len3plus = length_plus_count[1]
        self.gaps_2seq_len3plus = length_plus_count[2]
        self.gaps_all_except_1_len3plus = length_plus_count[self.taxa_num - 1]
