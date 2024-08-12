import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO
from collections import defaultdict, Counter
from typing import Optional, Tuple
from Bio import Seq
import os
import sys
from feature_extraction.pypythia.predictor import DifficultyPredictor
from feature_extraction.pypythia.prediction import get_all_features
from feature_extraction.pypythia.raxmlng import RAxMLNG
from feature_extraction.pypythia.msa import MSA

ROOT_DIRECRTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIRECRTORY)
RAXML_NG_EXE = "raxml-ng"

class Statistics:
    def __init__(self, msa_file: str, code: str):
        self.aln = None
        self.taxa_num = 0
        self.stats = {"code": code, 'taxa_num': 0, "constant_sites_pct": 0, "n_unique_sites": 0,
                      "pypythia_msa_difficulty": 0, "entropy_mean": 0,
                      "entropy_median": 0, "entropy_var": 0, "entropy_pct_25": 0,
                      "entropy_pct_75": 0, "entropy_min": 0, "entropy_max": 0, "av_gaps": 0, "msa_len": -1,
                      "seq_max_len": -1,
                      "seq_min_len": -1, "total_gaps": 0, "gaps_len_one": 0, "gaps_len_two": 0, "gaps_len_three": 0,
                      "gaps_len_three_plus": 0, "avg_unique_gap": 0, "num_unique_gaps": 0, "gaps_1seq_len1": 0,
                      "gaps_2seq_len1": 0, "gaps_all_except_1_len1": 0,
                      "gaps_1seq_len2": 0, "gaps_2seq_len2": 0, "gaps_all_except_1_len2": 0, "gaps_1seq_len3": 0,
                      "gaps_2seq_len3": 0, "gaps_all_except_1_len3": 0, "gaps_1seq_len3plus": 0,
                      "gaps_2seq_len3plus": 0, "gaps_all_except_1_len3plus": 0, "num_cols_no_gaps": 0,
                      "num_cols_1_gap": 0,
                      "num_cols_2_gaps": 0, "num_cols_all_gaps_except1": 0, "msa_path": msa_file}

    def set_alignment(self) -> Optional[AlignIO.MultipleSeqAlignment]:
        try:
            with open(f"{self.stats['msa_path']}") as file:
                self.aln = AlignIO.read(file, "fasta")
            return self.aln
        except Exception as e:
            print(
                f"Didn't manage to get an alignment from the {self.stats['msa_path']}. Check the file path and contents \n")
            return None

    def set_length(self) -> int:
        self.stats["msa_len"] = self.aln.get_alignment_length()
        return int(self.stats["msa_len"])

    def set_taxa_num(self) -> None:
        self.taxa_num = len(self.aln)
        self.stats['taxa_num'] = self.taxa_num

    def set_values(self) -> None:
        min_length = 1000000000
        max_length = -1
        total_gap_char = 0
        i = 0
        self.gap_positions = {}
        self.total_gaps_count = defaultdict(int)
        #
        for seq_index, record in enumerate(self.aln):
            i += 1
            total_gap_char += str(record.seq).count('-')
            len_no_gaps = len(str(record.seq).replace('-', ''))
            if len_no_gaps < min_length:
                min_length = len_no_gaps
            if len_no_gaps > max_length:
                max_length = len_no_gaps
            self.record_gap_lengths(record.seq, seq_index)
        self.calculate_counts()

        # per column
        for pos in range(self.stats['msa_len']):
            # Extract the column by iterating over all sequences
            column = [record.seq[pos] for record in self.aln]

            num_gaps = column.count('-')
            # Check if any sequence in this column contains a gap
            if num_gaps == 0:
                self.stats['num_cols_no_gaps'] += 1
            elif num_gaps == 1:
                self.stats['num_cols_1_gap'] += 1
            elif num_gaps == 2:
                self.stats['num_cols_2_gaps'] += 1
            elif num_gaps == (self.taxa_num - 1):
                self.stats['num_cols_all_gaps_except1'] += 1
        self.stats["seq_min_len"] = min_length
        self.stats["seq_max_len"] = max_length
        self.stats["gaps_len_one"] = self.total_gaps_count[1]  # double counts the "same" gap in different sequences
        self.stats["gaps_len_two"] = self.total_gaps_count[2]  # double counts the "same" gap in different sequences
        self.stats["gaps_len_three"] = self.total_gaps_count[3]  # double counts the "same" gap in different sequences
        self.stats["gaps_len_three_plus"] = sum(count for length, count in self.total_gaps_count.items() if
                                                length > 3)  # double counts the "same" gap in different sequences

        # Noa's part incorporated
        alignment_df, alignment_df_fixed, alignment_df_unique = self.get_alignment_df()
        counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
        probabilities = [
            list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
            for col in
            list(alignment_df)]
        entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
        self.stats["constant_sites_pct"] = sum([1 for et in entropy if et == 0]) / len(entropy)
        self.stats["n_unique_sites"] = len(alignment_df_unique.columns)
        try:  ### doesn't work, needs to be fixed!!
            self.stats["pypythia_msa_difficulty"] = self.pypythia(model="GTR+G")
        except Exception:
            self.stats["pypythia_msa_difficulty"] = -1

        multi_dimensional_features = {"entropy": entropy}
        for feature in multi_dimensional_features:
            new_dict = self.get_summary_statistics_dict(feature, multi_dimensional_features[feature])
            for key, value in new_dict.items():
                self.stats[key] = value

    def get_min_len(self) -> int:
        return int(self.stats["seq_min_len"])

    def get_stats(self) -> dict:
        return self.stats

    def get_max_len(self) -> int:
        return int(self.stats["seq_max_len"])

    def record_gap_lengths(self, sequence: Seq.Seq, seq_index: int) -> None:
        current_length = 0
        start_index = None

        for i, char in enumerate(sequence):
            if char == '-':
                if current_length == 0:
                    start_index = i
                current_length += 1
            else:
                if current_length > 0:
                    if (current_length, start_index) in self.gap_positions and seq_index not in self.gap_positions[
                        (current_length, start_index)]:
                        self.gap_positions[(current_length, start_index)].append(seq_index)
                    else:
                        self.gap_positions[(current_length, start_index)] = [seq_index]
                    self.total_gaps_count[current_length] += 1
                current_length = 0

        # Record if the sequence ends with gaps
        if current_length > 0:
            if (current_length, start_index) in self.gap_positions and seq_index not in self.gap_positions[
                (current_length, start_index)]:
                self.gap_positions[(current_length, start_index)].append(seq_index)
            else:
                self.gap_positions[(current_length, start_index)] = [seq_index]
            self.total_gaps_count[current_length] += 1

        self.stats["total_gaps"] = sum(count for length, count in self.total_gaps_count.items())

    def calculate_counts(self) -> None:
        length_count = {1: Counter(), 2: Counter(), 3: Counter()}
        length_plus_count = Counter()
        total_length = 0
        num_of_gaps = 0
        unique_gaps = 0
        unique_gaps_length = 0

        for (length, start_index), seq_set in self.gap_positions.items():
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
            num_of_gaps += len(seq_set)
            total_length += len(seq_set) * length

        self.stats["av_gaps"] = total_length / num_of_gaps
        self.stats["num_unique_gaps"] = unique_gaps
        self.stats["avg_unique_gap"] = unique_gaps_length / unique_gaps
        self.stats["gaps_1seq_len1"] = length_count[1][1]
        self.stats["gaps_2seq_len1"] = length_count[1][2]
        self.stats["gaps_all_except_1_len1"] = length_count[1][self.taxa_num - 1]
        self.stats["gaps_1seq_len2"] = length_count[2][1]
        self.stats["gaps_2seq_len2"] = length_count[2][2]
        self.stats["gaps_all_except_1_len2"] = length_count[2][self.taxa_num - 1]
        self.stats["gaps_1seq_len3"] = length_count[3][1]
        self.stats["gaps_2seq_len3"] = length_count[3][2]
        self.stats["gaps_all_except_1_len3"] = length_count[3][self.taxa_num - 1]
        self.stats["gaps_1seq_len3plus"] = length_plus_count[1]
        self.stats["gaps_2seq_len3plus"] = length_plus_count[2]
        self.stats["gaps_all_except_1_len3plus"] = length_plus_count[self.taxa_num - 1]

    def alignment_list_to_df(self, alignment_data: list) -> pd.DataFrame:
        alignment_list = [list(alignment_data[i].seq) for i in range(len(alignment_data))]
        loci_num = len(alignment_data[0].seq)
        columns = list(range(0, loci_num))
        original_alignment_df = pd.DataFrame(alignment_list, columns=columns)
        return original_alignment_df

    def get_alignment_df(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | int:
        try:
            with open(self.stats["msa_path"]) as file:
                data = list(SeqIO.parse(file, 'fasta'))
                if len(data) == 0:
                    raise Exception("zero value")
        except:
            try:
                with open(self.stats["msa_path"]) as file:
                    data = list(SeqIO.parse(file, 'phylip-relaxed'))
            except:
                return -1
        original_alignment_df = self.alignment_list_to_df(data)
        alignment_df_fixed = original_alignment_df.replace('-', np.nan)
        alignment_df_unique = original_alignment_df.T.drop_duplicates().T
        return original_alignment_df, alignment_df_fixed, alignment_df_unique

    import pickle
    def pypythia(self, model):
        with open(f"{ROOT_DIRECRTORY}/pypythia/predictor.pckl", "rb") as file:
            predictor = DifficultyPredictor(file)
        raxmlng = RAxMLNG(RAXML_NG_EXE)
        msa = MSA(self.stats["msa_path"])
        msa_features = get_all_features(raxmlng, msa, model)
        difficulty = predictor.predict(msa_features)
        return difficulty

    def pct_25(values):
        return np.percentile(values, 25)

    def pct_75(values):
        return np.percentile(values, 75)

    def get_summary_statistics_dict(self, feature_name, values,
                                    funcs={'mean': np.mean, 'median': np.mean, 'var': np.var, 'pct_25': pct_25,
                                           'pct_75': pct_75, 'min': np.min, 'max': np.max, }):
        res = {}
        for func in funcs:
            res.update({f'{feature_name}_{func}': (funcs[func](values))})
        return res