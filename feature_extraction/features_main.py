from classes.statistics_calculator import Statistics
import os
import csv
import sys

ROOT_DIRECRTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIRECRTORY)


def main(base_folder):
    folder = f"{base_folder}/MSAs/"
    # need to add num_taxa
    columns_all = ['code', 'taxa_num', 'constant_sites_pct', 'n_unique_sites', 'pypythia_msa_difficulty', 'entropy_mean',
                   'entropy_median', 'entropy_var', 'entropy_pct_25', 'entropy_pct_75', 'entropy_min', 'entropy_max',
                   'av_gaps', 'msa_len', 'seq_max_len', 'seq_min_len', 'total_gaps', 'gaps_len_one', 'gaps_len_two',
                   'gaps_len_three', 'gaps_len_three_plus', 'avg_unique_gap', 'num_unique_gaps', 'gaps_1seq_len1',
                   'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
                   'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
                   'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'num_cols_no_gaps',
                   'num_cols_1_gap', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'msa_path']

    with open(f"{base_folder}/all_features_ALL_NEW_541.csv", mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns_all)
        writer.writeheader()

        for code in os.listdir(folder):
                if code.startswith('.DS_Store'):
                    continue
                for msa in os.listdir(f"{folder}/{code}/"):
                    try:
                        statistics_calculator = Statistics(f"{folder}/{code}/{msa}", code)
                        statistics_calculator.set_alignment()
                        statistics_calculator.set_length()
                        statistics_calculator.set_taxa_num()
                        statistics_calculator.set_values()
                        all_msa_features = statistics_calculator.get_stats()
                        writer.writerow(all_msa_features)

                        print(code + "\n")

                    except Exception as e:
                        print(f"failed to get features from {code}: exception {e}")

if __name__ == '__main__':
    base_folder = "/Users/kpolonsky/Downloads/TEST/"
    main(base_folder)