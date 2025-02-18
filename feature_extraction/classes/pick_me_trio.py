import numpy as np
import pandas as pd
from typing import Literal

from matplotlib import pyplot as plt


class PickMeGameTrio:
    def __init__(self, features_file: str, prediction_file: str, predicted_measure: Literal['msa_distance', 'tree_distance'] = 'msa_distance', error: float = 0.0) -> None:
        self.features_file = features_file
        self.prediction_file = prediction_file
        self.error = error
        self.true_score = ''
        self.predicted_score = 'predicted_score'
        # self.sum_of_pairs_score = 'normalised_sop_score'
        self.sum_of_pairs_score = 'sop_score'
        self.pickme_df = None
        self.pickme_sop_df = None
        self.accumulated_data = {}
        self.accumulated_sop_data = {}
        self.results = []
        self.winners = []
        self.overall_win = []

        # Load the two CSV files into DataFrames
        df1 = pd.read_csv(self.features_file)
        df2 = pd.read_csv(self.prediction_file)

        df1['code1'] = df1['code1'].astype(str)
        df2['code1'] = df2['code1'].astype(str)

        df = pd.merge(df1, df2, on=['code', 'code1'], how='inner')
        df = df[~df['code'].str.contains('test_original', na=False)]
        # groups = ['BBS11','BBS12','BBS50','BBS30','BBS20', 'BBA']

        if predicted_measure == "msa_distance":
            self.true_score = 'dpos_dist_from_true'


        # results = []
        # winners = []
        # overall_win = []
        for code in df['code1'].unique():
            # if not code.startswith(groups[5]):
            #     continue
            code_df = df[df['code1'] == code]
            substrings = ['original', 'concat']
            mask = code_df['code'].str.contains('|'.join(substrings), case=False, na=False)
            code_df = code_df[~mask]


            # True_score for 'MSA.MAFFT.aln.With_Names'
            default_mafft_true_scores = code_df[code_df['code'] == 'MSA.MAFFT.aln.With_Names'][self.true_score].values
            default_mafft_true_score = default_mafft_true_scores[0] if len(default_mafft_true_scores) > 0 else np.nan

            # True_score for 'MSA.PRANK.aln.best.fas'
            default_prank_true_scores = code_df[code_df['code'].isin(['MSA.PRANK.aln.With_Names', 'MSA.PRANK.aln.best.fas'])][
                self.true_score].values
            default_prank_true_score = default_prank_true_scores[0] if len(default_prank_true_scores) > 0 else np.nan

            # True_score for code_filename of the form 'MSA.MUSCLE.aln.best.{code}.fas'
            default_muscle_true_scores = \
            code_df[code_df['code'].str.contains(r'^MSA\.MUSCLE\.aln\.best\.[\w]+\.fas$', regex=True)][
                self.true_score].values
            default_muscle_true_score = default_muscle_true_scores.min() if len(default_muscle_true_scores) > 0 else np.nan

            # True_score for BALI-PHY DEFAULT
            # default_baliphy_true_scores = code_df[code_df['code'] == 'bali_phy_msa.199.fasta'][self.true_score].values
            default_baliphy_true_scores = \
                code_df[code_df['code'].str.contains(r'^MSA\.BALIPHY\.aln\.best\.[\w]+\.fas$', regex=True)][
                    self.true_score].values
            default_baliphy_true_score = default_baliphy_true_scores[0] if len(default_baliphy_true_scores) > 0 else np.nan

            # Minimum true_score and filename among MAFFT alternative MSAs
            substrings = ['muscle', 'prank', '_TRUE.fas', 'true_tree.txt', 'bali_phy', 'BALIPHY', 'original']
            mask = code_df['code'].str.contains('|'.join(substrings), case=False, na=False)
            mafft_df = code_df[~mask]
            mafft_scores = []
            if not mafft_df.empty:
                # TRUE MAFFT MIN --> TRUE BEST
                min_mafft_row = mafft_df.loc[mafft_df[self.true_score].idxmin()]
                min_mafft_true_score = min_mafft_row[self.true_score]
                mafft_scores.append(default_mafft_true_score)
                min_mafft_code_filename = min_mafft_row['code']

                # PREDICTED MAFFT MIN --> MAFFT BEST PREDICTED
                min_mafft_predicted_score_row = mafft_df.loc[mafft_df[self.predicted_score].idxmin()]
                min_mafft_predicted_filename = min_mafft_predicted_score_row['code']
                min_mafft_predicted_true_score = min_mafft_predicted_score_row[self.true_score]
                mafft_scores.append(min_mafft_predicted_true_score)
                min_mafft_predicted_score = min_mafft_predicted_score_row[self.predicted_score]
                sorted_mafft_temp_df = mafft_df.sort_values(by=self.predicted_score, ascending=True)
                top_20_mafft_rows = sorted_mafft_temp_df.head(20)
                top_20_mafft_filenames = top_20_mafft_rows['code'].tolist()
                top_20_mafft_scores = top_20_mafft_rows[self.true_score].tolist()

                # MAX SOP MAFFT --> BEST SOP MAFFT
                max_mafft_SoP_score_row = mafft_df.loc[mafft_df[self.sum_of_pairs_score].idxmax()]
                max_mafft_SoP_filename = max_mafft_SoP_score_row['code']
                max_mafft_SoP_true_score = max_mafft_SoP_score_row[self.true_score]
                mafft_scores.append(max_mafft_SoP_true_score)
                max_mafft_SoP_score = max_mafft_SoP_score_row[self.sum_of_pairs_score]
                sorted_mafft_temp_df = mafft_df.sort_values(by=self.sum_of_pairs_score, ascending=False)
                top_20_mafft_SoP_rows = sorted_mafft_temp_df.head(20)
                top_20_mafft_SoP_filenames = top_20_mafft_SoP_rows['code'].tolist()
                top_20_mafft_SoP_scores = top_20_mafft_SoP_rows[self.true_score].tolist()

            else:
                print(f"no mafft for code {code}")
                min_mafft_true_score = np.nan
                min_mafft_code_filename = np.nan
                min_mafft_predicted_filename = np.nan
                min_mafft_predicted_true_score = np.nan
                min_mafft_predicted_score = np.nan
                top_20_mafft_scores = []

                max_mafft_SoP_filename = np.nan
                max_mafft_SoP_true_score = np.nan
                max_mafft_SoP_score = np.nan
                top_20_mafft_SoP_scores = []


            # Minimum true_score and code_filename among PRANK alternative MSAs
            prank_df = code_df[code_df['code'].str.contains('prank', case=False, na=False, regex=True)]
            prank_scores = []
            if not prank_df.empty:
                # TRUE PRANK MIN = TRUE BEST
                min_prank_row = prank_df.loc[prank_df[self.true_score].idxmin()]
                min_prank_true_score = min_prank_row[self.true_score]
                prank_scores.append(default_prank_true_score)
                min_prank_code_filename = min_prank_row['code']

                # PREDICTED PRANK MIN = PRANK BEST PREDICTED
                min_prank_predicted_score_row = prank_df.loc[prank_df[self.predicted_score].idxmin()]
                min_prank_predicted_filename = min_prank_predicted_score_row['code']
                min_prank_predicted_true_score = min_prank_predicted_score_row[self.true_score]
                prank_scores.append(min_prank_predicted_true_score)
                min_prank_predicted_score = min_prank_predicted_score_row[self.predicted_score]
                sorted_prank_temp_df = prank_df.sort_values(by=self.predicted_score, ascending=True)
                top_20_prank_rows = sorted_prank_temp_df.head(20)
                top_20_prank_filenames = top_20_prank_rows['code'].tolist()
                top_20_prank_scores = top_20_prank_rows[self.true_score].tolist()

                # MAX SOP PRANK = BEST SOP PRANK
                max_prank_SoP_score_row = prank_df.loc[prank_df[self.sum_of_pairs_score].idxmax()]
                max_prank_SoP_filename = max_prank_SoP_score_row['code']
                max_prank_SoP_true_score = max_prank_SoP_score_row[self.true_score]
                prank_scores.append(max_prank_SoP_true_score)
                max_prank_SoP_score = max_prank_SoP_score_row[self.sum_of_pairs_score]
                sorted_prank_temp_df = prank_df.sort_values(by=self.sum_of_pairs_score, ascending=False)
                top_20_prank_SoP_rows = sorted_prank_temp_df.head(20)
                top_20_prank_SoP_filenames = top_20_prank_SoP_rows['code'].tolist()
                top_20_prank_SoP_scores = top_20_prank_SoP_rows[self.true_score].tolist()

            else:
                print(f"no prank files for code {code}")
                min_prank_true_score = np.nan
                min_prank_code_filename = np.nan
                min_prank_predicted_filename = np.nan
                min_prank_predicted_true_score = np.nan
                min_prank_predicted_score = np.nan
                top_20_prank_scores = []
                max_prank_SoP_filename = np.nan
                max_prank_SoP_true_score = np.nan
                max_prank_SoP_score = np.nan
                top_20_prank_SoP_scores = []


            # Minimum true_score and code_filename among MUSCLE alternative MSAs
            muscle_df = code_df[code_df['code'].str.contains('muscle', case=False, na=False, regex=True)]
            muscle_scores = []
            if not muscle_df.empty:
                min_muscle_row = muscle_df.loc[muscle_df[self.true_score].idxmin()]
                min_muscle_true_score = min_muscle_row[self.true_score]
                muscle_scores.append(default_muscle_true_score)
                min_muscle_code_filename = min_muscle_row['code']

                min_muscle_predicted_score_row = muscle_df.loc[muscle_df[self.predicted_score].idxmin()]
                min_muscle_predicted_filename = min_muscle_predicted_score_row['code']
                min_muscle_predicted_true_score = min_muscle_predicted_score_row[self.true_score]
                muscle_scores.append(min_muscle_predicted_true_score)
                min_muscle_predicted_score = min_muscle_predicted_score_row[self.predicted_score]
                sorted_muscle_temp_df = muscle_df.sort_values(by=self.predicted_score, ascending=True)
                top_20_muscle_rows = sorted_muscle_temp_df.head(20)
                top_20_muscle_filenames = top_20_muscle_rows['code'].tolist()
                top_20_muscle_scores = top_20_muscle_rows[self.true_score].tolist()

                max_muscle_SoP_score_row = muscle_df.loc[muscle_df[self.sum_of_pairs_score].idxmax()]
                max_muscle_SoP_filename = max_muscle_SoP_score_row['code']
                max_muscle_SoP_true_score = max_muscle_SoP_score_row[self.true_score]
                muscle_scores.append(max_muscle_SoP_true_score)
                max_muscle_SoP_score = max_muscle_SoP_score_row[self.sum_of_pairs_score]
                sorted_muscle_temp_df = muscle_df.sort_values(by=self.sum_of_pairs_score, ascending=False)
                top_20_muscle_SoP_rows = sorted_muscle_temp_df.head(20)
                top_20_muscle_SoP_filenames = top_20_muscle_SoP_rows['code'].tolist()
                top_20_muscle_SoP_scores = top_20_muscle_SoP_rows[self.true_score].tolist()

            else:
                print(f"no muscle files for {code}")
                min_muscle_true_score = np.nan
                min_muscle_code_filename = np.nan
                min_muscle_predicted_filename = np.nan
                min_muscle_predicted_true_score = np.nan
                min_muscle_predicted_score = np.nan
                top_20_muscle_scores = []
                max_muscle_SoP_filename = np.nan
                max_muscle_SoP_true_score = np.nan
                max_muscle_SoP_score = np.nan
                top_20_muscle_SoP_scores = []


            # Minimum true_score and code_filename among Bali-Phy alternative MSAs
            baliphy_df = code_df[code_df['code'].str.contains('bali_phy|BALIPHY', case=False, na=False, regex=True)]
            baliphy_scores = []
            if not baliphy_df.empty:
                min_baliphy_row = baliphy_df.loc[baliphy_df[self.true_score].idxmin()]
                min_baliphy_true_score = min_baliphy_row[self.true_score]
                baliphy_scores.append(default_baliphy_true_score)
                min_baliphy_code_filename = min_baliphy_row['code']

                min_baliphy_predicted_score_row = baliphy_df.loc[baliphy_df[self.predicted_score].idxmin()]
                min_baliphy_predicted_filename = min_baliphy_predicted_score_row['code']
                min_baliphy_predicted_true_score = min_baliphy_predicted_score_row[self.true_score]
                baliphy_scores.append(min_baliphy_predicted_true_score)
                min_baliphy_predicted_score = min_baliphy_predicted_score_row[self.predicted_score]
                sorted_baliphy_temp_df = baliphy_df.sort_values(by=self.predicted_score, ascending=True)
                top_20_baliphy_rows = sorted_baliphy_temp_df.head(20)
                top_20_baliphy_filenames = top_20_baliphy_rows['code'].tolist()
                top_20_baliphy_scores = top_20_baliphy_rows[self.true_score].tolist()

                max_baliphy_SoP_score_row = baliphy_df.loc[baliphy_df[self.sum_of_pairs_score].idxmax()]
                max_baliphy_SoP_filename = max_baliphy_SoP_score_row['code']
                max_baliphy_SoP_true_score = max_baliphy_SoP_score_row[self.true_score]
                baliphy_scores.append(max_baliphy_SoP_true_score)
                max_baliphy_SoP_score = max_baliphy_SoP_score_row[self.sum_of_pairs_score]
                sorted_baliphy_temp_df = baliphy_df.sort_values(by=self.sum_of_pairs_score, ascending=False)
                top_20_baliphy_SoP_rows = sorted_baliphy_temp_df.head(20)
                top_20_baliphy_SoP_filenames = top_20_baliphy_SoP_rows['code'].tolist()
                top_20_baliphy_SoP_scores = top_20_baliphy_SoP_rows[self.true_score].tolist()

            else:
                min_baliphy_true_score = np.nan
                min_baliphy_code_filename = np.nan

                max_baliphy_SoP_filename = np.nan
                min_baliphy_predicted_true_score = np.nan
                min_baliphy_predicted_score = np.nan
                top_20_baliphy_scores = []
                min_baliphy_predicted_filename = np.nan
                max_baliphy_SoP_true_score = np.nan
                max_baliphy_SoP_score = np.nan
                top_20_baliphy_SoP_scores = []
                min_baliphy_SoP_filename = np.nan

            # Minimum true_score code_filename
            # overall_scores = []
            # min_true_score_row = code_df.loc[code_df[self.true_score].idxmin()]
            # min_true_score_code_filename = min_true_score_row['code']
            # min_true_score_value = min_true_score_row[self.true_score]
            # overall_scores.append(min_true_score_value)

            min_true_score_row = code_df.loc[code_df[self.true_score].idxmin()]
            min_true_score_code_filename = min_true_score_row['code']
            min_true_score_value = min_true_score_row[self.true_score]

            # # Code_filename, predicted_score, and true_score of the filename with the minimum predicted_score
            min_predicted_score_row = code_df.loc[code_df[self.predicted_score].idxmin()]
            min_predicted_filename = min_predicted_score_row['code']
            min_predicted_true_score = min_predicted_score_row[self.true_score]
            min_predicted_score = min_predicted_score_row[self.predicted_score]
            sorted_temp_df_pred = code_df.sort_values(by=self.predicted_score, ascending=True)
            top_20_rows = sorted_temp_df_pred.head(20)
            top_20_filenames = top_20_rows['code'].tolist()
            top_20_scores = top_20_rows[self.true_score].tolist()

            # Code_filename, predicted_score, and true_score of the filename with the minimum predicted_score
            max_SoP_score_row = code_df.loc[code_df[self.sum_of_pairs_score].idxmax()]
            max_SoP_filename = max_SoP_score_row['code']
            max_SoP_true_score = max_SoP_score_row[self.true_score]
            max_SoP_score = max_SoP_score_row[self.sum_of_pairs_score]
            sorted_temp_df_sop = code_df.sort_values(by=self.sum_of_pairs_score, ascending=False)
            top_20_SoP_rows = sorted_temp_df_sop.head(20)
            top_20_SoP_filenames = top_20_SoP_rows['code'].tolist()
            top_20_SoP_scores = top_20_SoP_rows[self.true_score].tolist()


            overall_scores = []
            # overall_scores.append(min_true_score_value)
            overall_scores.append(min_predicted_true_score)
            overall_scores.append(max_SoP_true_score)
            overall_scores.append(default_mafft_true_score)
            overall_scores.append(default_prank_true_score)
            overall_scores.append(default_muscle_true_score)
            overall_scores.append(default_baliphy_true_score)
            # result_map = {0: "true min", 1: "predicted", 2: "sop", 3: "mafft", 4: "prank", 5: "muscle", 6: "baliphy"}
            result_map = {0: "min predicted", 1: "max sop", 2: "default mafft", 3: "default prank", 4: "default muscle", 5: "default baliphy"}
            index_of_min = overall_scores.index(min(overall_scores))
            overall_winner = result_map.get(int(index_of_min), "Unknown")

            #define winners
            # result_map = {0: "default", 1: "predicted", 2: "sop"}
            # if overall_scores:
            #     if len(set(overall_scores)) == 3:
            #         index_of_min = overall_scores.index(min(overall_scores))
            #         overall_winner = result_map.get(int(index_of_min), "Unknown")
            #     elif len(set(overall_scores)) == 2 and overall_scores[0] == overall_scores[1]: #true == predicted
            #         overall_winner = "predicted"
            #     elif len(set(overall_scores)) == 2 and overall_scores[0] == overall_scores[2]: #true == sop
            #         overall_winner = "sop"
            #     elif len(set(overall_scores)) == 2 and overall_scores[1] == overall_scores[2]: # predicted == sop
            #         overall_winner = "tie2(predicted=sop)"
            #     elif len(set(overall_scores)) == 1:
            #         overall_winner = "tie(predicted=sop=true)"
            # else:
            #     overall_winner = np.nan

            # mafft_winner, prank_winner, muscle_winner, baliphy_winner = np.nan

            # if overall_scores:
            #     if len(set(overall_scores)) == 3:
            #         index_of_min = overall_scores.index(min(overall_scores))
            #         overall_winner = result_map.get(int(index_of_min), "Unknown")
            #     elif len(set(overall_scores)) == 2 and overall_scores[0] == overall_scores[1]: #true == predicted
            #         overall_winner = "predicted"
            #     elif len(set(overall_scores)) == 2 and overall_scores[0] == overall_scores[2]: #true == sop
            #         overall_winner = "sop"
            #     elif len(set(overall_scores)) == 2 and overall_scores[1] == overall_scores[2]: # predicted == sop
            #         overall_winner = "tie2(predicted=sop)"
            #     elif len(set(overall_scores)) == 1:
            #         overall_winner = "tie(predicted=sop=true)"
            # else:
            #     overall_winner = np.nan

            # result_map = {0: "default", 1: "predicted", 2: "sop"}
            if mafft_scores:
                if mafft_scores[1] <= mafft_scores[0] and mafft_scores[1] < mafft_scores[2]:  # predicted <= default and predicted < sop
                    mafft_winner = "predicted"
                elif mafft_scores[2] <= mafft_scores[0] and mafft_scores[2] < mafft_scores[1]:  # sop <= default and sop < predicted
                    mafft_winner = "sop"
                elif mafft_scores[0] < mafft_scores[1] and mafft_scores[0] < mafft_scores[2]:  # default < predicted and default < sop
                    mafft_winner = "default"
                elif mafft_scores[0] == mafft_scores[1] == mafft_scores[2]:  # default = predicted = sop
                    mafft_winner = "tie(all three)"
                elif mafft_scores[1] < mafft_scores[0] and mafft_scores[1] == mafft_scores[2]:  # predicted = sop < default
                    mafft_winner = "tie(predicted and sop)"
                # elif mafft_scores[1] == mafft_scores[0] and mafft_scores[1] < mafft_scores[2]:  # predicted = default < sop
                #     # mafft_winner = "tie(predicted and default)"
                #     mafft_winner = "predicted"
                # elif mafft_scores[2] == mafft_scores[0] and mafft_scores[2] < mafft_scores[1]:  # sop = default < predicted
                #     # mafft_winner = "tie(sop and default)"
                #     mafft_winner = "sop"
                # elif mafft_scores[1] == mafft_scores[2] and mafft_scores[0] < mafft_scores[1]:  # sop = default < predicted
                #     # mafft_winner = "tie(sop and predicted)"
                #     mafft_winner = "default"
                else:
                    print(f"who are you mafft? {mafft_scores}\n")
                    mafft_winner = np.nan
            else:
                mafft_winner = np.nan

            if prank_scores:
                if prank_scores[1] <= prank_scores[0] and prank_scores[1] < prank_scores[2]:  # predicted < default and predicted < sop
                    prank_winner = "predicted"
                elif prank_scores[2] <= prank_scores[0] and prank_scores[2] < prank_scores[1]:  # sop < default and sop < predicted
                    prank_winner = "sop"
                elif prank_scores[0] < prank_scores[1] and prank_scores[0] < prank_scores[2]:  # default < predicted and default < sop
                    prank_winner = "default"
                elif prank_scores[0] == prank_scores[1] == prank_scores[2]:  # default = predicted = sop
                    prank_winner = "tie(all three)"
                elif prank_scores[1] < prank_scores[0] and prank_scores[1] == prank_scores[2]:  # predicted = sop < default
                    prank_winner = "tie(predicted and sop)"
                # elif prank_scores[1] == prank_scores[0] and prank_scores[1] < prank_scores[2]:  # predicted = default < sop
                #     # prank_winner = "tie(predicted and default)"
                #     prank_winner = "predicted"
                # elif prank_scores[2] == prank_scores[0] and prank_scores[2] < prank_scores[1]:  # sop = default < predicted
                #     # prank_winner = "tie(sop and default)"
                #     prank_winner = "sop"
                # elif prank_scores[1] == prank_scores[2] and prank_scores[0] < prank_scores[1]:  # sop = default < predicted
                #     # mafft_winner = "tie(sop and predicted)"
                #     prank_winner = "default"
                else:
                    print(f"who are you prank? {prank_scores}\n")
                    prank_winner = np.nan
            else:
                prank_winner = np.nan

            if muscle_scores:
                if muscle_scores[1] <= muscle_scores[0] and muscle_scores[1] < muscle_scores[2]:  # predicted < default and predicted < sop
                    muscle_winner = "predicted"
                elif muscle_scores[2] <= muscle_scores[0] and muscle_scores[2] < muscle_scores[1]:  # sop < default and sop < predicted
                    muscle_winner = "sop"
                elif muscle_scores[0] < muscle_scores[1] and muscle_scores[0] < muscle_scores[2]:  # default < predicted and default < sop
                    muscle_winner = "default"
                elif muscle_scores[0] == muscle_scores[1] == muscle_scores[2]:  # default = predicted = sop
                    muscle_winner = "tie(all three)"
                elif muscle_scores[1] < muscle_scores[0] and muscle_scores[1] == muscle_scores[2]:  # predicted = sop < default
                    muscle_winner = "tie(predicted and sop)"
                # elif muscle_scores[1] == muscle_scores[0] and muscle_scores[1] < muscle_scores[2]:  # predicted = default < sop
                #     # muscle_winner = "tie(predicted and default)"
                #     muscle_winner = "predicted"
                # elif muscle_scores[2] == muscle_scores[0] and muscle_scores[2] < muscle_scores[1]:  # sop = default < predicted
                #     # muscle_winner = "tie(sop and default)"
                #     muscle_winner = "sop"
                # elif muscle_scores[1] == muscle_scores[2] and muscle_scores[0] < muscle_scores[1]:  # sop = default < predicted
                #     # mafft_winner = "tie(sop and predicted)"
                #     muscle_winner = "default"
                else:
                    print(f"who are you muscle? {muscle_scores}\n")
                    muscle_winner = np.nan
            else:
                muscle_winner = np.nan

            if baliphy_scores:
                if baliphy_scores[1] <= baliphy_scores[0] and baliphy_scores[1] < baliphy_scores[2]:  # predicted < default and predicted < sop
                    baliphy_winner = "predicted"
                elif baliphy_scores[2] <= baliphy_scores[0] and baliphy_scores[2] < baliphy_scores[1]:  # sop < default and sop < predicted
                    baliphy_winner = "sop"
                elif baliphy_scores[0] < baliphy_scores[1] and baliphy_scores[0] < baliphy_scores[2]:  # default < predicted and default < sop
                    baliphy_winner = "default"
                elif baliphy_scores[0] == baliphy_scores[1] == baliphy_scores[2]:  # default = predicted = sop
                    baliphy_winner = "tie(all three)"
                elif baliphy_scores[1] < baliphy_scores[0] and baliphy_scores[1] == baliphy_scores[2]:  # predicted = sop < default
                    baliphy_winner = "tie(predicted and sop)"
                # elif baliphy_scores[1] == baliphy_scores[0] and baliphy_scores[1] < baliphy_scores[2]:  # predicted = default < sop
                #     # baliphy_winner = "tie(predicted and default)"
                #     baliphy_winner = "predicted"
                # elif baliphy_scores[2] == baliphy_scores[0] and baliphy_scores[2] < baliphy_scores[1]:  # sop = default < predicted
                #     # baliphy_winner = "tie(sop and default)"
                #     baliphy_winner = "sop"
                # elif baliphy_scores[1] == baliphy_scores[2] and baliphy_scores[0] < baliphy_scores[1]:  # sop = default < predicted
                #     # mafft_winner = "tie(sop and predicted)"
                #     baliphy_winner = "default"
                else:
                    print(f"who are you baliphy? {baliphy_scores}\n")
                    baliphy_winner = np.nan
            else:
                baliphy_winner = np.nan

            # if mafft_scores:
                # if len(set(mafft_scores)) == 3:
                #     index_of_min = mafft_scores.index(min(mafft_scores))
                #     mafft_winner = result_map.get(int(index_of_min), "Unknown")
                # elif len(set(mafft_scores)) == 2 and mafft_scores[1] < mafft_scores[0] and mafft_scores[1] < mafft_scores[2]:  # predicted < default and predicted < sop
                #     mafft_winner = "predicted"
                # elif len(set(mafft_scores)) == 2 and mafft_scores[2] < mafft_scores[0] and mafft_scores[2] < mafft_scores[1]: # default == predicted
                #     mafft_winner = "sop"
                # elif len(set(mafft_scores)) == 2 and mafft_scores[0] == mafft_scores[2]: # default == sop
                #     mafft_winner = "default"
                # elif len(set(mafft_scores)) == 2 and mafft_scores[1] == mafft_scores[2]: # predicted == sop
                #     mafft_winner = "tie2(predicted=sop)"
                # elif len(set(mafft_scores)) == 1:
                #     mafft_winner = "tie(predicted=sop=default)"
            # else:
            #     mafft_winner = np.nan

            # if prank_scores:
            #     if len(set(prank_scores)) == 3:
            #         index_of_min = prank_scores.index(min(prank_scores))
            #         prank_winner = result_map.get(int(index_of_min), "Unknown")
            #     elif len(set(prank_scores)) == 2 and prank_scores[0] == prank_scores[1]: #default == predicted
            #         prank_winner = "default"
            #     elif len(set(prank_scores)) == 2 and prank_scores[0] == prank_scores[2]: #default == sop
            #         prank_winner = "default"
            #     elif len(set(prank_scores)) == 2 and prank_scores[1] == prank_scores[2]: #predicted == sop
            #         prank_winner = "tie2(predicted=sop)"
            #     elif len(set(prank_scores)) == 1:
            #         prank_winner = "tie(predicted=sop=default)"
            # else:
            #     prank_winner = np.nan

            # if muscle_scores:
            #     if len(set(muscle_scores)) == 3:
            #         index_of_min = muscle_scores.index(min(muscle_scores))
            #         muscle_winner = result_map.get(int(index_of_min), "Unknown")
            #     elif len(set(muscle_scores)) == 2 and muscle_scores[0] == muscle_scores[1]: #true = predicted
            #         muscle_winner = "predicted"
            #     elif len(set(muscle_scores)) == 2 and muscle_scores[0] == muscle_scores[2]: #true = sop
            #         muscle_winner = "sop"
            #     elif len(set(muscle_scores)) == 2 and muscle_scores[1] == muscle_scores[2]: # predicted == sop
            #         muscle_winner = "tie2(predicted=sop)"
            #     elif len(set(muscle_scores)) == 1:
            #         muscle_winner = "tie(predicted=sop=true)"
            # else:
            #     muscle_winner = np.nan
            #
            # if baliphy_scores:
            #     if len(set(baliphy_scores)) == 3:
            #         index_of_min = baliphy_scores.index(min(baliphy_scores))
            #         baliphy_winner = result_map.get(int(index_of_min), "Unknown")
            #     elif len(set(baliphy_scores)) == 2 and baliphy_scores[0] == baliphy_scores[1]: #true = predicted
            #         baliphy_winner = "predicted"
            #     elif len(set(baliphy_scores)) == 2 and baliphy_scores[0] == baliphy_scores[2]: #true = sop
            #         baliphy_winner = "sop"
            #     elif len(set(baliphy_scores)) == 2 and baliphy_scores[1] == baliphy_scores[2]: # predicted == sop
            #         baliphy_winner = "tie2(predicted=sop)"
            #     elif len(set(baliphy_scores)) == 1:
            #         baliphy_winner = "tie(predicted=sop=true)"
            # else:
            #     baliphy_winner = np.nan

            # Append results for the current code
            if code not in self.results:
                self.results.append({
                    'code': code,
                    'min_true_score_filename': min_true_score_code_filename,
                    'min_true_score': min_true_score_value,
                    'min_predicted_score_filename': min_predicted_filename,
                    'min_predicted_score': min_predicted_score,
                    'min_predicted_true_score': min_predicted_true_score,
                    'max_SoP_filename': max_SoP_filename,
                    'max_SoP_score': max_SoP_score,
                    'max_SoP_true_score': max_SoP_true_score,
                    'min_mafft_true_score': min_mafft_true_score,
                    'min_mafft_filename': min_mafft_code_filename,
                    'default_mafft_true_score': default_mafft_true_score,
                    'min_mafft_predicted_score_filename': min_mafft_predicted_filename,
                    'min_mafft_predicted_score': min_mafft_predicted_score,
                    'min_mafft_predicted_true_score': min_mafft_predicted_true_score,
                    'max_mafft_SoP_score_filename': max_mafft_SoP_filename,
                    'max_mafft_SoP_score': max_mafft_SoP_score,
                    'max_mafft_SoP_true_score': max_mafft_SoP_true_score,
                    'min_prank_true_score': min_prank_true_score,
                    'min_prank_filename': min_prank_code_filename,
                    'default_prank_true_score': default_prank_true_score,
                    'min_prank_predicted_score_filename': min_prank_predicted_filename,
                    'min_prank_predicted_score': min_prank_predicted_score,
                    'min_prank_predicted_true_score': min_prank_predicted_true_score,
                    'max_prank_SoP_score_filename': max_prank_SoP_filename,
                    'max_prank_SoP_score': max_prank_SoP_score,
                    'max_prank_SoP_true_score': max_prank_SoP_true_score,
                    'min_muscle_true_score': min_muscle_true_score,
                    'min_muscle_filename': min_muscle_code_filename,
                    'default_muscle_true_score': default_muscle_true_score,
                    'min_muscle_predicted_score_filename': min_muscle_predicted_filename,
                    'min_muscle_predicted_score': min_muscle_predicted_score,
                    'min_muscle_predicted_true_score': min_muscle_predicted_true_score,
                    'max_muscle_SoP_score_filename': max_muscle_SoP_filename,
                    'max_muscle_SoP_score': max_muscle_SoP_score,
                    'max_muscle_SoP_true_score': max_muscle_SoP_true_score,
                    'min_baliphy_true_score': min_baliphy_true_score,
                    'min_baliphy_filename': min_baliphy_code_filename,
                    'default_baliphy_true_score': default_baliphy_true_score,
                    'min_baliphy_predicted_score_filename': min_baliphy_predicted_filename,
                    'min_baliphy_predicted_score': min_baliphy_predicted_score,
                    'min_baliphy_predicted_true_score': min_baliphy_predicted_true_score,
                    'max_baliphy_SoP_score_filename': max_baliphy_SoP_filename,
                    'max_baliphy_SoP_score': max_baliphy_SoP_score,
                    'max_baliphy_SoP_true_score': max_baliphy_SoP_true_score,
                    'mafft_winner': mafft_winner,
                    'prank_winner': prank_winner,
                    'muscle_winner': muscle_winner,
                    'baliphy_winner': baliphy_winner,
                    'overall_winner': overall_winner
                })
            if code not in self.winners:
                self.winners.append({
                    'code': code,
                    'mafft_winner': mafft_winner,
                    'prank_winner': prank_winner,
                    'muscle_winner': muscle_winner,
                    'baliphy_winner': baliphy_winner
                    # 'overall_winner': overall_winner
                })

            if code not in self.overall_win:
                self.overall_win.append({
                    'code': code,
                    'overall_winner': overall_winner
                })

    def summarize(self):
        # Create a DataFrame from the results
        results_df = pd.DataFrame(self.results)
        results_df = results_df.drop('code', axis=1)
        winners_df = pd.DataFrame(self.winners)
        winners_df = winners_df.drop('code', axis=1)
        overall_winners_df = pd.DataFrame(self.overall_win)
        overall_winners_df = overall_winners_df.drop('code', axis=1)
        self.pickme_df = results_df
        self.winners_df = winners_df
        self.overall_winners_df = overall_winners_df
        print(len(self.pickme_df), len(self.winners_df), len(self.overall_winners_df))

    def save_to_csv(self, i: int):
        filename = f'./out/pick_me_trio_v{i}.csv'
        self.pickme_df.to_csv(filename, index=False)
        # filename = f'./out/pick_me_SoP_program_v{i}.csv'
        # self.pickme_sop_df.to_csv(filename, index=False)

    def plot_results(self, i: int):
        plotname = f'./out/pick_me_trio_plot_v{i}.png'
        df = self.winners_df
        counts = df.apply(lambda x: x.value_counts()).fillna(0)
        percentages = counts / len(df) * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages.T.plot(kind='bar', stacked=True, ax=ax)

        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('MSA Aligner')
        ax.set_title('Distribution of Winners for Each MSA Aligner')

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.),
                        ha='center', va='center', color='black', fontsize=10)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fname=plotname, format='png')
        plt.show()

    def plot_overall_results(self, i: int):
        plotname = f'./out/pick_me_trio_overall_plot_v{i}.png'
        df = self.overall_winners_df
        # counts = df.apply(lambda x: x.value_counts()).fillna(0)
        legend = ["min predicted", "max sop", "default mafft", "default prank", "default muscle", "default baliphy"]  # Add 'D' to ensure it gets counted as 0 if missing
        counts = df.apply(lambda x: x.value_counts()).fillna(0)
        counts = counts.reindex(legend, fill_value=0)

        percentages = counts / len(df) * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages.T.plot(kind='bar', stacked=True, ax=ax)

        ax.set_ylabel('Percentage (%)')
        ax.set_title('Distribution of Overall Winners')

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.),
                        ha='center', va='center', color='black', fontsize=10)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fname=plotname, format='png')
        plt.show()


class PickMeAggregator:
    def __init__(self) -> None:
        self.results = []
        self.winners = []
        self.overall_win = []

    def add(self, pick_me_run: PickMeGameTrio):
        for result in pick_me_run.results:
            if not any(item['code'] == result['code'] for item in self.results):
                self.results.append(result)

        for result in pick_me_run.winners:
            if not any(item['code'] == result['code'] for item in self.winners):
                self.winners.append(result)

        for result in pick_me_run.overall_win:
            if not any(item['code'] == result['code'] for item in self.overall_win):
                self.overall_win.append(result)

    def save_to_csv(self):
        leng = len(self.pickme_df)
        filename = f'./out/pick_me_Results_ALL_{leng}.csv'
        self.pickme_df.to_csv(filename, index=False)
        leng = len(self.winners_df)
        filename = f'./out/pick_me_Winners_ALL_{leng}.csv'
        self.winners_df.to_csv(filename, index=False)
        leng = len(self.overall_winners_df)
        filename = f'./out/pick_me_overall_Winners_ALL_{leng}.csv'
        self.overall_winners_df.to_csv(filename, index=False)

    def summarize(self):
        # Create a DataFrame from the results
        results_df = pd.DataFrame(self.results)
        # results_df = results_df.drop('code', axis=1)
        winners_df = pd.DataFrame(self.winners)
        # winners_df = winners_df.drop('code', axis=1)
        overall_winners_df = pd.DataFrame(self.overall_win)
        # overall_winners_df = overall_winners_df.drop('code', axis=1)
        self.pickme_df = results_df
        self.winners_df = winners_df
        self.overall_winners_df = overall_winners_df
        print(len(self.pickme_df), len(self.winners_df), len(self.overall_winners_df))

    def plot_results(self):
        plotname = f'./out/pick_me_trio_plot_ALL.png'
        df = self.winners_df
        df = df.drop('code', axis=1)
        counts = df.apply(lambda x: x.value_counts()).fillna(0)
        percentages = counts / len(df) * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages.T.plot(kind='bar', stacked=True, ax=ax)

        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('MSA Aligner')
        ax.set_title('Distribution of Winners for Each MSA Aligner')

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.),
                        ha='center', va='center', color='black', fontsize=10)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fname=plotname, format='png')
        plt.show()

    def plot_overall_results(self):
        plotname = f'./out/pick_me_overall_plot_ALL.png'
        df = self.overall_winners_df
        df = df.drop('code', axis=1)
        # counts = df.apply(lambda x: x.value_counts()).fillna(0)
        legend = ["min predicted", "max sop", "default mafft", "default prank", "default muscle",
                  "default baliphy"]  # Add 'D' to ensure it gets counted as 0 if missing
        counts = df.apply(lambda x: x.value_counts()).fillna(0)
        counts = counts.reindex(legend, fill_value=0)

        percentages = counts / len(df) * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages.T.plot(kind='bar', stacked=True, ax=ax)

        ax.set_ylabel('Percentage (%)')
        ax.set_title('Distribution of Overall Winners')

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.),
                        ha='center', va='center', color='black', fontsize=10)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fname=plotname, format='png')
        plt.show()