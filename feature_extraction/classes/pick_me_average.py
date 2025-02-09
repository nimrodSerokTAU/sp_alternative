import numpy as np
import pandas as pd
from typing import Literal

from matplotlib import pyplot as plt


class PickMeGameAverage:
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
        self.default_mafft_scores = []
        self.default_prank_scores = []
        self.default_muscle_scores = []
        self.default_baliphy_scores = []
        self.min_predicted_true_scores = []
        self.max_sop_true_scores = []
        self.predicted_mafft_scores = []
        self.predicted_prank_scores = []
        self.predicted_muscle_scores = []
        self.predicted_baliphy_scores = []
        self.sop_mafft_scores = []
        self.sop_prank_scores = []
        self.sop_muscle_scores = []
        self.sop_baliphy_scores = []

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



        for code in df['code1'].unique():
            # if not code.startswith(groups[5]):
            #     continue
            code_df = df[df['code1'] == code]


            # True_score for 'MSA.MAFFT.aln.With_Names'
            default_mafft_true_scores = code_df[code_df['code'] == 'MSA.MAFFT.aln.With_Names'][self.true_score].values
            default_mafft_true_score = default_mafft_true_scores[0] if len(default_mafft_true_scores) > 0 else np.nan
            self.default_mafft_scores.append(default_mafft_true_score)

            # True_score for 'MSA.PRANK.aln.best.fas'
            default_prank_true_scores = code_df[code_df['code'].isin(['MSA.PRANK.aln.With_Names', 'MSA.PRANK.aln.best.fas'])][
                self.true_score].values
            default_prank_true_score = default_prank_true_scores[0] if len(default_prank_true_scores) > 0 else np.nan
            self.default_prank_scores.append(default_prank_true_score)

            # True_score for code_filename of the form 'MSA.MUSCLE.aln.best.{code}.fas'
            default_muscle_true_scores = \
            code_df[code_df['code'].str.contains(r'^MSA\.MUSCLE\.aln\.best\.[\w]+\.fas$', regex=True)][
                self.true_score].values
            default_muscle_true_score = default_muscle_true_scores.min() if len(default_muscle_true_scores) > 0 else np.nan
            self.default_muscle_scores.append(default_muscle_true_score)

            # True_score for BALI-PHY DEFAULT
            # default_baliphy_true_scores = code_df[code_df['code'] == 'bali_phy_msa.199.fasta'][self.true_score].values
            default_baliphy_true_scores = \
                code_df[code_df['code'].str.contains(r'^MSA\.BALIPHY\.aln\.best\.[\w]+\.fas$', regex=True)][
                    self.true_score].values
            default_baliphy_true_score = default_baliphy_true_scores[0] if len(default_baliphy_true_scores) > 0 else np.nan
            self.default_baliphy_scores.append(default_baliphy_true_score)

            # Minimum true_score and filename among MAFFT alternative MSAs
            substrings = ['muscle', 'prank', '_TRUE.fas', 'true_tree.txt', 'bali_phy', 'BALIPHY', 'original']
            mask = code_df['code'].str.contains('|'.join(substrings), case=False, na=False)
            mafft_df = code_df[~mask]
            # mafft_scores = []
            if not mafft_df.empty:
                # TRUE MAFFT MIN --> TRUE BEST
                min_mafft_row = mafft_df.loc[mafft_df[self.true_score].idxmin()]
                min_mafft_true_score = min_mafft_row[self.true_score]
                # mafft_scores.append(default_mafft_true_score)
                min_mafft_code_filename = min_mafft_row['code']

                # PREDICTED MAFFT MIN --> MAFFT BEST PREDICTED
                min_mafft_predicted_score_row = mafft_df.loc[mafft_df[self.predicted_score].idxmin()]
                min_mafft_predicted_filename = min_mafft_predicted_score_row['code']
                min_mafft_predicted_true_score = min_mafft_predicted_score_row[self.true_score]
                # mafft_scores.append(min_mafft_predicted_true_score)
                self.predicted_mafft_scores.append(min_mafft_predicted_true_score)
                min_mafft_predicted_score = min_mafft_predicted_score_row[self.predicted_score]
                sorted_mafft_temp_df = mafft_df.sort_values(by=self.predicted_score, ascending=True)
                top_20_mafft_rows = sorted_mafft_temp_df.head(20)
                top_20_mafft_filenames = top_20_mafft_rows['code'].tolist()
                top_20_mafft_scores = top_20_mafft_rows[self.true_score].tolist()

                # MAX SOP MAFFT --> BEST SOP MAFFT
                max_mafft_SoP_score_row = mafft_df.loc[mafft_df[self.sum_of_pairs_score].idxmax()]
                max_mafft_SoP_filename = max_mafft_SoP_score_row['code']
                max_mafft_SoP_true_score = max_mafft_SoP_score_row[self.true_score]
                # mafft_scores.append(max_mafft_SoP_true_score)
                self.sop_mafft_scores.append(max_mafft_SoP_true_score)
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
            # prank_scores = []
            if not prank_df.empty:
                # TRUE PRANK MIN = TRUE BEST
                min_prank_row = prank_df.loc[prank_df[self.true_score].idxmin()]
                min_prank_true_score = min_prank_row[self.true_score]
                # prank_scores.append(default_prank_true_score)
                min_prank_code_filename = min_prank_row['code']

                # PREDICTED PRANK MIN = PRANK BEST PREDICTED
                min_prank_predicted_score_row = prank_df.loc[prank_df[self.predicted_score].idxmin()]
                min_prank_predicted_filename = min_prank_predicted_score_row['code']
                min_prank_predicted_true_score = min_prank_predicted_score_row[self.true_score]
                # prank_scores.append(min_prank_predicted_true_score)
                self.predicted_prank_scores.append(min_prank_predicted_true_score)
                min_prank_predicted_score = min_prank_predicted_score_row[self.predicted_score]
                sorted_prank_temp_df = prank_df.sort_values(by=self.predicted_score, ascending=True)
                top_20_prank_rows = sorted_prank_temp_df.head(20)
                top_20_prank_filenames = top_20_prank_rows['code'].tolist()
                top_20_prank_scores = top_20_prank_rows[self.true_score].tolist()

                # MAX SOP PRANK = BEST SOP PRANK
                max_prank_SoP_score_row = prank_df.loc[prank_df[self.sum_of_pairs_score].idxmax()]
                max_prank_SoP_filename = max_prank_SoP_score_row['code']
                max_prank_SoP_true_score = max_prank_SoP_score_row[self.true_score]
                # prank_scores.append(max_prank_SoP_true_score)
                self.sop_prank_scores.append(max_prank_SoP_true_score)
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
            # muscle_scores = []
            if not muscle_df.empty:
                min_muscle_row = muscle_df.loc[muscle_df[self.true_score].idxmin()]
                min_muscle_true_score = min_muscle_row[self.true_score]
                # muscle_scores.append(default_muscle_true_score)
                min_muscle_code_filename = min_muscle_row['code']

                min_muscle_predicted_score_row = muscle_df.loc[muscle_df[self.predicted_score].idxmin()]
                min_muscle_predicted_filename = min_muscle_predicted_score_row['code']
                min_muscle_predicted_true_score = min_muscle_predicted_score_row[self.true_score]
                # muscle_scores.append(min_muscle_predicted_true_score)
                self.predicted_muscle_scores.append(min_muscle_predicted_true_score)
                min_muscle_predicted_score = min_muscle_predicted_score_row[self.predicted_score]
                sorted_muscle_temp_df = muscle_df.sort_values(by=self.predicted_score, ascending=True)
                top_20_muscle_rows = sorted_muscle_temp_df.head(20)
                top_20_muscle_filenames = top_20_muscle_rows['code'].tolist()
                top_20_muscle_scores = top_20_muscle_rows[self.true_score].tolist()

                max_muscle_SoP_score_row = muscle_df.loc[muscle_df[self.sum_of_pairs_score].idxmax()]
                max_muscle_SoP_filename = max_muscle_SoP_score_row['code']
                max_muscle_SoP_true_score = max_muscle_SoP_score_row[self.true_score]
                # muscle_scores.append(max_muscle_SoP_true_score)
                self.sop_muscle_scores.append(max_muscle_SoP_true_score)
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
            # baliphy_scores = []
            if not baliphy_df.empty:
                min_baliphy_row = baliphy_df.loc[baliphy_df[self.true_score].idxmin()]
                min_baliphy_true_score = min_baliphy_row[self.true_score]
                # baliphy_scores.append(default_baliphy_true_score)
                min_baliphy_code_filename = min_baliphy_row['code']

                min_baliphy_predicted_score_row = baliphy_df.loc[baliphy_df[self.predicted_score].idxmin()]
                min_baliphy_predicted_filename = min_baliphy_predicted_score_row['code']
                min_baliphy_predicted_true_score = min_baliphy_predicted_score_row[self.true_score]
                # baliphy_scores.append(min_baliphy_predicted_true_score)
                self.predicted_baliphy_scores.append(min_baliphy_predicted_true_score)
                min_baliphy_predicted_score = min_baliphy_predicted_score_row[self.predicted_score]
                sorted_baliphy_temp_df = baliphy_df.sort_values(by=self.predicted_score, ascending=True)
                top_20_baliphy_rows = sorted_baliphy_temp_df.head(20)
                top_20_baliphy_filenames = top_20_baliphy_rows['code'].tolist()
                top_20_baliphy_scores = top_20_baliphy_rows[self.true_score].tolist()

                max_baliphy_SoP_score_row = baliphy_df.loc[baliphy_df[self.sum_of_pairs_score].idxmax()]
                max_baliphy_SoP_filename = max_baliphy_SoP_score_row['code']
                max_baliphy_SoP_true_score = max_baliphy_SoP_score_row[self.true_score]
                # baliphy_scores.append(max_baliphy_SoP_true_score)
                self.sop_baliphy_scores.append(max_baliphy_SoP_true_score)
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


            min_true_score_row = code_df.loc[code_df[self.true_score].idxmin()]
            min_true_score_code_filename = min_true_score_row['code']
            min_true_score_value = min_true_score_row[self.true_score]

            # # Code_filename, predicted_score, and true_score of the filename with the minimum predicted_score
            min_predicted_score_row = code_df.loc[code_df[self.predicted_score].idxmin()]
            min_predicted_filename = min_predicted_score_row['code']
            min_predicted_true_score = min_predicted_score_row[self.true_score]
            self.min_predicted_true_scores.append(min_predicted_true_score)
            min_predicted_score = min_predicted_score_row[self.predicted_score]
            sorted_temp_df_pred = code_df.sort_values(by=self.predicted_score, ascending=True)
            top_20_rows = sorted_temp_df_pred.head(20)
            top_20_filenames = top_20_rows['code'].tolist()
            top_20_scores = top_20_rows[self.true_score].tolist()

            # Code_filename, predicted_score, and true_score of the filename with the minimum predicted_score
            max_SoP_score_row = code_df.loc[code_df[self.sum_of_pairs_score].idxmax()]
            max_SoP_filename = max_SoP_score_row['code']
            max_SoP_true_score = max_SoP_score_row[self.true_score]
            self.max_sop_true_scores.append(max_SoP_true_score)
            max_SoP_score = max_SoP_score_row[self.sum_of_pairs_score]
            sorted_temp_df_sop = code_df.sort_values(by=self.sum_of_pairs_score, ascending=False)
            top_20_SoP_rows = sorted_temp_df_sop.head(20)
            top_20_SoP_filenames = top_20_SoP_rows['code'].tolist()
            top_20_SoP_scores = top_20_SoP_rows[self.true_score].tolist()

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
                    'max_baliphy_SoP_true_score': max_baliphy_SoP_true_score
                })


    def summarize(self):
        # Create a DataFrame from the results
        results_df = pd.DataFrame(self.results)
        results_df = results_df.drop('code', axis=1)
        self.pickme_df = results_df
        print(len(self.pickme_df))

    def save_to_csv(self, i: int):
        filename = f'./out/pick_me_avg_v{i}.csv'
        self.pickme_df.to_csv(filename, index=False)

    def plot_results(self, i: int):
        plotname = f'./out/pick_me_avg_plot_v{i}.png'
        # summary_df = pd.DataFrame

        # overall_avg_scores = [np.mean(self.max_sop_true_scores), np.mean(self.min_predicted_true_scores), np.mean(self.default_mafft_scores), np.mean(self.default_prank_scores), np.mean(self.default_muscle_scores), np.mean(self.default_baliphy_scores)]
        mafft_avg_scores = [np.nanmean(self.default_mafft_scores),np.nanmean(self.predicted_mafft_scores), np.nanmean(self.sop_mafft_scores)]
        prank_avg_scores = [np.nanmean(self.default_prank_scores), np.nanmean(self.predicted_prank_scores), np.nanmean(self.sop_prank_scores)]
        muscle_avg_scores = [np.nanmean(self.default_muscle_scores), np.nanmean(self.predicted_muscle_scores), np.nanmean(self.sop_muscle_scores)]
        baliphy_avg_scores = [np.nanmean(self.default_baliphy_scores), np.nanmean(self.predicted_baliphy_scores), np.nanmean(self.sop_baliphy_scores)]

        # categories = ['default', 'predicted', 'sop']
        methods = ['MAFFT', 'PRANK', 'MUSCLE', 'BALIPHY']


        x = np.arange(len(methods))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x - width, [mafft_avg_scores[0], prank_avg_scores[0], muscle_avg_scores[0], baliphy_avg_scores[0]],
               width, label='default', color='b')
        ax.bar(x, [mafft_avg_scores[1], prank_avg_scores[1], muscle_avg_scores[1], baliphy_avg_scores[1]], width,
               label='predicted', color='g')
        ax.bar(x + width, [mafft_avg_scores[2], prank_avg_scores[2], muscle_avg_scores[2], baliphy_avg_scores[2]],
               width, label='sop', color='r')

        ax.set_xlabel('Method')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores by Method for Different Categories')

        ax.set_xticks(x)
        ax.set_xticklabels(methods)

        ax.legend(title='Score Type')

        plt.tight_layout()
        plt.savefig(fname=plotname, format='png')
        plt.show()

    def plot_overall_results(self, i: int):
        plotname = f'./out/pick_me_avg_overall_plot_v{i}.png'
        overall_avg_scores = [np.nanmean(self.max_sop_true_scores), np.nanmean(self.min_predicted_true_scores), np.nanmean(self.default_mafft_scores), np.nanmean(self.default_prank_scores), np.nanmean(self.default_muscle_scores), np.nanmean(self.default_baliphy_scores)]
        categories = ["max sop", "min predicted", "default mafft", "default prank", "default muscle", "default baliphy"]

        plt.figure(figsize=(10, 6))
        plt.bar(categories, overall_avg_scores, color='skyblue')

        # Add labels and title
        plt.xlabel('Categories')
        plt.ylabel('Average Scores')
        plt.title('Average Scores for Different Categories')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Optionally, add a grid for better visibility of the bars
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.savefig(fname=plotname, format='png')
        plt.show()