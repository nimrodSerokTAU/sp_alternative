import numpy as np
import pandas as pd
from typing import Literal

from matplotlib import pyplot as plt


class PickMeGame:
    def __init__(self, features_file: str, prediction_file: str, predicted_measure: Literal['msa_distance', 'tree_distance'] = 'msa_distance', error: float = 0.0) -> None:
        self.features_file = features_file
        self.prediction_file = prediction_file
        self.error = error
        self.true_score = ''
        self.predicted_score = 'predicted_score'
        self.pickme_df = None

        # Load the two CSV files into DataFrames
        # df = pd.read_csv(self.features_file)
        df1 = pd.read_csv(self.features_file)
        df2 = pd.read_csv(self.prediction_file)
        df = pd.merge(df1, df2, on=['code', 'code1'], how='inner')

        if predicted_measure == "msa_distance":
            self.true_score = 'dpos_dist_from_true'
        elif predicted_measure == "tree_distance":
            self.true_score = ''


        results = []
        for code in df['code1'].unique():
            code_df = df[df['code1'] == code]

            # Minimum true_score code_filename
            min_true_score_row = code_df.loc[code_df[self.true_score].idxmin()]
            min_true_score_code_filename = min_true_score_row['code']
            min_true_score_value = min_true_score_row[self.true_score]

            # True_score for 'MSA.MAFFT.aln.With_Names'
            default_mafft_true_scores = code_df[code_df['code'] == 'MSA.MAFFT.aln.With_Names'][self.true_score].values
            default_mafft_true_score = default_mafft_true_scores[0] if len(default_mafft_true_scores) > 0 else np.nan

            # True_score for 'MSA.PRANK.aln.best.fas'
            default_prank_true_scores = code_df[code_df['code'] == 'MSA.PRANK.aln.best.fas'][self.true_score].values
            default_prank_true_score = default_prank_true_scores[0] if len(default_prank_true_scores) > 0 else np.nan

            # True_score for code_filename of the form 'MSA.MUSCLE.aln.best.{code}.fas'
            default_muscle_true_scores = \
            code_df[code_df['code'].str.contains(r'^MSA\.MUSCLE\.aln\.best\.[\w]+\.fas$', regex=True)][
                self.true_score].values
            default_muscle_true_score = default_muscle_true_scores.min() if len(default_muscle_true_scores) > 0 else np.nan

            # Minimum true_score and filename among MAFFT alternative MSAs
            substrings = ['muscle', 'prank', '_TRUE.fas', 'true_tree.txt']
            mask = code_df['code'].str.contains('|'.join(substrings), case=False, na=False)
            mafft_df = code_df[~mask]
            if not mafft_df.empty:
                min_mafft_row = mafft_df.loc[mafft_df[self.true_score].idxmin()]
                min_mafft_true_score = min_mafft_row[self.true_score]
                min_mafft_code_filename = min_mafft_row['code']
            else:
                min_mafft_true_score = np.nan
                min_mafft_code_filename = np.nan

            # Minimum true_score and code_filename among PRANK alternative MSAs
            prank_df = code_df[code_df['code'].str.contains('prank', case=False, na=False, regex=True)]
            if not prank_df.empty:
                min_prank_row = prank_df.loc[prank_df[self.true_score].idxmin()]
                min_prank_true_score = min_prank_row[self.true_score]
                min_prank_code_filename = min_prank_row['code']
            else:
                min_prank_true_score = np.nan
                min_prank_code_filename = np.nan

            # Minimum true_score and code_filename among MUSCLE alternative MSAs
            muscle_df = code_df[code_df['code'].str.contains('muscle', case=False, na=False, regex=True)]
            if not muscle_df.empty:
                min_muscle_row = muscle_df.loc[muscle_df[self.true_score].idxmin()]
                min_muscle_true_score = min_muscle_row[self.true_score]
                min_muscle_code_filename = min_muscle_row['code']
            else:
                min_muscle_true_score = np.nan
                min_muscle_code_filename = np.nan

            # Code_filename, predicted_score, and true_score of the filename with the minimum predicted_score
            min_predicted_score_row = code_df.loc[code_df[self.predicted_score].idxmin()]
            min_predicted_filename = min_predicted_score_row['code']
            min_predicted_true_score = min_predicted_score_row[self.true_score]
            min_predicted_score = min_predicted_score_row[self.predicted_score]
            sorted_temp_df = code_df.sort_values(by=self.predicted_score, ascending=True)
            top_20_rows = sorted_temp_df.head(20)
            top_20_filenames = top_20_rows['code'].tolist()

            # Calculate boolean fields
            min_true_equals_min_predicted = (min_true_score_value == (min_predicted_score + error))
            min_predicted_le_default_mafft = (min_predicted_true_score <= (default_mafft_true_score + error))
            min_predicted_le_default_prank = (min_predicted_true_score <= (default_prank_true_score + error))
            min_predicted_le_default_muscle = (min_predicted_true_score <= (default_muscle_true_score + error))
            min_predicted_le_min_mafft = (min_predicted_true_score <= (min_mafft_true_score + error))
            min_predicted_le_min_prank = (min_predicted_true_score <= (min_prank_true_score + error))
            min_predicted_le_min_muscle = (min_predicted_true_score <= (min_muscle_true_score + error))
            min_true_in_top20_min_predicted = (min_true_score_code_filename in top_20_filenames)

            # Append results for the current code
            results.append({
                'code': code,
                'min_true_score_filename': min_true_score_code_filename,
                'min_true_score': min_true_score_value,
                'min_predicted_score_filename': min_predicted_filename,
                'min_predicted_score': min_predicted_score,
                'min_predicted_true_score': min_predicted_true_score,
                'default_mafft_true_score': default_mafft_true_score,
                'default_prank_true_score': default_prank_true_score,
                'default_muscle_true_score': default_muscle_true_score,
                'min_mafft_true_score': min_mafft_true_score,
                'min_mafft_filename': min_mafft_code_filename,
                'min_prank_true_score': min_prank_true_score,
                'min_prank_filename': min_prank_code_filename,
                'min_muscle_true_score': min_muscle_true_score,
                'min_muscle_filename': min_muscle_code_filename,
                'min_true_equals_min_predicted': min_true_equals_min_predicted,
                'min_predicted_le_default_mafft': min_predicted_le_default_mafft,
                'min_predicted_le_default_prank': min_predicted_le_default_prank,
                'min_predicted_le_default_muscle': min_predicted_le_default_muscle,
                'min_predicted_le_min_mafft': min_predicted_le_min_mafft,
                'min_predicted_le_min_prank': min_predicted_le_min_prank,
                'min_predicted_le_min_muscle': min_predicted_le_min_muscle,
                'min_true_in_top20_min_predicted': min_true_in_top20_min_predicted
            })

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)
        self.pickme_df = results_df

    def save_to_csv(self, i: int):
        filename = f'./out/pick_me_v{i}.csv'
        self.pickme_df.to_csv(filename, index=False)

    def plot_results(self, i: int):
        plotname = f'./out/pick_me_plot_v{i}.png'
        df = self.pickme_df
        n = df.shape[0]
        # Calculate the percentage of True values for each condition
        percentages = {
            'Did we find the minimum?': (df['min_true_equals_min_predicted'].mean() * 100),
            'Are we better or equal to the default MAFFT?': (df['min_predicted_le_default_mafft'].mean() * 100),
            'Are we better or equal to the default PRANK?': (df['min_predicted_le_default_prank'].mean() * 100),
            'Are we better or equal to default MUSCLE?': (df['min_predicted_le_default_muscle'].mean() * 100),
            'Are we better than the best out of all MAFFT Alternatives?': (df['min_predicted_le_min_mafft'].mean() * 100),
            'Are we better than the best out of all PRANK Alternatives?': (df['min_predicted_le_min_prank'].mean() * 100),
            'Are we better than the best out of all MUSCLE Alternatives?': (df['min_predicted_le_min_muscle'].mean() * 100),
            'Was the real minimum among our (predicted) top 20?': (df['min_true_in_top20_min_predicted'].mean() * 100)
        }

        percentages_df = pd.DataFrame(list(percentages.items()), columns=['Condition', 'Percentage'])

        # Sort by 'Percentage'
        # percentages_df = percentages_df.sort_values(by='Percentage', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(percentages_df['Condition'], percentages_df['Percentage'], color='skyblue')

        ax.set_xlabel('Percentage (%)', fontsize=14)
        ax.set_ylabel('Comparison', fontsize=14)
        fig.suptitle(f'Percentage of True Value Answers for Each Pick-Me Question, n={n}, error = {self.error}',
                     fontsize=16)

        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=12)

        # Add %
        for bar in bars:
            width = bar.get_width()
            label = f'{width:.2f}%'
            ax.text(width, bar.get_y() + bar.get_height() / 2, label,
                    va='center', ha='left', fontsize=12, color='black')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make room for the suptitle

        plt.savefig(fname=plotname, format='png')
        plt.show()
        plt.close()

