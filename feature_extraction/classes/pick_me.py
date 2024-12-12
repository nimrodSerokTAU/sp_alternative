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
        self.pickme_sop_df = None
        self.accumulated_data = {}
        self.accumulated_sop_data = {}

        # Load the two CSV files into DataFrames
        # df = pd.read_csv(self.features_file)
        df1 = pd.read_csv(self.features_file)
        df2 = pd.read_csv(self.prediction_file)

        df1['code1'] = df1['code1'].astype(str)
        df2['code1'] = df2['code1'].astype(str)

        df = pd.merge(df1, df2, on=['code', 'code1'], how='inner')

        # merged_df = pd.merge(df1, df2, on=['code', 'code1'], how='outer', indicator=True)
        # non_matching_df2 = merged_df[merged_df['_merge'] == 'right_only']
        # print(non_matching_df2)

        if predicted_measure == "msa_distance":
            self.true_score = 'dpos_dist_from_true'
        elif predicted_measure == "tree_distance":
            self.true_score = 'rf_from_true'


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
            # default_prank_true_scores = code_df[code_df['code'] == 'MSA.PRANK.aln.best.fas'][self.true_score].values
            # default_prank_true_scores = code_df[code_df['code'] == 'MSA.PRANK.aln.With_Names'][self.true_score].values
            default_prank_true_scores = code_df[code_df['code'].isin(['MSA.PRANK.aln.With_Names', 'MSA.PRANK.aln.best.fas'])][
                self.true_score].values
            default_prank_true_score = default_prank_true_scores[0] if len(default_prank_true_scores) > 0 else np.nan

            # True_score for code_filename of the form 'MSA.MUSCLE.aln.best.{code}.fas'
            default_muscle_true_scores = \
            code_df[code_df['code'].str.contains(r'^MSA\.MUSCLE\.aln\.best\.[\w]+\.fas$', regex=True)][
                self.true_score].values
            default_muscle_true_score = default_muscle_true_scores.min() if len(default_muscle_true_scores) > 0 else np.nan

            # True_score for 'bali_phy_msa.199.fasta'
            default_baliphy_true_scores = code_df[code_df['code'] == 'bali_phy_msa.199.fasta'][self.true_score].values
            default_baliphy_true_score = default_baliphy_true_scores[0] if len(default_baliphy_true_scores) > 0 else np.nan


            # Minimum true_score and filename among MAFFT alternative MSAs
            substrings = ['muscle', 'prank', '_TRUE.fas', 'true_tree.txt', 'bali_phy']
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

            # Minimum true_score and code_filename among Bali-Phy alternative MSAs
            baliphy_df = code_df[code_df['code'].str.contains('bali_phy', case=False, na=False, regex=True)]
            if not baliphy_df.empty:
                min_baliphy_row = baliphy_df.loc[baliphy_df[self.true_score].idxmin()]
                min_baliphy_true_score = min_baliphy_row[self.true_score]
                min_baliphy_code_filename = min_baliphy_row['code']
            else:
                min_baliphy_true_score = np.nan
                min_baliphy_code_filename = np.nan

            # Code_filename, predicted_score, and true_score of the filename with the minimum predicted_score
            min_predicted_score_row = code_df.loc[code_df[self.predicted_score].idxmin()]
            min_predicted_filename = min_predicted_score_row['code']
            min_predicted_true_score = min_predicted_score_row[self.true_score]
            min_predicted_score = min_predicted_score_row[self.predicted_score]
            sorted_temp_df = code_df.sort_values(by=self.predicted_score, ascending=True)
            top_20_rows = sorted_temp_df.head(20)
            top_20_filenames = top_20_rows['code'].tolist()
            top_20_scores = top_20_rows[self.true_score].tolist()

            # Calculate boolean fields
            min_true_equals_min_predicted = (min_true_score_value == (min_predicted_true_score + error))
            # min_true_equals_min_predicted = (min_true_score_code_filename == min_predicted_filename)
            min_predicted_le_default_mafft = (min_predicted_true_score <= (default_mafft_true_score + error))
            min_predicted_le_default_prank = (min_predicted_true_score <= (default_prank_true_score + error))
            min_predicted_le_default_muscle = (min_predicted_true_score <= (default_muscle_true_score + error))
            min_predicted_le_default_baliphy = (min_predicted_true_score <= (default_baliphy_true_score + error))
            min_predicted_le_min_mafft = (min_predicted_true_score <= (min_mafft_true_score + error))
            min_predicted_le_min_prank = (min_predicted_true_score <= (min_prank_true_score + error))
            min_predicted_le_min_muscle = (min_predicted_true_score <= (min_muscle_true_score + error))
            min_predicted_le_min_baliphy = (min_predicted_true_score <= (min_baliphy_true_score + error))
            # min_true_in_top20_min_predicted = (min_true_score_code_filename in top_20_filenames)
            min_true_in_top20_min_predicted = (min_true_score_value in top_20_scores)

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
                'default_baliphy_true_score': default_baliphy_true_score,
                'min_mafft_true_score': min_mafft_true_score,
                'min_mafft_filename': min_mafft_code_filename,
                'min_prank_true_score': min_prank_true_score,
                'min_prank_filename': min_prank_code_filename,
                'min_muscle_true_score': min_muscle_true_score,
                'min_muscle_filename': min_muscle_code_filename,
                'min_baliphy_true_score': min_baliphy_true_score,
                'min_baliphy_filename': min_baliphy_code_filename,
                'min_true_equals_min_predicted': min_true_equals_min_predicted,
                'min_predicted_le_default_mafft': min_predicted_le_default_mafft,
                'min_predicted_le_default_prank': min_predicted_le_default_prank,
                'min_predicted_le_default_muscle': min_predicted_le_default_muscle,
                'min_predicted_le_default_baliphy': min_predicted_le_default_baliphy,
                'min_predicted_le_min_mafft': min_predicted_le_min_mafft,
                'min_predicted_le_min_prank': min_predicted_le_min_prank,
                'min_predicted_le_min_muscle': min_predicted_le_min_muscle,
                'min_predicted_le_min_baliphy': min_predicted_le_min_baliphy,
                'min_true_in_top20_min_predicted': min_true_in_top20_min_predicted
            })

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)
        self.pickme_df = results_df

        #ADDING SoP
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
            # default_prank_true_scores = code_df[code_df['code'] == 'MSA.PRANK.aln.best.fas'][self.true_score].values
            # default_prank_true_scores = code_df[code_df['code'] == 'MSA.PRANK.aln.With_Names'][self.true_score].values
            default_prank_true_scores = code_df[code_df['code'].isin(['MSA.PRANK.aln.With_Names', 'MSA.PRANK.aln.best.fas'])][
                self.true_score].values
            default_prank_true_score = default_prank_true_scores[0] if len(default_prank_true_scores) > 0 else np.nan

            # True_score for code_filename of the form 'MSA.MUSCLE.aln.best.{code}.fas'
            default_muscle_true_scores = \
                code_df[code_df['code'].str.contains(r'^MSA\.MUSCLE\.aln\.best\.[\w]+\.fas$', regex=True)][
                    self.true_score].values
            default_muscle_true_score = default_muscle_true_scores.min() if len(
                default_muscle_true_scores) > 0 else np.nan

            # True_score for 'bali_phy_msa.199.fasta'
            default_baliphy_true_scores = code_df[code_df['code'] == 'bali_phy_msa.199.fasta'][self.true_score].values
            default_baliphy_true_score = default_baliphy_true_scores[0] if len(
                default_baliphy_true_scores) > 0 else np.nan


            # Minimum true_score and filename among MAFFT alternative MSAs
            substrings = ['muscle', 'prank', '_TRUE.fas', 'true_tree.txt', 'bali_phy']
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

            # Minimum true_score and code_filename among Bali-Phy alternative MSAs
            baliphy_df = code_df[code_df['code'].str.contains('bali_phy', case=False, na=False, regex=True)]
            if not baliphy_df.empty:
                min_baliphy_row = baliphy_df.loc[baliphy_df[self.true_score].idxmin()]
                min_baliphy_true_score = min_baliphy_row[self.true_score]
                min_baliphy_code_filename = min_baliphy_row['code']
            else:
                min_baliphy_true_score = np.nan
                min_baliphy_code_filename = np.nan


            # Code_filename, predicted_score, and true_score of the filename with the minimum predicted_score
            max_SoP_score_row = code_df.loc[code_df['normalised_sop_score'].idxmax()]
            max_SoP_filename = max_SoP_score_row['code']
            max_SoP_true_score = max_SoP_score_row[self.true_score]
            max_SoP_score = max_SoP_score_row['normalised_sop_score']
            sorted_temp_df = code_df.sort_values(by='normalised_sop_score', ascending=False)
            top_20_SoP_rows = sorted_temp_df.head(20)
            top_20_SoP_filenames = top_20_SoP_rows['code'].tolist()
            top_20_SoP_scores = top_20_SoP_rows[self.true_score].tolist()

            # Calculate boolean fields
            min_true_equals_max_SoP = (min_true_score_value == (max_SoP_true_score + error))
            # min_true_equals_min_predicted = (min_true_score_code_filename == min_predicted_filename)
            min_SoP_le_default_mafft = (max_SoP_true_score <= (default_mafft_true_score + error))
            min_SoP_le_default_prank = (max_SoP_true_score <= (default_prank_true_score + error))
            min_SoP_le_default_muscle = (max_SoP_true_score <= (default_muscle_true_score + error))
            min_SoP_le_default_baliphy = (max_SoP_true_score <= (default_baliphy_true_score + error))
            min_SoP_le_min_mafft = (max_SoP_true_score <= (min_mafft_true_score + error))
            min_SoP_le_min_prank = (max_SoP_true_score <= (min_prank_true_score + error))
            min_SoP_le_min_muscle = (max_SoP_true_score <= (min_muscle_true_score + error))
            min_SoP_le_min_baliphy = (max_SoP_true_score <= (min_baliphy_true_score + error))
            # min_true_in_top20_min_predicted = (min_true_score_code_filename in top_20_filenames)
            min_true_in_top20_min_SoP = (min_true_score_value in top_20_SoP_scores)

            # Append results for the current code
            results.append({
                'code': code,
                'min_true_score_filename': min_true_score_code_filename,
                'min_true_score': min_true_score_value,
                'min_SoP_filename': max_SoP_filename,
                'min_SoP_score': max_SoP_score,
                'min_SoP_true_score': max_SoP_true_score,
                'default_mafft_true_score': default_mafft_true_score,
                'default_prank_true_score': default_prank_true_score,
                'default_muscle_true_score': default_muscle_true_score,
                'default_baliphy_true_score': default_baliphy_true_score,
                'min_mafft_true_score': min_mafft_true_score,
                'min_mafft_filename': min_mafft_code_filename,
                'min_prank_true_score': min_prank_true_score,
                'min_prank_filename': min_prank_code_filename,
                'min_muscle_true_score': min_muscle_true_score,
                'min_muscle_filename': min_muscle_code_filename,
                'min_baliphy_true_score': min_baliphy_true_score,
                'min_baliphy_filename': min_baliphy_code_filename,
                'min_true_equals_min_SoP': min_true_equals_max_SoP,
                'min_SoP_le_default_mafft': min_SoP_le_default_mafft,
                'min_SoP_le_default_prank': min_SoP_le_default_prank,
                'min_SoP_le_default_muscle': min_SoP_le_default_muscle,
                'min_SoP_le_default_baliphy': min_SoP_le_default_baliphy,
                'min_SoP_le_min_mafft': min_SoP_le_min_mafft,
                'min_SoP_le_min_prank': min_SoP_le_min_prank,
                'min_SoP_le_min_muscle': min_SoP_le_min_muscle,
                'min_SoP_le_min_baliphy': min_SoP_le_min_baliphy,
                'min_true_in_top20_min_SoP': min_true_in_top20_min_SoP
            })

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)
        self.pickme_sop_df = results_df


    def save_to_csv(self, i: int):
        filename = f'./out/pick_me_v{i}.csv'
        self.pickme_df.to_csv(filename, index=False)
        filename = f'./out/pick_me_SoP_v{i}.csv'
        self.pickme_sop_df.to_csv(filename, index=False)

    def plot_results(self, i: int):
        plotname = f'./out/pick_me_plot_v{i}.png'
        df = self.pickme_df
        n = df.shape[0]
        # Calculate the percentage of True values for each condition
        percentages = {
            'Did we find the minimum !score?': (df['min_true_equals_min_predicted'].mean() * 100),
            'Are we better or equal to the default MAFFT?': (df['min_predicted_le_default_mafft'].mean() * 100),
            'Are we better or equal to the default PRANK?': (df['min_predicted_le_default_prank'].mean() * 100),
            'Are we better or equal to default MUSCLE?': (df['min_predicted_le_default_muscle'].mean() * 100),
            'Are we better or equal to default Bali-Phy?': (df['min_predicted_le_default_baliphy'].mean() * 100),
            'Are we better than the best out of all MAFFT Alternatives?': (df['min_predicted_le_min_mafft'].mean() * 100),
            'Are we better than the best out of all PRANK Alternatives?': (df['min_predicted_le_min_prank'].mean() * 100),
            'Are we better than the best out of all MUSCLE Alternatives?': (df['min_predicted_le_min_muscle'].mean() * 100),
            'Are we better than the best out of all Bali-Phy Alternatives?': (
                        df['min_predicted_le_min_baliphy'].mean() * 100),
            'Was the real minimum !score among our (predicted) top 20 true scores?': (df['min_true_in_top20_min_predicted'].mean() * 100)
        }

        # Accumulate percentages
        for condition, percentage in percentages.items():
            if condition not in self.accumulated_data:
                self.accumulated_data[condition] = {'sum': 0, 'count': 0}

            self.accumulated_data[condition]['sum'] += percentage
            self.accumulated_data[condition]['count'] += 1

        percentages_df = pd.DataFrame(list(percentages.items()), columns=['Condition', 'Percentage'])

        # Sort by 'Percentage'
        # percentages_df = percentages_df.sort_values(by='Percentage', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(percentages_df['Condition'], percentages_df['Percentage'], color='skyblue')

        ax.set_xlabel('Percentage of YES answers (%)', fontsize=14)
        ax.set_ylabel('Question', fontsize=14)
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


    def plot_SoP_results(self, i: int):
        plotname = f'./out/pick_me_SoP_plot_v{i}.png'
        df = self.pickme_sop_df
        n = df.shape[0]
        # Calculate the percentage of True values for each condition

        percentages = {
            'Did SoP find the minimum !score?': (df['min_true_equals_min_SoP'].mean() * 100),
            'Are SoP better or equal to the default MAFFT?': (df['min_SoP_le_default_mafft'].mean() * 100),
            'Are SoP better or equal to the default PRANK?': (df['min_SoP_le_default_prank'].mean() * 100),
            'Are SoP better or equal to default MUSCLE?': (df['min_SoP_le_default_muscle'].mean() * 100),
            'Are SoP better or equal to default Bali-Phy?': (df['min_SoP_le_default_baliphy'].mean() * 100),
            'Are SoP better than the best out of all MAFFT Alternatives?': (df['min_SoP_le_min_mafft'].mean() * 100),
            'Are SoP better than the best out of all PRANK Alternatives?': (df['min_SoP_le_min_prank'].mean() * 100),
            'Are SoP better than the best out of all MUSCLE Alternatives?': (df['min_SoP_le_min_muscle'].mean() * 100),
            'Are SoP better than the best out of all Bali-Phy Alternatives?': (df['min_SoP_le_min_baliphy'].mean() * 100),
            'Was the real minimum !score among SoP top 20 true scores?': (df['min_true_in_top20_min_SoP'].mean() * 100)
        }

        # Accumulate percentages
        for condition, percentage in percentages.items():
            if condition not in self.accumulated_sop_data:
                self.accumulated_sop_data[condition] = {'sum': 0, 'count': 0}

            self.accumulated_sop_data[condition]['sum'] += percentage
            self.accumulated_sop_data[condition]['count'] += 1

        percentages_df = pd.DataFrame(list(percentages.items()), columns=['Condition', 'Percentage'])

        # Sort by 'Percentage'
        # percentages_df = percentages_df.sort_values(by='Percentage', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(percentages_df['Condition'], percentages_df['Percentage'], color='skyblue')

        ax.set_xlabel('Percentage of YES answers (%)', fontsize=14)
        ax.set_ylabel('Question', fontsize=14)
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

    def plot_average_results(self, k):
        plotname = f'./out/pick_me_average_plot.png'
        # Create a DataFrame to store average percentages
        average_percentages = {
            condition: data['sum'] / data['count']
            for condition, data in self.accumulated_data.items()
        }

        df_size = self.pickme_df
        n = df_size.shape[0]
        # Convert the dictionary to a DataFrame
        average_percentages_df = pd.DataFrame(list(average_percentages.items()),
                                              columns=['Condition', 'Average_Percentage'])

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(average_percentages_df['Condition'], average_percentages_df['Average_Percentage'], color='skyblue')

        ax.set_xlabel('Percentage of YES answers (%)', fontsize=14)
        ax.set_ylabel('Question', fontsize=14)

        import textwrap

        subtitle = f'Average Percentage of True Value Answers for Each Pick-Me Question over {k} prediction runs, n={n}, error = {self.error}'
        wrapped_subtitle = "\n".join(textwrap.wrap(subtitle, width=80))  # Adjust width as needed

        fig.suptitle(wrapped_subtitle, fontsize=16)

        # fig.suptitle(f'Average Percentage of True Value Answers for Each Pick-Me Question over {k} prediction runs, n={n}, error = {self.error}',
        #              fontsize=16)

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


        # average for sop score-based pick me
        plotname = f'./out/pick_me_average_SoP_plot.png'
        # Create a DataFrame to store average percentages
        average_percentages = {
            condition: data['sum'] / data['count']
            for condition, data in self.accumulated_sop_data.items()
        }

        df_size = self.pickme_df
        n = df_size.shape[0]
        # Convert the dictionary to a DataFrame
        average_percentages_df = pd.DataFrame(list(average_percentages.items()),
                                              columns=['Condition', 'Average_Percentage'])

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(average_percentages_df['Condition'], average_percentages_df['Average_Percentage'], color='skyblue')

        ax.set_xlabel('Percentage of YES answers (%)', fontsize=14)
        ax.set_ylabel('Question', fontsize=14)


        import textwrap

        subtitle = f'Average Percentage of True Value Answers for Each Pick-Me Question over {k} prediction runs, n={n}, error = {self.error}'
        wrapped_subtitle = "\n".join(textwrap.wrap(subtitle, width=80))  # Adjust width as needed

        fig.suptitle(wrapped_subtitle, fontsize=16)

        # fig.suptitle(
        #     f'Average Percentage of True Value Answers for Each Pick-Me Question over {k} prediction runs, n={n}, error = {self.error}',
        #     fontsize=16)

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