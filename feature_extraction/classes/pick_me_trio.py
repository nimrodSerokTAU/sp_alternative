import os

import numpy as np
import pandas as pd
from typing import Literal, List, Any, Iterator, Tuple, Optional, Dict
from matplotlib import pyplot as plt
from feature_extraction.classes.regressor import _read_features_into_df

class PickMeGameTrio:
    def __init__(self, features_file: str, prediction_file: str,
                 true_score_name: Literal['ssp_from_true', 'dseq_from_true', 'dpos_from_true'] = 'dpos_from_true',
                 error: float = 0.0, subset = None) -> None:
        self.features_file: str = features_file
        self.prediction_file: str = prediction_file
        self.error: float = error
        self.true_score: str = ''
        self.predicted_score: str = 'predicted_score'
        self.predicted_class: str = 'predicted_class_prob'
        # self.predicted_score = 'predicted_score2'
        # self.sum_of_pairs_score = 'normalised_sop_score'
        self.sum_of_pairs_score: str = 'sp_BLOSUM62_GO_-10_GE_-0.5'
        self.pickme_df: Optional[pd.DataFrame] = None
        self.pickme_sop_df: Optional[pd.DataFrame] = None
        self.accumulated_data: Dict[str, float] = {}
        self.accumulated_sop_data: Dict[str, float] = {}
        self.results: List[Any] = []
        self.winners: List[Any] = []
        self.overall_win: List[Any] = []
        self.subset: Optional[float] = subset
        self.true_score = true_score_name

    # def set_scores(self, df):
    #     scores = []
    #     if not df.empty:
    #         if self.subset is not None and isinstance(self.subset, int):
    #             df = df.sample(min(self.subset, len(df)))
    #         max_SoP_score_row = df.loc[df[self.sum_of_pairs_score].idxmax()]
    #         max_SoP_true_score = max_SoP_score_row[self.true_score]
    #
    #         top_5_lowest_dpos = df.nsmallest(5, self.predicted_score)
    #         min_predicted_score_row = top_5_lowest_dpos.loc[top_5_lowest_dpos[self.sum_of_pairs_score].idxmax()]
    #         # min_predicted_score_row = df.loc[df[self.predicted_score].idxmin()]
    #         min_predicted_true_score = min_predicted_score_row[self.true_score]
    #         scores.append(min_predicted_true_score)
    #         scores.append(max_SoP_true_score)
    #     else:
    #         scores.extend([np.nan, np.nan])
    #     return scores

    def set_scores(self, df):
        scores = []
        if not df.empty:
            if self.subset is not None and isinstance(self.subset, int):
                df = df.sample(min(self.subset, len(df)))
            max_SoP_score_row = df.loc[df[self.sum_of_pairs_score].idxmax()]
            max_SoP_true_score = max_SoP_score_row[self.true_score]
            # max_SoP_score_row = df.loc[df['normalised_sop_score'].idxmax()]
            # max_SoP_true_score = max_SoP_score_row[self.true_score]

            # group1 = df.nlargest(20, self.sum_of_pairs_score)
            # group1 = df.nsmallest(7, 'nj_parsimony_score')
            # group1 = df.nlargest(10, self.predicted_class)
            # group1 = df.nsmallest(15, self.predicted_score)
            # group1 = df.loc[df[self.predicted_class]>=0.4]
            # min_predicted_score_row = group1.loc[group1[self.predicted_score].idxmin()]
            # min_predicted_score_row = group1.loc[group1[self.predicted_score].idxmin()]
            # min_predicted_score_row = group1.loc[group1[self.sum_of_pairs_score].idxmax()]
            # min_predicted_score_row = group1.loc[group1['nj_parsimony_score'].idxmin()]
            # min_predicted_score_row = group1.loc[group1['number_of_gap_segments'].idxmin()]
            # min_predicted_score_row = df.loc[df[''].idxmin()]
            min_predicted_score_row = df.loc[df[self.predicted_score].idxmin()]
            # min_value = df[self.predicted_score].min()
            # min_rows = df[df[self.predicted_score] == min_value]
            # min_predicted_score_row = min_rows.loc[min_rows[self.sum_of_pairs_score].idxmax()]
            # min_predicted_score_row = min_rows.loc[min_rows['nj_parsimony_score'].idxmin()]
            min_predicted_true_score = min_predicted_score_row[self.true_score]
            scores.append(min_predicted_true_score)
            scores.append(max_SoP_true_score)
        else:
            scores.extend([np.nan, np.nan])
        return scores

    def choose_winner(self, scores):
        #{0:predicted, 1:sop, 2:default}
        # scores = np.array(scores)
        # scores[np.isnan(scores)] = np.inf

        if scores:
            # nan_count = sum(np.isnan(scores))
            #
            # if nan_count == 3:
            #     return np.nan
            #
            # if nan_count == 1:
            #     scores = np.array(scores, dtype=np.float64)
            #     scores[np.isnan(scores)] = np.inf

            scores[1] = 100  # sop is always 100

            if scores[1] < scores[0] and scores[1] <= scores[2]:  # sop < predicted and sop <= default
                winner = "SoP"
            elif scores[2] < scores[0] and scores[2] < scores[1]:  # default < predicted and default < sop
                winner = "Default"
            elif scores[0] < scores[1] and scores[0] < scores[2]:  #  predicted < sop and predicted <= default
                winner = "Predicted"
            elif scores[0] == scores[1] == scores[2]:  # default = predicted = sop
                winner = "Tie(all three)"
            elif scores[0] < scores[2] and scores[0] == scores[1]:  # predicted = sop < default
                winner = "Tie(predicted and SoP)"
            elif scores[0] < scores[1] and scores[0] == scores[2]:  #  predicted < sop and predicted <= default
                winner = "Tie(predicted and default)"
            else:
                print(f"who are you winner? {scores}\n")
                winner = np.nan
        else:
            winner = np.nan
        return winner

    def choose_overall_winner(self, scores):
        result_map = {0: "min predicted", 1: "max sop", 2: "default mafft", 3: "default prank", 4: "default muscle",
                      5: "default baliphy"}
        min_score = min(scores)
        min_indices = [i for i, score in enumerate(scores) if score == min_score]
        min_results = [result_map[i] for i in min_indices]
        if 'min predicted' in min_results and 'max sop' in min_results:
            winner = "Tie(predicted and SoP)"
        elif 'min predicted' in min_results:
            winner = "Predicted"
        elif 'max sop' in min_results:
            winner = "SoP"
        else:
            winner = "Default"
        return winner
    def run(self,i=0):

        # Load the two CSV files into DataFrames
        # df1 = pd.read_csv(self.features_file)
        df1 = _read_features_into_df(self.features_file)
        df2 = pd.read_csv(self.prediction_file)

        df1['code1'] = df1['code1'].astype(str)
        df2['code1'] = df2['code1'].astype(str)

        df = pd.merge(df1, df2, on=['code', 'code1'], how='inner')
        df = df[~df['code'].str.contains('test_original', na=False)]
        df = df[df['code'] != 'code1']
        df = df[df['taxa_num'] > 3]
        # df = df[~df['code'].str.contains('muscle', case=False, na=False, regex=True)] #TODO remove
        df.to_csv("/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/features_w_predictions.csv")
        # groups = ['BBS11','BBS12','BBS50','BBS30','BBS20', 'BBA']

        for code in df['code1'].unique():
            # if not code.startswith(groups[0]):
            #     continue
            code_df = df[df['code1'] == code]
            substrings = ['original', 'concat', '_alt_']
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
            mafft_scores = self.set_scores(mafft_df)
            mafft_scores.append(default_mafft_true_score)
            # mafft_scores.append(100) #TODO

            prank_df = code_df[code_df['code'].str.contains('prank', case=False, na=False, regex=True)]
            prank_scores = self.set_scores(prank_df)
            prank_scores.append(default_prank_true_score)
            # prank_scores.append(100) #TODO

            muscle_df = code_df[code_df['code'].str.contains('muscle', case=False, na=False, regex=True)]
            muscle_scores = self.set_scores(muscle_df)
            muscle_scores.append(default_muscle_true_score)
            # muscle_scores.append(100) #TODO

            baliphy_df = code_df[code_df['code'].str.contains('bali_phy|BALIPHY', case=False, na=False, regex=True)]
            baliphy_scores = self.set_scores(baliphy_df)
            baliphy_scores.append(default_baliphy_true_score)
            # baliphy_scores.append(100) #TODO

            overall_scores = self.set_scores(code_df)
            # overall_scores.extend([default_mafft_true_score, default_prank_true_score, default_muscle_true_score, default_baliphy_true_score])
            overall_scores.extend([np.nan, np.nan, np.nan,
                                   np.nan]) #TODO - use this line if you want to exclude defaults from overall results

            mafft_winner = self.choose_winner(mafft_scores)
            prank_winner = self.choose_winner(prank_scores)
            muscle_winner = self.choose_winner(muscle_scores)
            baliphy_winner = self.choose_winner(baliphy_scores)

            overall_winner = self.choose_overall_winner(overall_scores)

            if code not in self.winners:
                self.winners.append({
                    'code': code,
                    'MAFFT': mafft_winner,
                    'PRANK': prank_winner,
                    'Muscle': muscle_winner,
                    'BAli-Phy': baliphy_winner
                })

            if code not in self.overall_win:
                self.overall_win.append({
                    'code': code,
                    'overall_winner': overall_winner
                })


    def summarize(self):
        winners_df = pd.DataFrame(self.winners)
        self.winners_df_w_code = winners_df
        winners_df = winners_df.drop('code', axis=1)
        self.winners_df = winners_df

        overall_winners_df = pd.DataFrame(self.overall_win)
        self.overall_winners_df_w_code = overall_winners_df
        overall_winners_df = overall_winners_df.drop('code', axis=1)
        self.overall_winners_df = overall_winners_df

    def save_to_csv(self, i: int):
        filename = f'./out/pick_me_trio_v{i}.csv'
        self.winners_df_w_code.to_csv(filename, index=False)
        filename = f'./out/pick_me_trio_overall_v{i}.csv'
        self.overall_winners_df_w_code.to_csv(filename, index=False)

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

        ax.legend(title=None, loc='upper left', fontsize=12)  # <- Remove title, move to upper left

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fname=plotname, format='png')
        plt.show()

    def plot_overall_results(self, i: int):
        plotname = f'./out/pick_me_trio_overall_plot_v{i}.png'
        df = self.overall_winners_df
        # counts = df.apply(lambda x: x.value_counts()).fillna(0)
        legend = ["SoP", "Predicted", 'Tie(predicted and SoP)']
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

        ax.legend(title=None, loc='upper left', fontsize=12)  # <- Remove title, move to upper left

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