import inspect
import logging
import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from typing import Literal
from scipy.stats import pearsonr
import visualkeras
import joblib
from scipy.stats import pearsonr, gaussian_kde, norm, rankdata

import pydot
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation, BatchNormalization, Input, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# from classes.regressor import Regressor
import shap

from dl_model.data_processing.group_aware_zscore_scaler import GroupAwareScalerZ


# def _assign_aligner(row: pd.Series) -> str:
#     code = row['code'].lower()
#     not_mafft = ['muscle', 'prank', '_true.fas', 'true_tree.txt', 'bali_phy', 'baliphy', 'original']
#
#     if not any(sub in code for sub in not_mafft):
#         return 'mafft'
#     if 'muscle' in code:
#         return 'muscle'
#     elif 'prank' in code:
#         return 'prank'
#     elif 'bali_phy' in code or 'baliphy' in code:
#         return 'baliphy'
#
#     return 'true'
def   _assign_aligner(row: pd.Series) -> str:
    code = row['code'].lower()
    not_mafft = ['muscle', 'prank', '_true.fas', 'true_tree.txt', 'bali_phy', 'baliphy', 'original']

    if row['code'] == row['code1']: #TODO - added, check that works properly
        return 'true'
    elif not any(sub in code for sub in not_mafft):
        return 'mafft'
    elif 'muscle' in code:
        return 'muscle'
    elif 'prank' in code:
        return 'prank'
    elif 'bali_phy' in code or 'baliphy' in code:
        return 'baliphy'

    return 'true'

def _check_file_type(file_path: str) -> str:
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.parquet':
        return 'parquet'
    elif file_extension == '.csv':
        return 'csv'
    else:
        return 'unknown'

# def _rank_percentile_scale_targets(y_true: pd.Series, group_codes: pd.Series) -> pd.Series:
#     if not y_true.index.equals(group_codes.index):
#         raise ValueError("y_true and group_codes must have the same index")
#
#     df = pd.DataFrame({
#         "y": y_true,
#         "group": group_codes
#     })
#
#     scaled_series = pd.Series(index=y_true.index, dtype=np.float32)
#
#     for group_val, group_df in df.groupby("group"):
#         vals = group_df["y"].values
#         if len(vals) == 1:
#             scaled = np.array([0.0])
#         else:
#             ranks = rankdata(vals, method="average")
#             scaled = (ranks - 1) / (len(vals) - 1)
#         scaled_series.loc[group_df.index] = scaled.astype('float32')
#
#     return scaled_series

def _rank_percentile_scale_targets(y_true: pd.Series, group_codes: pd.Series) -> pd.Series:
    if not y_true.index.equals(group_codes.index):
        raise ValueError("y_true and group_codes must have the same index")

    df = pd.DataFrame({
        "y": y_true,
        "group": group_codes
    })

    scaled_series = pd.Series(index=y_true.index, dtype=np.float32)

    for group_val, group_df in df.groupby("group"):
        vals = group_df["y"].values
        if len(vals) == 1:
            scaled = np.array([0.0])
        else:
            ranks = rankdata(vals, method="average")
            scaled = (ranks - 1) / (len(vals) - 1)
        scaled_series.loc[group_df.index] = scaled.astype('float32')

    return scaled_series

def _zscore_targets_per_group(y_true: pd.Series, group_codes: pd.Series) -> pd.Series:
    """Per-group z-score scaling that preserves index and returns a Series."""
    if not y_true.index.equals(group_codes.index):
        raise ValueError("y_true and group_codes must have the same index")

    df = pd.DataFrame({"y": y_true, "group": group_codes})
    scaled_series = pd.Series(index=y_true.index, dtype=np.float32)

    for group_val, group_df in df.groupby("group"):
        vals = group_df["y"].values
        mean, std = np.mean(vals), np.std(vals)

        if std == 0:
            scaled_vals = np.zeros_like(vals, dtype=np.float32)
        else:
            scaled_vals = ((vals - mean) / std).astype(np.float32)

        # assign scaled values to same indices
        scaled_series.loc[group_df.index] = scaled_vals

    return scaled_series

def _check_missing_values(df: pd.DataFrame, verbose) -> pd.DataFrame:
    if verbose == 1:
        print("Missing values in each column:\n", df.isnull().sum())
    df = df.dropna()

    return df

# def _print_correlations(df: pd.DataFrame, true_score_name: str) -> None:
#     corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df[true_score_name])
#     print(f"Pearson Correlation of Normalized SOP and {true_score_name}: {corr_coefficient1:.4f}\n",
#           f"P-value of non-correlation: {p_value1:.6f}\n")
#     corr_coefficient1, p_value1 = pearsonr(df['sop_score'], df[true_score_name])
#     print(f"Pearson Correlation of SOP and {true_score_name}: {corr_coefficient1:.4f}\n",
#           f"P-value of non-correlation: {p_value1:.6f}\n")

def _print_correlations(df: pd.DataFrame, true_score_name: str) -> None:
    # corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score_Blosum50'], df[true_score_name])
    # print(f"Pearson Correlation of Normalized SOP Blosum50 and {true_score_name}: {corr_coefficient1:.4f}\n",
    #       f"P-value of non-correlation: {p_value1:.6f}\n")
    corr_coefficient1, p_value1 = pearsonr(df['sp_norm_BLOSUM62_GO_-10_GE_-0.5'], df[true_score_name])
    print(f"Pearson Correlation of Normalized SOP Blosum62 and {true_score_name}: {corr_coefficient1:.4f}\n",
          f"P-value of non-correlation: {p_value1:.6f}\n")
    # corr_coefficient1, p_value1 = pearsonr(df['sop_score_Blosum50'], df[true_score_name])
    # print(f"Pearson Correlation of SOP Blosum 50 and {true_score_name}: {corr_coefficient1:.4f}\n",
    #       f"P-value of non-correlation: {p_value1:.6f}\n")
    corr_coefficient1, p_value1 = pearsonr(df['sp_BLOSUM62_GO_-10_GE_-0.5'], df[true_score_name])
    print(f"Pearson Correlation of SOP Blosum 62 and {true_score_name}: {corr_coefficient1:.4f}\n",
          f"P-value of non-correlation: {p_value1:.6f}\n")

# def _assign_true_score_name(predicted_measure: str) -> str:
#     if predicted_measure == 'msa_distance':
#         # true_score_name = "dpos_dist_from_true"
#         true_score_name = "dpos_ng_dist_from_true"
#     elif predicted_measure == 'tree_distance':
#         true_score_name = 'normalized_rf'
#     elif predicted_measure == 'class_label':
#         true_score_name = 'class_label'
#     return true_score_name

def _assign_true_score_name(predicted_measure: str) -> str:
    if predicted_measure == 'msa_distance':
        # true_score_name = "dpos_from_true"
        true_score_name = "dseq_from_true"
        # true_score_name = "ssp_from_true"
    elif predicted_measure == 'tree_distance':
        true_score_name = 'normalized_rf'
    elif predicted_measure == 'class_label':
        true_score_name = 'class_label'
    return true_score_name

def _read_features_into_df(features_file: str) -> pd.DataFrame:
    df = pd.DataFrame()
    file_type = _check_file_type(features_file)
    if file_type == 'parquet':
        df = pd.read_parquet(features_file, engine='pyarrow')
    elif file_type == 'csv':
        df = pd.read_csv(features_file)
    else:
        print(f"features file is of unknown format\n")
    df['code1'] = df['code1'].astype(str)

    return df
class RegPretrained:
    def __init__(self, features_file: str, mode: int = 1, remove_correlated_features: bool = False,
                 predicted_measure: Literal['msa_distance', 'class_label'] = 'msa_distance', i: int = 0,
                 verbose: int = 1, empirical: bool = False,
                 scaler_type: Literal['pretrained_rank', 'pretrained_standard', "pretrained_zscore"] = 'pretrained_standard',
                 scaler_path: str = None, pretrained_model: str = None) -> None:
        self.empirical: bool = empirical
        self.verbose: int = verbose
        self.features_file: str = features_file
        self.predicted_measure: Literal['msa_distance', 'class_label'] = predicted_measure
        self.mode: int = mode
        # self.num_estimators = n_estimators
        self.X_test, self.y_test = None, None
        self.X, self.y, self.y_pred = None, None, None
        self.prediction = None
        self.main_codes_test = None
        self.file_codes_test = None
        self.final_features_names = None
        self.remove_correlated_features: bool = remove_correlated_features
        self.scaler_type: Literal['pretrained_rank', 'pretrained_standard', "pretrained_zscore"] = scaler_type
        self.scaler_path: str = scaler_path
        self.pretrained_model: str = pretrained_model

        df = _read_features_into_df(self.features_file)
        print(len(df))
        self.true_score_name = _assign_true_score_name(self.predicted_measure)
        df = df.drop_duplicates(subset=[col for col in df.columns if col != 'code'])  # TODO - comment
        print(len(df))

        df['aligner'] = df.apply(_assign_aligner, axis=1)
        df = df[df['aligner'] != 'muscle']  # TODO - removed all True MSAs
        df = df[df['taxa_num'] > 3]
        df.to_csv('/Users/kpolonsky/Documents/sp_alternative/dl_model/out/features_w_aligner.csv',
                  index=False)

        df = _check_missing_values(df, self.verbose)
        if self.verbose == 1:
            _print_correlations(df, self.true_score_name)

        self._prepare_test_set(df)

        self._finalize_features(df)
        self._scale(i=i)

        if self.verbose == 1:
            _print_correlations(self.test_df, self.true_score_name)

    def _scale(self, i: int = 0) -> None:
        if self.scaler_type == 'pretrained_rank':
            # self.scaler = GroupAwareScaler(global_scaler=RobustScaler())
            self.scaler = GroupAwareScalerZ(mode='rank', global_scaler=RobustScaler())
            self.scaler.load(self.scaler_path)

            self.X_test_scaled = self.scaler.transform(self.test_df)
            self.final_features_names = self.scaler.get_feature_names_out()

            """ SCALED y-labels """
            self.y_test_scaled = _rank_percentile_scale_targets(y_true=self.y_test,
                                                                group_codes=self.main_codes_test)
            self.y_test = self.y_test_scaled
            """ SCALED y-labels """

        if self.scaler_type == 'pretrained_zscore':
            # self.scaler = GroupAwareScaler()
            self.scaler = GroupAwareScalerZ(mode='zscore', global_scaler=RobustScaler())
            self.scaler.load(self.scaler_path)

            self.X_test_scaled = self.scaler.transform(self.test_df)
            self.final_features_names = self.scaler.get_feature_names_out()

            """ SCALED y-labels """
            self.y_test_scaled = _zscore_targets_per_group(y_true=self.y_test, group_codes=self.main_codes_test)
            self.y_test = self.y_test_scaled
            """ SCALED y-labels """

        elif self.scaler_type == 'pretrained_standard':
            scaler = joblib.load(self.scaler_path)
            self.X_test_scaled = scaler.transform(self.X_test)
            self.X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
            # in case of standard we don't touch y-labels

            # self.y_test_scaled = _zscore_targets_per_group(self.y_test, self.main_codes_test)
            #
            # self.y_test = self.y_test_scaled

        # Check the size of each set
        if self.verbose == 1:
            print(f"Test set size: {self.test_df.shape}")

        self.X_test_scaled = self.X_test_scaled.astype('float64')
        if self.true_score_name != 'class_label':
            self.y_test = self.y_test.astype('float64')
        elif self.true_score_name == 'class_label':
            self.y_test = self.y_test.astype('int')

        if self.verbose == 1:
            print(f"Test set size  (final): {self.X_test_scaled.shape}")

    def _prepare_test_set(self, df: pd.DataFrame) -> None:
        self.unique_code1 = df['code1'].unique()
        self.test_code1 = self.unique_code1

        if self.verbose == 1:
            print(f"the testing set is: {self.test_code1} \n")

        # Create training and test DataFrames by filtering based on 'code1'
        self.test_df = df[df['code1'].isin(self.test_code1)]

        self.main_codes_test = self.test_df['code1']
        self.file_codes_test = self.test_df['code']
        self.aligners_test = self.test_df['aligner']

    def _finalize_features(self, df):
        columns_to_drop_dft = ['ssp_from_true', 'dseq_from_true', 'dpos_from_true', 'code', 'code1',
                               'aligner']

        if self.empirical == True:
            columns_to_choose = ['MEAN_RES_PAIR_SCORE', 'MEAN_COL_SCORE', 'sp_ge_norm', 'bl_75_pct', 'msa_length',
                                 'k_mer_average_K5', 'bl_25_pct', 'sp_mismatch_count_norm', 'sp_match_count_norm',
                                 'av_gap_segment_length', 'bl_max', 'entropy_mean', 'constant_sites_pct', 'seq_min_len',
                                 'avg_unique_gap_length', 'k_mer_90_pct_K5', 'k_mer_95_pct_K5', 'bl_min', 'num_cols_no_gaps',
                                 'parsimony_max', 'sp_go_norm', 'num_cols_2_gaps', 'gaps_all_except_1_len2', 'taxa_num',
                                 'parsimony_min', 'k_mer_95_pct_K10', 'entropy_max', 'num_cols_1_gap',
                                 'num_cols_all_gaps_except1', 'k_mer_90_pct_K20', 'sp_HENIKOFF_with_gaps_PAM250_GO_-6_GE_-1',
                                 'gaps_len_four_plus', 'gaps_2seq_len4plus', 'sp_mismatch_norm_BLOSUM62', 'gaps_1seq_len1',
                                 'double_char_count', 'k_mer_95_pct_K20', 'bl_sum', 'n_unique_sites',
                                 'sp_HENIKOFF_with_gaps_BLOSUM62_GO_-10_GE_-0.5', 'sp_norm_BLOSUM62_GO_-10_GE_-0.5',
                                 'seq_max_len', 'parsimony_25_pct', 'num_unique_gaps_norm', 'bl_mean',
                                 'sp_BLOSUM62_GO_-10_GE_-0.5', 'sp_mismatch_PAM250', 'sp_mismatch_norm_PAM250',
                                 'gaps_all_except_1_len4plus']

        else:
            columns_to_choose = ['sp_mismatch_norm_PAM250','sp_mismatch_norm_BLOSUM62', 'sp_norm_PAM250_GO_-6_GE_-0.2', 'sp_PAM250_GO_-6_GE_-0.2',
                                 'constant_sites_pct', 'sp_match_count', 'sp_match_count_norm', 'sp_mismatch_count', 'sp_mismatch_count_norm',
                                 'msa_length', 'taxa_num',
                                 'entropy_sum', 'entropy_mean', 'entropy_75_pct',
                                 'sp_go_norm', 'sp_ge_norm', 'parsimony_sum', 'parsimony_mean', 'parsimony_25_pct','parsimony_75_pct', 'parsimony_max',
                                 'k_mer_average_K5', 'k_mer_90_pct_K5', 'single_char_count',
                                 'num_cols_no_gaps', 'num_cols_2_gaps', 'gaps_len_four_plus', 'gaps_all_except_1_len4plus', 'gaps_2seq_len1',
                                 'num_cols_all_gaps_except1', 'av_gap_segment_length', 'num_unique_gaps',
                                 'bl_mean', 'bl_max', 'bl_25_pct', 'n_unique_sites'
                                 ]


        self.y = df[self.true_score_name]

        if self.mode == 1:
            self.X = df.drop(
                columns=columns_to_drop_dft)
            self.X_test = self.test_df.drop(
                columns=columns_to_drop_dft)

        if self.mode == 3:
            self.X = df[columns_to_choose]
            self.X_test = self.test_df[
                columns_to_choose]

        if self.remove_correlated_features:
            correlation_matrix = self.X.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if
                       any(upper_triangle[column] > 0.9)]
            if self.verbose == 1:
                print("Correlated features to drop:", to_drop)
            self.X = self.X.drop(columns=to_drop)
            self.X_test = self.X_test.drop(columns=to_drop)

        # Set train and test Labels
        self.y_test = self.test_df[self.true_score_name]

    def _load_model(self) -> float:
        self.model = load_model(self.pretrained_model, safe_mode=False)
        # loss = self.model.evaluate(self.X_test_scaled, self.y_test)
        # if self.verbose == 1:
        #     print(f"Test Loss: {loss}")

        self.y_pred = self.model.predict(self.X_test_scaled)
        self.y_pred = np.ravel(self.y_pred)  # flatten multi-dimensional array into one-dimensional
        self.y_pred = self.y_pred.astype('float64')

        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred
        })

        df_res.to_csv(f'./out/prediction_DL_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        mse = mean_squared_error(self.y_test, self.y_pred)
        if self.verbose == 1:
            print(f"Mean Squared Error: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")
        return mse

    def _features_importance(self) -> None:
        try:
            # # explain features importance
            if self.final_features_names is not None: #TODO testing this option
                X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.final_features_names)
            else:
                X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
            X_test_subset = X_test_scaled_with_names.sample(n=500, random_state=42)  # Take a sample of 500 rows
            explainer = shap.Explainer(self.model, X_test_subset)
            shap_values = explainer(X_test_subset)
            # explainer = shap.Explainer(model, X_test_scaled_with_names)
            # shap_values = explainer(X_test_scaled_with_names)
            joblib.dump(explainer,
                        f'/dl_model/out/explainer_{i}_mode{self.mode}_{self.predicted_measure}.pkl')
            joblib.dump(shap_values,
                        f'/dl_model/out/shap_values__{i}_mode{self.mode}_{self.predicted_measure}.pkl')
            matplotlib.use('Agg')

            feature_names = [
                a + ": " + str(b) for a, b in zip(X_test_subset.columns, np.abs(shap_values.values).mean(0).round(3))
            ]

            shap.summary_plot(shap_values, X_test_subset, max_display=40, feature_names=feature_names)
            # shap.summary_plot(shap_values, X_test_scaled_with_names, max_display=30, feature_names=feature_names)
            plt.savefig(f'/Users/kpolonsky/Documents/sp_alternative/dl_model/out/summary_plot_{i}.png', dpi=300,
                        bbox_inches='tight')
            # plt.show()
            plt.close()

            shap.plots.waterfall(shap_values[0], max_display=40)
            plt.savefig(f'/Users/kpolonsky/Documents/sp_alternative/dl_model/out/waterfall_plot_{i}.png', dpi=300,
                        bbox_inches='tight')
            # plt.show()
            plt.close()

            shap.force_plot(shap_values[0], X_test_subset[0], matplotlib=True, show=False)
            plt.savefig(f'/Users/kpolonsky/Documents/sp_alternative/dl_model/out/force_plot_{i}.png')
            # plt.show()
            plt.close()

            shap.plots.bar(shap_values, max_display=40)
            plt.savefig(f'/Users/kpolonsky/Documents/sp_alternative/dl_model/out/bar_plot_{i}.png', dpi=300,
                        bbox_inches='tight')
            # plt.show()
            plt.close()
        except Exception as e:
            print(f"Did not manage to get features importance\n")

    def plot_results(self, model_name: Literal["svr", "rf", "knn-r", "gbr", "dl"], mse: float, i: int) -> None:
        plt.figure(figsize=(12, 8))
        plt.scatter(self.y_test, self.y_pred, color='blue', edgecolor='k', alpha=0.7)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', linestyle='--')
        corr_coefficient, _ = pearsonr(self.y_test, self.y_pred)
        plt.text(
            0.05, 0.95,
            f'Pearson Correlation: {corr_coefficient:.2f}, MSE: {mse:.6f}',
            transform=plt.gca().transAxes,
            fontsize=18,
            verticalalignment='top'
        )
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        if model_name == "svr":
            title = "Support Vector Regression"
        elif model_name == "rf":
            title = "Random Forest Regression"
        elif model_name == "knn-r":
            title = "K-Nearest Neighbors Regression"
        elif model_name == "gbr":
            title = "Gradient Boosting Regression"
        elif model_name == "dl":
            title = "Deep learning"
        plt.title(f'{title}: Predicted vs. True Values')
        plt.grid(True)
        plt.savefig(fname=f'./out/regression_results_{i}_mode{self.mode}_{self.predicted_measure}.png', format='png')
        plt.show()
        plt.close()

        kde = gaussian_kde([self.y_pred, self.y_test], bw_method=0.1)
        density = kde([self.y_test, self.y_pred])
        plt.figure(figsize=(8, 6))
        r, _ = pearsonr(self.y_pred, self.y_test)
        plt.text(0.65, 0.95, f'Pearson r = {r:.3f}',
                 ha='right', va='top',
                 transform=plt.gca().transAxes,
                 fontsize=16, color='black', weight='bold', zorder=2)

        plt.ylim(bottom=min(self.y_test), top=max(self.y_test) * 1.1)
        scatter = plt.scatter(self.y_pred, self.y_test, c=density, cmap='plasma', edgecolors='none',
                              alpha=0.7)
        cbar = plt.colorbar(scatter, label='Density')
        cbar.set_label('Density', fontsize=18, weight='bold', labelpad=10)

        plt.xlabel('Predicted distance', fontsize=16, weight='bold', labelpad=10)
        plt.ylabel('dpos distance ("true distance")', fontsize=16, weight='bold', labelpad=10)
        plt.tight_layout()
        plt.savefig(fname=f'./out/regression_results_{i}_mode{self.mode}_{self.predicted_measure}2.png', format='png')
        plt.show()
        plt.close()


def log_function_run(func, *args, **kwargs):
    function_name = func.__name__
    parameters = ', '.join([f'{key}={value}' for key, value in kwargs.items()])
    parameters += ', '.join([str(arg) for arg in args])

    source_code = inspect.getsource(func)

    logging.info(f'Running function: {function_name} with parameters: {parameters}')
    logging.info(f'Source code of {function_name}:\n{source_code}')
    return func(*args, **kwargs)

if __name__ == '__main__':
    mse_values = []
    n = 1
    for i in range(n):
        # regressor = log_function_run(RegPretrained, features_file="./out/orthomam_monophyly_features2.csv",
        #                              mode=1,
        #                              predicted_measure='msa_distance', i=i, remove_correlated_features=False,
        #                              empirical=False, scaler_type="pretrained_rank", scaler_path = './out/OrthoMaM12/DISTANT_SET_INDELible/distant_model2/scaler_0_mode1_dseq_from_true.pkl',
        #                              pretrained_model='./out/OrthoMaM12/DISTANT_SET_INDELible/distant_model2/regressor_model_0_mode1_dseq_from_true.keras')

        regressor = log_function_run(RegPretrained, features_file="../out/balibase_features_with_foldmason_231025.csv",
                                     mode=1,
                                     predicted_measure='msa_distance', i=i, remove_correlated_features=False,
                                     empirical=False, scaler_type="pretrained_standard",
                                     scaler_path='../out/OrthoMaM12/DISTANT_SET_INDELible/semi-deduped/Final_Linear_output/model1_optimized/scaler_0_mode1_dseq_from_true.pkl',
                                     pretrained_model='./out/OrthoMaM12/DISTANT_SET_INDELible/semi-deduped/Final_Linear_output/model1_optimized/regressor_model_0_mode1_dseq_from_true.keras')

        # regressor = log_function_run(RegPretrained, features_file="./out/ortho_monophyly_v2_features_241025.csv",
        #                              mode=1,
        #                              predicted_measure='msa_distance', i=i, remove_correlated_features=False,
        #                              empirical=True, scaler_type="pretrained_zscore",
        #                              scaler_path='./out/hybrid_ranknet/scaler_0_mode1_dseq_from_true.pkl',
        #                              pretrained_model='./out/hybrid_ranknet/regressor_model_0_mode1_dseq_from_true.keras')

        mse = regressor._load_model()
        regressor._features_importance()
        mse_values.append(mse)
        regressor.plot_results("dl", mse, i)
