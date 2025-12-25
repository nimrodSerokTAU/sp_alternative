import math
import os
import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, classification_report, precision_recall_curve, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal, List, Any, Iterator, Tuple, Optional
from scipy.stats import pearsonr, gaussian_kde, norm, rankdata
# import torch.nn as nn
import visualkeras
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import shap
import pickle
import pydot
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, PReLU, Activation, BatchNormalization, Input, ELU, Attention, Reshape, Embedding, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2, l1_l2, l1
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from feature_extraction.classes.group_aware_scaler import GroupAwareScaler
from feature_extraction.classes.group_aware_zscore_scaler import GroupAwareScalerZ
from feature_extraction.classes.batch_generator import BatchGenerator
from feature_extraction.classes.ranking_metrics import PerBatchRankingMetrics
import random

# def _choose_codes_with_full_data(df):
#     taxa_values = [20, 30, 40, 50, 60]
#     filtered = df[(df['Count'] >= 1600) & (df['taxa'].isin(taxa_values))]
#
#     result = (
#         filtered.groupby('taxa', group_keys=False)
#         .apply(lambda x: x.sample(n=55, random_state=42) if len(x) >= 55 else x)
#     )
#     code_list = result['Code'].tolist()
#     print(code_list)
#     return code_list

import pandas as pd

# seed = 42
# np.random.seed(seed)
# random.seed(seed)
# tf.random.set_seed(seed)

def _summarize_data(df):
    from collections import defaultdict

    summary = {}

    for taxa_num in df['taxa_num'].unique():
        taxa_df = df[df['taxa_num'] == taxa_num]

        code1_dict = defaultdict(dict)

        for code1 in taxa_df['code1'].unique():
            sub_df = taxa_df[taxa_df['code1'] == code1]

            aligner_counts = sub_df.groupby('aligner')['code'].nunique().to_dict()

            code1_dict[code1] = aligner_counts

        summary[taxa_num] = {
            'unique_code1_count': taxa_df['code1'].nunique(),
            # 'code1_details': code1_dict #TODO
        }
    print(summary, "\n")
    return summary

def _get_balanced_dpos_distribution(df):
    threshold = 0.01
    keep_fraction = 0.2
    low_group = df[df['dpos_from_true'] < threshold]
    high_group = df[df['dpos_from_true'] >= threshold]
    downsampled_low = low_group.sample(frac=keep_fraction, random_state=42)
    balanced_df = pd.concat([downsampled_low, high_group]).reset_index(drop=True)

    plt.hist(df['dpos_from_true'], bins=100, alpha=0.5, label='Original')
    plt.hist(balanced_df['dpos_from_true'], bins=100, alpha=0.5, label='Balanced')
    plt.legend()
    plt.title("Distribution of 'dpos_from_true'")
    plt.show()

    return balanced_df

def _get_balanced_frequent_codes(df, min_code1_count, frequent_codes):
    frequent_codes_set = set(frequent_codes)
    frequent_codes_balanced_per_taxa = []

    for taxa, group in df.groupby('taxa_num'):
        valid_codes = group[group['code1'].isin(frequent_codes_set)]['code1'].unique()

        if len(valid_codes) >= min_code1_count:
            sampled_codes = pd.Series(valid_codes).sample(n=min_code1_count, random_state=42).tolist() #TODO: random seed is fixed
            frequent_codes_balanced_per_taxa.extend(sampled_codes)
        else:
            print(f"Skipping taxa_num={taxa} — only {len(valid_codes)} valid codes (need {min_code1_count})")

    return frequent_codes_balanced_per_taxa

def _balanced_sample_by_code1_and_taxa(df):
    code_counts = df['code1'].value_counts()
    frequent_codes = code_counts[code_counts >= 1600].index #TODO
    # frequent_codes = code_counts[code_counts >= 1235].index #TODO
    # frequent_codes = code_counts[code_counts >= 1400].index  # TODO

    filtered_df = df[df['code1'].isin(frequent_codes)]

    code1_taxa_counts = (
        filtered_df[['code1', 'taxa_num']]
        .drop_duplicates()
        .groupby('taxa_num')
        .size()
    )

    min_code1_count = code1_taxa_counts.min()

    frequent_codes_balanced_per_taxa = _get_balanced_frequent_codes(filtered_df, min_code1_count, frequent_codes)
    selected_codes_set = set(frequent_codes_balanced_per_taxa) #TODO the set has names of the MSAs but not per code the correct way would be choosing by a pair of code and code1
    final_df = df[df['code1'].isin(selected_codes_set)]

    code_counts_final = final_df['code1'].value_counts()
    print(code_counts_final)

    num_unique = final_df['code1'].nunique()
    print(f"Number of unique codes: {num_unique}")

    return final_df

def _balanced_sample_1600(df):
    code_counts = df['code1'].value_counts()
    # frequent_codes = code_counts[code_counts >= 1235].index #TODO
    frequent_codes = code_counts[code_counts >= 1604].index #TODO change number
    # frequent_codes = code_counts[code_counts >= 1200].index  # TODO change number

    final_df = df[df['code1'].isin(frequent_codes)]
    num_unique = final_df['code1'].nunique()
    print(f"Number of unique codes: {num_unique}")

    return final_df



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

def _check_missing_values(df: pd.DataFrame, verbose) -> pd.DataFrame:
    if verbose == 1:
        print("Missing values in each column:\n", df.isnull().sum())
    df = df.dropna()

    return df

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

# def _zscore_targets_per_group(y_true, group_codes):
#     df = pd.DataFrame({"y": y_true, "group": group_codes}).reset_index(drop=True)
#     y_scaled = np.zeros(len(df))
#
#     for g, gdf in df.groupby("group"):
#         vals = gdf["y"].values
#         mean, std = np.mean(vals), np.std(vals)
#         if std == 0:
#             scaled_vals = np.zeros_like(vals)
#         else:
#             scaled_vals = (vals - mean) / std
#         # y_scaled[gdf.index.values] = scaled_vals
#
#     # return y_scaled
#     return np.asarray(y_scaled)
import numpy as np
import pandas as pd

def _zscore_targets_per_group(y_true: pd.Series, group_codes: pd.Series) -> pd.Series:
    """Per-group z-score scaling that preserves index and returns a Series."""
    if not y_true.index.equals(group_codes.index):
        raise ValueError("y_true and group_codes must have the same index")

    df = pd.DataFrame({"y": y_true, "group": group_codes})
    scaled_series = pd.Series(index=y_true.index, dtype=np.float32)

    for group_val, group_df in df.groupby("group"):
        vals = group_df["y"].values
        mean, std = np.mean(vals), np.std(vals)
        # print(f"Group: {group_val}, Mean: {mean}, Std: {std}")

        if std == 0:
            scaled_vals = np.zeros_like(vals, dtype=np.float32)
        else:
            scaled_vals = ((vals - mean) / std).astype(np.float32)

        # assign scaled values to same indices
        scaled_series.loc[group_df.index] = scaled_vals

    return scaled_series

def _zscore_targets_global(y_true: pd.Series) -> pd.Series:
    df = pd.DataFrame({"y": y_true})
    vals = df["y"].values
    mean, std = np.mean(vals), np.std(vals)

    if std == 0:
        scaled_vals = np.zeros_like(vals, dtype=np.float32)
    else:
        scaled_vals = ((vals - mean) / std).astype(np.float32)

    scaled_series = pd.Series(scaled_vals, index=y_true.index, name=y_true.name)

    return scaled_series

# def _load_or_compute_prostt5_embeddings(df):
#     msa_embeddings = {}
#     for code1 in df['code1'].unique():
#         msa_fasta = f"./msa_sequences/{code1}.fasta"
#         msa_sequences = [s for s in SeqIO.parse(msa_fasta, "fasta")]
#         msa_embeddings[code1] = aggregate_msa_embeddings([str(s.seq) for s in msa_sequences])
#     emb_df = pd.DataFrame.from_dict(msa_embeddings, orient="index")
#     emb_df.columns = [f"prostt5_{i}" for i in range(emb_df.shape[1])]
#     emb_df.reset_index().rename(columns={"index": "code1"})
#     return emb_df


class Regressor:
    '''
    features_file: file with all features and labels
    test_size: portion of the codes to be separated into a test set; all MSAs for that specific code would be on the same side of the train-test split
    mode: 1 is all features (default), 2 is all except SoP features, 3 is chosen list of features
    remove_correlated_features: if the highly correlated features should be removed (boolean, default value: False)
    scale_labels: y-labels by default are also rank-percentile scaled
    '''
    def __init__(self, features_file: str, test_size: float, mode: int = 1,
                 remove_correlated_features: bool = False,
                 i: int = 0, verbose: int = 1, empirical: bool = False,
                 scaler_type_features: Literal['standard', 'rank', 'zscore'] = 'standard',
                 scaler_type_labels: Literal['standard', 'rank', 'zscore'] = 'standard',
                 true_score_name: Literal['ssp_from_true', 'dseq_from_true', 'dpos_from_true', 'RF_phangorn_norm'] = 'dseq_from_true',
                 explain_features_importance: bool = False, with_embeddings: bool = False,
                 deduplicated: bool = False) -> None:

        self.explain_features_importance = explain_features_importance
        self.empirical = empirical
        self.verbose = verbose
        self.features_file: str = features_file
        self.test_size: float = test_size
        self.mode: int = mode
        # self.num_estimators = n_estimators
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X, self.y, self.y_pred = None, None, None
        self.prediction = None
        self.main_codes_train = None # these are the codes we can use for batch generation
        self.file_codes_train = None
        self.main_codes_test = None
        self.file_codes_test = None
        self.final_features_names = None
        self.remove_correlated_features: bool = remove_correlated_features
        self.scaler_type_features = scaler_type_features
        self.scaler_type_labels = scaler_type_labels
        self.with_embeddings = with_embeddings
        self.deduplicated = deduplicated
        if scaler_type_features not in {'standard', 'rank', 'zscore'}:
            raise ValueError(f"Invalid scaler_type_features: {scaler_type_features}, the allowed options are 'standard' , 'rank' or 'zscore'\n")
        if scaler_type_labels not in {'standard', 'rank', 'zscore'}:
            raise ValueError(f"Invalid scaler_type_labels: {scaler_type_labels}, the allowed options are 'standard' , 'rank' or 'zscore'\n")
        self.true_score_name = true_score_name
        if true_score_name not in {'ssp_from_true', 'dseq_from_true', 'dpos_from_true', 'RF_phangorn_norm'}:
            raise ValueError(f"Invalid true_score_name: {true_score_name}, the allowed options are 'ssp_from_true', 'dseq_from_true', 'dpos_from_true', 'RF_phangorn_norm'\n")

        df = _read_features_into_df(self.features_file)

        # if self.with_embeddings:  # TODO - add embeddings
        #     prostt5_embeddings = _load_or_compute_prostt5_embeddings(df)
        #     df = df.merge(prostt5_embeddings, on='code1', how='left')

        df['aligner'] = df.apply(_assign_aligner, axis=1)
        # df = df[df['aligner'] != 'true']  # TODO - removed all True MSAs
        # df = df[df['aligner'] == 'mafft']  # TODO - removed all True MSAs

        if self.true_score_name == 'RF_phangorn_norm':
            df = df[df[self.true_score_name] != 'ERROR']
            df[self.true_score_name] = df[self.true_score_name].astype(float)


        # df = df[df['dpos_from_true'] != 0] #TODO - comment
        # summary = _summarize_data(df)  # TODO
        # df = df[df['taxa_num'] >= 40]  #TODO - comment
        # df = df[(df['taxa_num'] >= 20) & (df['taxa_num'] <= 40)]
        # df = df[df['taxa_num'] == 40]  #TODO - comment
        # summary = _summarize_data(df)  # TODO
        # df = _get_balanced_dpos_distribution(df)
        # summary = _summarize_data(df)  # TODO
        # df = df[(df['dpos_from_true'] <= 0.6)]

        #
        # df = _balanced_sample_by_code1_and_taxa(
        #     df)  # TODO - comment, this is just to get balanced set per taxa with 1600 alt MSAs for each code
        # summary = _summarize_data(df)  # TODO

        # df = df[df['code1'].str.startswith('BBS', na=False)]

        if self.deduplicated:
            df = df.drop_duplicates(subset=[col for col in df.columns if col != 'code'])  # TODO - comment
            summary = _summarize_data(df)  # TODO

        else: # not dropping all duplicates but very problematic codes
            df2 = df.drop_duplicates(subset=[col for col in df.columns if col != 'code'])
            # dup_counts = df.groupby('code1').size()
            # dup_counts_removed = df.groupby('code1').size() - df2.groupby('code1').size()
            problematic_codes = df2.groupby('code1').filter(lambda x: len(x) < 1100)['code1'].unique() #TODO: change number
            print("Problematic codes to be removed due to too many duplicates:\n", problematic_codes)
            df_cleaned = df[~df['code1'].isin(problematic_codes)]
            df = df_cleaned

        if self.true_score_name != 'RF_phangorn_norm':
            if self.empirical:
                df = _balanced_sample_1600(df)
            else:
                df = _balanced_sample_by_code1_and_taxa(df)
                #TODO - comment, this is just to get balanced set per taxa with 1600 alt MSAs for each code
            summary = _summarize_data(df)


        if self.verbose == 1:
            plt.hist(df[self.true_score_name], bins=100, alpha=0.5, label='Final')
            plt.legend()
            plt.title(f"Distribution of {self.true_score_name}")
            plt.show()

        if self.verbose == 1:
            df.to_csv('./out/features_w_aligner.csv', index=False)

        df = _check_missing_values(df, self.verbose)
        if self.verbose == 1:
            _print_correlations(df, self.true_score_name)

        self._split_into_training_test(df, test_size)

        self._finalize_features(df)
        summary = _summarize_data(df)  # TODO
        self._scale(i=i)

        if self.verbose == 1:
            _print_correlations(self.test_df, self.true_score_name)
        self._save_scaled(i)


    def _scale(self, i: int = 0):
        if self.scaler_type_features == 'standard':
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)  # calculate scaling parameters (fit)
            self.X_test_scaled = self.scaler.transform(self.X_test)  # use the same scaling parameters as in train scaling
            if self.verbose == 1:
                joblib.dump(self.scaler, f'./out/scaler_{i}_mode{self.mode}_{self.true_score_name}.pkl')

        elif self.scaler_type_features == 'rank':
            # self.scaler = GroupAwareScaler(global_scaler=RobustScaler())
            self.scaler = GroupAwareScalerZ(mode="rank", use_global=False, global_scaler=RobustScaler())
            self.X_train_scaled = self.scaler.fit_transform(self.train_df, group_col="code1", feature_cols=self.X_train.columns)
            self.X_test_scaled = self.scaler.transform(self.test_df)
            self.final_features_names = self.scaler.get_feature_names_out()
            if self.verbose == 1:
                self.scaler.save(
                    f'./out/scaler_{i}_mode{self.mode}_{self.true_score_name}.pkl')

            # """ SCALED y-labels """
            # self.y_train_scaled = _rank_percentile_scale_targets(y_true=self.y_train,
            #                                                      group_codes=self.main_codes_train)
            # self.y_test_scaled = _rank_percentile_scale_targets(y_true=self.y_test,
            #                                                     group_codes=self.main_codes_test)
            # self.y_train = self.y_train_scaled
            # self.y_test = self.y_test_scaled
            # """ SCALED y-labels """

        elif self.scaler_type_features == 'zscore':
            self.scaler = GroupAwareScalerZ(mode="zscore", use_global=False, global_scaler=RobustScaler())
            self.X_train_scaled = self.scaler.fit_transform(self.train_df, group_col="code1",
                                                            feature_cols=self.X_train.columns)
            self.X_test_scaled = self.scaler.transform(self.test_df)
            # self.final_features_names = [f"{c}_zscore" for c in self.X_train.columns]
            self.final_features_names = self.scaler.get_feature_names_out()

            if self.verbose == 1:
                self.scaler.save(f'./out/scaler_{i}_mode{self.mode}_{self.true_score_name}.pkl')

            # Scale labels (targets) within each MSA-batch using z-score
            # self.y_train_scaled = _zscore_targets_per_group(self.y_train, self.main_codes_train)
            # self.y_test_scaled = _zscore_targets_per_group(self.y_test, self.main_codes_test)
            # self.y_train_scaled = _zscore_targets_global(self.y_train)
            # self.y_test_scaled = _zscore_targets_global(self.y_test)
            # self.y_train_scaled = _rank_percentile_scale_targets(y_true=self.y_train,
            #                                                      group_codes=self.main_codes_train)
            # self.y_test_scaled = _rank_percentile_scale_targets(y_true=self.y_test,
            #                                                     group_codes=self.main_codes_test)
            #
            # self.y_train = self.y_train_scaled
            # self.y_test = self.y_test_scaled

        if self.scaler_type_labels == 'rank':
            """ SCALED y-labels """
            self.y_train_scaled = _rank_percentile_scale_targets(y_true=self.y_train,
                                                                 group_codes=self.main_codes_train)
            self.y_test_scaled = _rank_percentile_scale_targets(y_true=self.y_test,
                                                                group_codes=self.main_codes_test)
            self.y_train = self.y_train_scaled
            self.y_test = self.y_test_scaled
            """ SCALED y-labels """

        elif self.scaler_type_labels == 'zscore':
            """ SCALED y-labels """
            self.y_train_scaled = _zscore_targets_per_group(self.y_train, self.main_codes_train)
            self.y_test_scaled = _zscore_targets_per_group(self.y_test, self.main_codes_test)
            self.y_train = self.y_train_scaled
            self.y_test = self.y_test_scaled
            """ SCALED y-labels """

        ### if standard, do nothing to y-labels

        # Check the size of each set
        if self.verbose == 1:
            print(f"Training set size: {self.train_df.shape}")
            print(f"Test set size: {self.test_df.shape}")

        self.X_train_scaled = self.X_train_scaled.astype('float64')
        self.X_test_scaled = self.X_test_scaled.astype('float64')
        if self.true_score_name != 'class_label':
            self.y_train = self.y_train.astype('float64')
            self.y_test = self.y_test.astype('float64')
        elif self.true_score_name == 'class_label':
            self.y_train = self.y_train.astype('int')
            self.y_test = self.y_test.astype('int')

        if self.verbose == 1:
            print(f"Training set size (final): {self.X_train_scaled.shape}")
            print(f"Test set size  (final): {self.X_test_scaled.shape}")


    def _save_scaled(self, i):
        # writing train set into csv
        x_train_scaled_to_save = pd.DataFrame(self.X_train_scaled)
        # x_train_scaled_to_save.columns = self.X_train.columns #TODO uncomment this line
        x_train_scaled_to_save['code'] = self.file_codes_train.reset_index(drop=True)
        x_train_scaled_to_save['code1'] = self.main_codes_train.reset_index(drop=True)
        x_train_scaled_to_save['class_label'] = self.y_train.reset_index(drop=True) #TODO uncomment this line
        if self.verbose == 1:
            x_train_scaled_to_save.to_csv(
                f'./out/train_scaled_{i}.csv', index=False)
            self.train_df.to_csv(f'./out/train_unscaled_{i}.csv',
                                 index=False)

        # writing test set into csv
        x_test_scaled_to_save = pd.DataFrame(self.X_test_scaled)
        # x_test_scaled_to_save.columns = self.X_test.columns #TODO uncomment this line
        x_test_scaled_to_save['code'] = self.file_codes_test.reset_index(drop=True)
        x_test_scaled_to_save['code1'] = self.main_codes_test.reset_index(drop=True)
        x_test_scaled_to_save['class_label'] = self.y_test.reset_index(drop=True) #TODO uncomment this line
        if self.verbose == 1:
            x_test_scaled_to_save.to_csv(f'./out/test_scaled_{i}.csv',
                                         index=False)
            self.test_df.to_csv(f'./out/test_unscaled_{i}.csv',
                                index=False)


    def _split_into_training_test(self, df, test_size):
        self.unique_code1 = df['code1'].unique()
        # self.train_code1, self.test_code1 = train_test_split(self.unique_code1,
        #                                            test_size=test_size)
        self.train_code1, self.test_code1 = train_test_split(self.unique_code1, test_size=test_size, random_state=42) # TODO add random state for reproducability

        if self.verbose == 1:
            print(f"the training set is: {self.train_code1} \n")
            print(f"the testing set is: {self.test_code1} \n")

        # Create training and test DataFrames by filtering based on 'code1'
        self.train_df = df[df['code1'].isin(self.train_code1)]

        # train only on PRANK and BALIPHY
        # self.train_df = self.train_df[self.train_df['aligner'].isin(['prank', 'baliphy'])]  #TODO - trying to train with PRANK and BALIPHY only

        self.test_df = df[df['code1'].isin(self.test_code1)]
        # self.train_codes = train_code1
        # self.test_codes = test_code1

        self.main_codes_train = self.train_df['code1']
        self.file_codes_train = self.train_df['code']
        self.aligners_train = self.train_df['aligner']

        self.main_codes_test = self.test_df['code1']
        self.file_codes_test = self.test_df['code']
        self.aligners_test = self.test_df['aligner']


    def _finalize_features(self, df):
        columns_to_drop_dft = ['ssp_from_true', 'dseq_from_true', 'dpos_from_true', 'RF_phangorn_norm', 'code', 'code1',
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
            self.X_train = self.train_df.drop(
                columns=columns_to_drop_dft)
            self.X_test = self.test_df.drop(
                columns=columns_to_drop_dft)

        # if self.mode == 2:
        #     self.X = df.drop(
        #         columns=columns_to_drop_extended)
        #     self.X_train = self.train_df.drop(
        #         columns=columns_to_drop_extended)
        #     self.X_test = self.test_df.drop(
        #         columns=columns_to_drop_extended)

        if self.mode == 3:
            self.X = df[columns_to_choose]
            self.X_train = self.train_df[
                columns_to_choose]
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
            self.X_train = self.X_train.drop(columns=to_drop)
            self.X_test = self.X_test.drop(columns=to_drop)

        # Set train and test Labels
        self.y_train = self.train_df[self.true_score_name]
        self.y_test = self.test_df[self.true_score_name]

    def deep_learning(self, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2, verbose: int = 1,
                      learning_rate: float = 0.01, neurons: list[int] = [0, 128, 64, 16], dropout_rate: float = 0.2,
                      l1: float = 1e-5, l2: float = 1e-5, i: int = 0, undersampling: bool = False, repeats: int = 1,
                      mixed_portion: float = 0, top_k: int = 4, mse_weight: float = 1, ranking_weight: float = 1,
                      per_aligner: bool = False, loss_fn: Literal["mse", "custom_mse", "hybrid_mse_ranknet_loss",
                      "hybrid_mse_approx_ndcg_loss", "hybrid_mse_ranknet_dynamic", "kendall_loss",
                      "approx_ndcg_loss"] = 'mse', regularizer_name: Literal["l1", 'l2','l1_l2'] = 'l2',
                      batch_generation: Literal['standard', 'custom'] = 'standard', alpha: float = 0.5,
                      eps: float = 1e-5) -> tuple[float, float, float, float, float, float, float, float, float]:
        history = None
        tf.config.set_visible_devices([], 'GPU') #disable GPU in tensorflow

        def rank_loss(y_true: tf.Tensor, y_pred: tf.Tensor, top_k: int) -> tf.Tensor:
            tf.compat.v1.enable_eager_execution()
            @tf.function
            def print_func(y_true, y_pred, paired, sorted_paired):
                tf.print("y_true:", y_true)
                tf.print("y_pred:", y_pred)
                tf.print("y_paired:", paired)
                tf.print("sorted_paired:", sorted_paired)
                return y_true, y_pred, paired, sorted_paired

            paired = tf.stack([y_true, y_pred], axis=1)

            sorted_indices = tf.argsort(paired[:, 0], axis=0, direction='ASCENDING')
            sorted_paired = tf.gather(paired, sorted_indices, axis=0)

            true_top_k = sorted_paired[:, :top_k, 0]
            pred_top_k = sorted_paired[:, :top_k, 1]
            rank_diff = K.mean(K.square(K.cast(true_top_k - pred_top_k, dtype=tf.float32)))

            return rank_diff

        # Combine MSE loss with rank-based loss
        def mse_with_rank_loss(y_true: tf.Tensor, y_pred: tf.Tensor, top_k: int = 4, mse_weight: float = 1, ranking_weight: float = 0.3) -> tf.Tensor:

            mse_loss = K.mean(K.square(K.cast(y_true - y_pred, dtype=tf.float32)))  # MSE loss
            top_k_rank_loss = rank_loss(y_true, y_pred, top_k)
            mse_weight = tf.cast(mse_weight, dtype=tf.float32)
            ranking_weight = tf.cast(ranking_weight, dtype=tf.float32)
            top_k_rank_loss = tf.cast(top_k_rank_loss, dtype=tf.float32)
            total_loss = mse_weight * mse_loss + ranking_weight * top_k_rank_loss

            return total_loss

        def topk_ranking_loss(y_true: tf.Tensor, y_pred: tf.Tensor, top_k: int=3, margin: float =0.0) -> tf.Tensor:
            # Flatten into shape (batch_size,)
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])

            n = tf.shape(y_true)[0]

            # Indices of the true top-k (lowest true label = best MSA)
            true_order = tf.argsort(y_true, axis=0)  # ascending
            true_topk = true_order[:top_k]  # shape (k,)

            # Predicted scores for all items
            pred_all = y_pred  # shape (n,)

            # Predicted scores for the true top-k
            pred_topk = tf.gather(y_pred, true_topk)  # shape (k,)

            # Expand for pairwise comparison:
            # pred_topk_expanded: (k, n)
            # pred_all_expanded:  (k, n)
            pred_topk_expanded = pred_topk[:, None]
            pred_all_expanded = pred_all[None, :]

            # Hinge ranking loss:
            # Want: pred_topk[i] + margin < pred_all[j] for all j outside top_k
            loss_matrix = tf.nn.relu(margin + pred_topk_expanded - pred_all_expanded)

            # Create mask to ignore comparisons inside the top-k set
            topk_mask = tf.scatter_nd(
                indices=true_topk[:, None],
                updates=tf.ones_like(true_topk, dtype=tf.float32),
                shape=(n,)
            )

            # Mask: 1 for non-top-k, 0 for top-k
            non_topk_mask = 1.0 - topk_mask  # shape (n,)
            non_topk_mask = non_topk_mask[None, :]  # broadcast to (k, n)

            # Apply mask (only penalize when comparing top-k to non-top-k)
            loss_matrix = loss_matrix * non_topk_mask

            # Final loss: average over all valid comparisons
            return tf.reduce_mean(loss_matrix)

        def mse_with_topk_rank_loss(y_true: tf.Tensor, y_pred: tf.Tensor, top_k: int = 4,
                               ranking_weight: float = 0.3, margin: float = 0.0) -> tf.Tensor:

            mse_loss = K.mean(K.square(K.cast(y_true - y_pred, dtype=tf.float32)))  # MSE loss
            top_k_rank_loss = topk_ranking_loss(y_true, y_pred, top_k=top_k, margin=margin)
            ranking_weight = tf.cast(ranking_weight, dtype=tf.float32)
            top_k_rank_loss = tf.cast(top_k_rank_loss, dtype=tf.float32)
            total_loss = mse_loss + ranking_weight * top_k_rank_loss

            return total_loss

        def kendall_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
            diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)
            sign_true = tf.sign(diff_true)
            # smooth sign for differentiability
            sign_pred = tf.tanh(diff_pred)
            tau = tf.reduce_mean(sign_true * sign_pred)
            # mask = 1.0 - tf.eye(tf.shape(y_true)[0])
            # tau = tf.reduce_sum(sign_true * sign_pred * mask) * 2.0 / (
            #             tf.cast(tf.shape(y_true)[0], tf.float32) * (tf.cast(tf.shape(y_true)[0], tf.float32) - 1.0))

            return 1 - tau  # maximize Kendall's τ

        def soft_kendall_loss(y_true, y_pred, tau=1.0): #TAU 1.0-5.0
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
            diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

            sign_true = tf.nn.tanh(tau * diff_true)
            sign_pred = tf.nn.tanh(tau * diff_pred)

            corr = tf.reduce_mean(sign_true * sign_pred)
            return 1 - corr


        def ranknet_loss(y_true, y_pred, margin=0.0, eps=1e-4):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
            diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

            S = tf.cast(diff_true < 0, tf.float32)
            P = tf.nn.sigmoid(diff_pred - margin)

            # tf.print("P min:", tf.reduce_min(P), "P max:", tf.reduce_max(P), "S unique:",
            #          tf.unique(tf.reshape(S, [-1]))[0])

            loss = - (S * tf.math.log(P + eps) + (1 - S) * tf.math.log(1 - P + eps))
            # loss = - (S * tf.math.log(P + 1e-8) + (1 - S) * tf.math.log(1 - P + 1e-8))
            return tf.reduce_mean(loss)

        def hybrid_mse_ranknet_loss(y_true, y_pred, alpha=0.5, eps=1e-4):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            rank_loss = ranknet_loss(y_true, y_pred, eps=eps)
            total_loss = alpha * mse + (1 - alpha) * rank_loss

            # tf.print("MSE:", mse, "Rank loss:", rank_loss, "Total loss:", total_loss, "Ratio rank/mse:", rank_loss/mse)

            return total_loss

        def hybrid_mse_ranknet_dynamic(y_true, y_pred, alpha_base=0.98, eps=1e-6):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            rank_loss = ranknet_loss(y_true, y_pred, eps=eps)
            ratio = tf.stop_gradient(rank_loss / (mse + eps))
            alpha = tf.clip_by_value(alpha_base / (1 + ratio), 0.9, 0.995)
            return alpha * mse + (1 - alpha) * rank_loss

        def listnet_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            true_prob = tf.nn.softmax(y_true, axis=-1)
            pred_prob = tf.nn.softmax(y_pred, axis=-1)

            loss = -tf.reduce_sum(true_prob * tf.math.log(pred_prob + 1e-10))
            return loss


        def approx_ndcg_loss(y_true, y_pred, eps=1e-10):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # Pairwise differences
            pred_diffs = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

            # Soft ranks (expected ranks from sigmoid pairwise comparisons)
            soft_rank = tf.reduce_sum(tf.nn.sigmoid(-pred_diffs), axis=-1) + 1.0

            # Compute gains and discounts
            gains = tf.pow(2.0, y_true) - 1.0
            discounts = 1.0 / tf.math.log(1.0 + soft_rank) / tf.math.log(2.0)

            dcg = tf.reduce_sum(gains * discounts)
            # Ideal DCG (with perfect ranking)
            sorted_true = tf.sort(y_true, direction='DESCENDING')
            ideal_gains = tf.pow(2.0, sorted_true) - 1.0
            ideal_discounts = 1.0 / tf.math.log(1.0 + tf.range(1, tf.size(y_true) + 1, dtype=tf.float32)) / tf.math.log(
                2.0)
            idcg = tf.reduce_sum(ideal_gains * ideal_discounts)

            ndcg = dcg / (idcg + eps)
            return 1.0 - ndcg  # Loss to minimize

        def hybrid_mse_approx_ndcg_loss(y_true, y_pred, alpha=0.5, eps=1e-10):
            """
            Hybrid loss combining Mean Squared Error and ApproxNDCG loss.
            alpha: weight for MSE (0.0–1.0)
            eps: numerical stability
            """
            # ----- MSE component -----
            mse = tf.reduce_mean(tf.square(y_true - y_pred))

            # ----- ApproxNDCG component -----
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # Pairwise differences for smooth rank
            pred_diffs = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

            # Soft rank approximation
            soft_rank = tf.reduce_sum(tf.nn.sigmoid(-pred_diffs), axis=-1) + 1.0

            # Gains and discounts
            gains = tf.pow(2.0, y_true) - 1.0
            discounts = 1.0 / (tf.math.log(1.0 + soft_rank) / tf.math.log(2.0))

            dcg = tf.reduce_sum(gains * discounts)

            # Ideal DCG
            sorted_true = tf.sort(y_true, direction='DESCENDING')
            ideal_gains = tf.pow(2.0, sorted_true) - 1.0
            ideal_discounts = 1.0 / (
                        tf.math.log(1.0 + tf.range(1, tf.size(y_true) + 1, dtype=tf.float32)) / tf.math.log(2.0))
            idcg = tf.reduce_sum(ideal_gains * ideal_discounts)

            ndcg = dcg / (idcg + eps)
            ndcg_loss = 1.0 - ndcg

            # ----- Combine -----
            total_loss = alpha * mse + (1.0 - alpha) * ndcg_loss
            return total_loss

        def lambda_rank_loss(y_true, y_pred, eps=1e-10):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # Pairwise differences
            diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
            diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

            # Compute pairwise labels: 1 if i < j (true_i < true_j means i is better)
            S = tf.cast(diff_true < 0, tf.float32)

            # Delta NDCG weighting (approximate gain drop if misordered)
            gain = tf.pow(2.0, y_true) - 1.0
            rank = tf.argsort(tf.argsort(-y_true)) + 1.0
            discount = 1.0 / tf.math.log1p(rank) / tf.math.log(2.0)
            dcg = tf.reduce_sum(gain * discount)
            delta_ndcg = tf.abs(tf.expand_dims(discount, 1) - tf.expand_dims(discount, 0))

            P = tf.nn.sigmoid(diff_pred)
            loss = delta_ndcg * (S * -tf.math.log(P + eps) + (1 - S) * -tf.math.log(1 - P + eps))
            return tf.reduce_mean(loss)

        def weighted_ranknet_loss(y_true, y_pred, topk_weight_decay=0.3, eps=1e-6):
            """
            Weighted RankNet loss giving higher importance to top-ranked (lowest y_true) items.
            y_true: smaller = better (e.g. percentile rank)
            """
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # --- Pairwise differences ---
            diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
            diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

            # --- Pairwise label: 1 if i should rank higher than j (i.e., smaller y_true) ---
            S = tf.cast(diff_true < 0, tf.float32)  # notice the '<' instead of '>'
            P = tf.nn.sigmoid(-diff_pred)  # flip sign so that smaller y_pred = higher rank

            # --- Compute item weights: higher weight for smaller (better) y_true ---
            ranks = tf.argsort(tf.argsort(y_true))  # smaller = better
            item_weights = tf.exp(-topk_weight_decay * tf.cast(ranks, tf.float32))
            pair_weights = (tf.expand_dims(item_weights, 1) + tf.expand_dims(item_weights, 0)) / 2.0

            # --- Weighted RankNet loss ---
            loss = -pair_weights * (S * tf.math.log(P + eps) + (1.0 - S) * tf.math.log(1.0 - P + eps))

            # normalize
            loss = tf.reduce_sum(loss) / (tf.reduce_sum(pair_weights) + eps)
            return loss

        def hybrid_weighted_ranknet_loss(
                y_true,
                y_pred,
                topk_weight_decay=0.3,
                alpha=0.5,
                beta=0.2,
                margin=0.2,
                eps=1e-6
        ):
            """
            Hybrid RankNet loss designed for 'smaller = better' targets (e.g. percentile ranks).
            Emphasizes correct ordering *and* separation of the best item in each group.

            Parameters
            ----------
            y_true : tensor of shape [n_items]
                Ground truth ranks or scores (smaller = better).
            y_pred : tensor of shape [n_items]
                Model-predicted scores (smaller = better).
            topk_weight_decay : float
                Controls how fast weights decay for lower-ranked items.
            alpha : float
                Weight for pointwise 'best classification' term.
            beta : float
                Weight for margin separation term.
            margin : float
                Minimum margin enforced between the best item and others.
            """

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # -----------------------------
            # (1) Pairwise RankNet component
            # -----------------------------
            diff_true = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
            diff_pred = tf.expand_dims(y_pred, 1) - tf.expand_dims(y_pred, 0)

            # i should rank higher if its true label is smaller
            S = tf.cast(diff_true < 0, tf.float32)
            P = tf.nn.sigmoid(-diff_pred)

            ranks = tf.argsort(tf.argsort(y_true))
            item_weights = tf.exp(-topk_weight_decay * tf.cast(ranks, tf.float32))
            pair_weights = (tf.expand_dims(item_weights, 1) + tf.expand_dims(item_weights, 0)) / 2.0

            pairwise_loss = -pair_weights * (
                    S * tf.math.log(P + eps) + (1.0 - S) * tf.math.log(1.0 - P + eps)
            )
            pairwise_loss = tf.reduce_sum(pairwise_loss) / (tf.reduce_sum(pair_weights) + eps)

            # -----------------------------
            # (2) "Best item" classification component
            # -----------------------------
            # Find the best (lowest) item(s)
            min_val = tf.reduce_min(y_true)
            best_mask = tf.cast(tf.equal(y_true, min_val), tf.float32)

            # Encourage the best item to have low predicted score
            best_class_loss = -tf.reduce_mean(
                best_mask * tf.math.log(tf.nn.sigmoid(-y_pred) + eps)
            )

            # -----------------------------
            # (3) Margin-based separation component
            # -----------------------------
            # For all items except the best, enforce a margin gap
            non_best_mask = 1.0 - best_mask
            best_pred = tf.reduce_min(y_pred)
            margin_loss = tf.reduce_mean(
                non_best_mask * tf.nn.relu(margin - (y_pred - best_pred))
            )

            # -----------------------------
            # (4) Combine
            # -----------------------------
            total_loss = pairwise_loss + alpha * best_class_loss + beta * margin_loss

            return total_loss

        # MODEL PART
        model = Sequential()
        model.add(Input(shape=(self.X_train_scaled.shape[1],)))

        if regularizer_name == "l1":
            ker_regularizer = regularizers.l1(l1=l1)

        elif regularizer_name == "l2":
            ker_regularizer = regularizers.l2(l2=l2)

        elif regularizer_name == "l1_l2":
            ker_regularizer = regularizers.l1_l2(l1=l1, l2=l2)

        else:
            raise ValueError(f"Invalid regularizer_name: {regularizer_name}, the allowed options are 'l1', 'l2' or 'l1_l2'\n")

        if not neurons[0] == 0:
            # first hidden
            model.add(
                Dense(neurons[0], kernel_initializer=GlorotUniform(), kernel_regularizer=ker_regularizer))
            model.add(BatchNormalization())
            # model.add(LeakyReLU(negative_slope=0.01))
            model.add(PReLU())
            model.add(Dropout(dropout_rate))

        if not neurons[1] == 0:
            # second hidden
            model.add(
                Dense(neurons[1], kernel_initializer=GlorotUniform(), kernel_regularizer=ker_regularizer))
            model.add(BatchNormalization())
            # model.add(LeakyReLU(negative_slope=0.01))
            model.add(PReLU())
            model.add(Dropout(dropout_rate))


        if not neurons[2] == 0:
            # third hidden
            model.add(
                Dense(neurons[2], kernel_initializer=GlorotUniform(), kernel_regularizer=ker_regularizer))
            model.add(BatchNormalization()) #normalizing before the activation function
            # model.add(LeakyReLU(negative_slope=0.01))
            model.add(PReLU())
            model.add(Dropout(dropout_rate))

        if not neurons[3] == 0:
            # fourth hidden
            model.add(Dense(neurons[3], kernel_initializer=GlorotUniform(),kernel_regularizer=ker_regularizer))
            model.add(BatchNormalization())
            # model.add(LeakyReLU(negative_slope=0.01))
            model.add(PReLU())
            model.add(Dropout(dropout_rate))

        # model.add(Dense(1, activation='sigmoid'))  #limits output to 0 to 1 range
        model.add(Dense(1, activation='linear'))  #TODO - not sure if linear is better

        optimizer = Adam(learning_rate=learning_rate)
        # optimizer = RMSprop(learning_rate=learning_rate)

        if loss_fn == 'mse':
            model.compile(optimizer=optimizer, loss='mean_squared_error')

        elif loss_fn == "custom_mse":
            model.compile(optimizer=optimizer,
                          loss = lambda y_true, y_pred:
                          mse_with_rank_loss(
                              y_true, y_pred,
                              top_k=top_k,
                              mse_weight=mse_weight,
                              ranking_weight=ranking_weight))

        elif loss_fn == "hybrid_mse_ranknet_loss":
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          hybrid_mse_ranknet_loss(
                              y_true,
                              y_pred,
                              alpha=alpha,
                              eps=eps))


        elif loss_fn == "kendall_loss":
            model.compile(optimizer=optimizer,
                            loss=lambda y_true, y_pred:
                            kendall_loss(y_true, y_pred))

        elif loss_fn == "listnet_loss":
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          listnet_loss(y_true, y_pred))

        elif loss_fn == "hybrid_mse_ranknet_dynamic":
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          hybrid_mse_ranknet_dynamic(
                              y_true,
                              y_pred,
                              alpha_base=alpha,
                              eps=eps))

        elif loss_fn == "approx_ndcg_loss":
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          approx_ndcg_loss(
                              y_true,
                              y_pred,
                              eps=1e-10))

        elif loss_fn == 'hybrid_mse_approx_ndcg_loss':
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          hybrid_mse_approx_ndcg_loss(
                              y_true,
                              y_pred,
                              alpha=alpha,
                              eps=eps))

        elif loss_fn == 'ranknet_loss':
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          ranknet_loss(
                              y_true,
                              y_pred,
                              margin=0.0,
                              eps=eps))

        elif loss_fn == 'lambda_rank_loss':
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          lambda_rank_loss(
                              y_true,
                              y_pred,
                              eps=eps))

        elif loss_fn == 'weighted_ranknet_loss':
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          weighted_ranknet_loss(
                              y_true,
                              y_pred,
                              topk_weight_decay=alpha,
                              eps=eps))

        elif loss_fn == 'hybrid_weighted_ranknet_loss':
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          hybrid_weighted_ranknet_loss(
                              y_true,
                              y_pred,
                              topk_weight_decay=alpha,
                              alpha=0.5,
                              beta=0.2,
                              margin=0.2,
                              eps=eps))

        elif loss_fn == 'mse_with_topk_rank_loss':
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred:
                          mse_with_topk_rank_loss(
                              y_true,
                              y_pred,
                              top_k=top_k,
                              ranking_weight=ranking_weight,
                              margin=0.0))


        unique_train_codes = self.main_codes_train.unique()
        train_msa_ids, val_msa_ids = train_test_split(unique_train_codes, test_size=0.2, random_state=42)  # TODO add random state for reproducability
        if self.verbose == 1:
            print(f"the training set is: {train_msa_ids} \n")
            print(f"the validation set is: {val_msa_ids} \n")
        # x_train_scaled_with_names = pd.DataFrame(self.X_train_scaled)
        # x_train_scaled_with_names.columns = self.X_train.columns
        batch_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train,
                                         true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids,
                                         val_msa_ids=val_msa_ids, aligners =self.aligners_train, batch_size=batch_size,
                                         validation_split=validation_split, is_validation=False, repeats=repeats,
                                         mixed_portion=mixed_portion, per_aligner=per_aligner, features_w_names=self.train_df)

        val_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train,
                                       true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids,
                                       val_msa_ids=val_msa_ids, aligners = self.aligners_train,
                                       batch_size=batch_size, validation_split=validation_split,
                                       is_validation=True, repeats=repeats, mixed_portion=mixed_portion,
                                       per_aligner=per_aligner, features_w_names=self.train_df)

        # Callback 1: early stopping
        # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # Callback 2: learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',  # to monitor
            patience=3,  # number of epochs with no improvement before reducing the learning rate
            verbose=1,
            factor=0.5,  # factor by which the learning rate will be reduced
            min_lr=1e-6  # lower bound on the learning rate
            # min_delta=1e-5  # the threshold for val loss improvement - to identify the plateau
        )

        # ranking_callback = PerBatchRankingMetrics(val_generator=val_generator, metric="kendall", verbose=1)
        # ranking_callback2 = PerBatchRankingMetrics(val_generator=val_generator, metric="spearman", verbose=1)

        val_kendall = None
        val_spearman = None
        if batch_generation == 'custom':
            callbacks = [
                early_stopping,
                lr_scheduler
                # ranking_callback,
                # ranking_callback2
            ]
            history = model.fit(batch_generator, epochs=epochs, validation_data=val_generator, verbose=verbose,
                                    callbacks=callbacks)
            # val_kendall = model.history.history["val_kendall"][-1] # get last kendall correlation (can get max instead if I want
            # val_spearman = model.history.history["val_spearman"][-1] # get last spearman correlation (can get max instead if I want

        elif batch_generation == 'standard':
            callbacks = [
                early_stopping,
                lr_scheduler
            ]
            history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size,
                                validation_split=validation_split, verbose=verbose,
                                callbacks=callbacks)

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        epochs = range(1, len(history.history['loss']) + 1)  # Integer epoch numbers
        plt.xticks(ticks=epochs)  # Set the ticks to integer epoch numbers

        plt.legend()
        if self.verbose == 1:
            plt.savefig(fname=f'./out/loss_graph_{i}_mode{self.mode}_{self.true_score_name}.png', format='png')
        plt.show()
        plt.close()

        # visualize model architecture
        if self.verbose == 1:
            plot_model(model, to_file=f'./out/model_architecture_{i}_mode{self.mode}_{self.true_score_name}.png', show_shapes=True, show_layer_names=True,
                       show_layer_activations=True)
            model.save(f'./out/regressor_model_{i}_mode{self.mode}_{self.true_score_name}.keras')
            # plot_model(model, to_file='./out/model_architecture.dot', show_shapes=True, show_layer_names=True)
            plot_model(
                model,
                to_file='./out/model_architecture.png',  # use .png or .jpg or .svg
                show_shapes=True,
                show_layer_names=True
            )

        # substrings = ['original', 'concat']
        # X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
        # mask = X_test_scaled_with_names['code'].str.contains('|'.join(substrings), case=False, na=False)
        # self.X_test_scaled = self.X_test_scaled[~mask]
        loss = model.evaluate(self.X_test_scaled, self.y_test)
        if self.verbose == 1:
            print(f"Test Loss: {loss}")

        self.y_pred = model.predict(self.X_test_scaled)
        self.y_pred = np.ravel(self.y_pred)  # flatten multi-dimensional array into one-dimensional
        self.y_pred = self.y_pred.astype('float64')

        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred
        })

        if self.verbose == 1:
            df_res.to_csv(f'./out/prediction_DL_{i}_mode{self.mode}_{self.true_score_name}.csv', index=False)

        mse = mean_squared_error(self.y_test, self.y_pred)
        if self.verbose == 1:
            print(f"Mean Squared Error: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")

        ### Evaluate per MSA-batch correlation and average over all MSA-batches
        # --- Compute per-MSA average correlation ---
        df_corr = pd.DataFrame({
            "msa_code": self.main_codes_test,
            "y_true": self.y_test,
            "y_pred": self.y_pred
        })

        per_msa_corrs = []
        per_msa_topk_corrs = []

        for msa_id, group in df_corr.groupby("msa_code"):

            if group["y_true"].nunique() > 1 and group["y_pred"].nunique() > 1:
                r, _ = pearsonr(group["y_true"], group["y_pred"])
                per_msa_corrs.append(r)

            group_topk = group.nlargest(top_k, "y_pred")
            if group_topk["y_true"].nunique() > 1 and group_topk["y_pred"].nunique() > 1:
                r_topk, _ = pearsonr(group_topk["y_true"], group_topk["y_pred"])
                per_msa_topk_corrs.append(r_topk)

        if len(per_msa_corrs) > 0:
            avg_per_msa_corr = np.mean(per_msa_corrs)
            avg_per_msa_topk_corr = np.mean(per_msa_topk_corrs)
            median_per_msa_corr = np.median(per_msa_corrs)

        else:
            avg_per_msa_corr = np.nan
            avg_per_msa_topk_corr = np.nan
            median_per_msa_corr = np.nan


        print(f"Average per-MSA Pearson correlation: {avg_per_msa_corr:.4f}")
        print(f"Median per-MSA Pearson correlation: {median_per_msa_corr:.4f}")
        print(f"Average per-MSA Top-{top_k} Pearson correlation: {avg_per_msa_topk_corr:.4f}")


        ### START - TRY Evaluating per-MSA top-50 metric % ###
        df_eval = pd.DataFrame({
            "msa_code": self.main_codes_test,
            "file_code": self.file_codes_test,
            "true_score": self.y_test,
            "predicted_score": self.y_pred
        })

        msa_stats = []

        for msa_id, group in df_eval.groupby("msa_code"):
            # Find index of best (lowest) predicted score
            best_pred_idx = group["predicted_score"].idxmin()
            best_true_score = group.loc[best_pred_idx, "true_score"]

            # Rank true scores ascending (1 = best)
            group = group.sort_values("true_score", ascending=True).reset_index(drop=True)
            group["true_rank"] = np.arange(1, len(group) + 1)

            # Find the true rank of the best-predicted sample
            best_pred_true_rank = group.loc[group["true_score"] == best_true_score, "true_rank"].values[0]

            # Check if within top 50 (or total count if group smaller)
            if best_pred_true_rank <= min(50, len(group)):
                msa_stats.append(1)
            else:
                msa_stats.append(0)

        top50_percentage = 100 * np.mean(msa_stats)
        print(
            f"Percentage of MSA groups where best predicted score is in top 50 true labels: {top50_percentage:.2f}%")

        ### END - TRY ADDING AND OPTIMIZING % OF MSA-BATCHES WHERE PREDICTED IN TRUE TOP50 ####

        if self.explain_features_importance:
            try:
                # # explain features importance
                if self.final_features_names is not None: #TODO testing this option
                    X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.final_features_names)
                else:
                    X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
                X_test_subset = X_test_scaled_with_names.sample(n=500, random_state=42)  # Take a sample of 500 rows
                explainer = shap.Explainer(model, X_test_subset)
                shap_values = explainer(X_test_subset)
                # explainer = shap.Explainer(model, X_test_scaled_with_names)
                # shap_values = explainer(X_test_scaled_with_names)
                if self.verbose == 1:
                    joblib.dump(explainer,
                                f'./out/explainer_{i}_mode{self.mode}_{self.true_score_name}.pkl')
                    joblib.dump(shap_values,
                                f'./out/shap_values__{i}_mode{self.mode}_{self.true_score_name}.pkl')
                # matplotlib.use('TkAgg')
                matplotlib.use('Agg')

                feature_names = [
                    a + ": " + str(b) for a, b in zip(X_test_subset.columns, np.abs(shap_values.values).mean(0).round(3))
                ]

                shap.summary_plot(shap_values, X_test_subset, max_display=40, feature_names=feature_names)
                # shap.summary_plot(shap_values, X_test_scaled_with_names, max_display=30, feature_names=feature_names)
                if self.verbose == 1:
                    plt.savefig(f'./out/summary_plot_{i}.png', dpi=300,
                                bbox_inches='tight')
                # plt.show()
                plt.close()

                shap.plots.waterfall(shap_values[0], max_display=40)
                if self.verbose == 1:
                    plt.savefig(f'./out/waterfall_plot_{i}.png', dpi=300,
                                bbox_inches='tight')
                # plt.show()
                plt.close()

                shap.force_plot(shap_values[0], X_test_subset[0], matplotlib=True, show=False)
                if self.verbose == 1:
                    plt.savefig(f'./out/force_plot_{i}.png')
                # plt.show()
                plt.close()

                shap.plots.bar(shap_values, max_display=40)
                if self.verbose == 1:
                    plt.savefig(f'./out/bar_plot_{i}.png', dpi=300,
                                bbox_inches='tight')
                # plt.show()
                plt.close()
            except Exception as e:
                print(f"Did not manage to get features importance: {e}\n")

        # return mse
        val_loss = history.history["val_loss"][-1]
        return mse, loss, val_loss, corr_coefficient, avg_per_msa_corr, avg_per_msa_topk_corr, top50_percentage, val_kendall, val_spearman

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
        plt.savefig(fname=f'./out/regression_results_{i}_mode{self.mode}_{self.true_score_name}.png', format='png')
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
        plt.savefig(fname=f'./out/regression_results_{i}_mode{self.mode}_{self.true_score_name}2.png', format='png')
        plt.show()
        plt.close()


        if model_name == "rf" or model_name == "gbr":
            importances = self.regressor.feature_importances_
            indices = np.argsort(importances)[::-1]

            importances_df = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': importances
            })

            top_n = 15
            top_features = importances_df.iloc[indices].head(top_n)

            # Plot feature importances
            plt.figure(figsize=(22, 21))
            plt.title(f'{title} Feature Importances', fontsize=20)
            plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
            plt.xlabel('Importance', fontsize=18)
            plt.ylabel('Features', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(fname=f'./out/features_importances_{i}_mode{self.mode}_{self.true_score_name}.png', format='png')
            plt.show()
            plt.close()
