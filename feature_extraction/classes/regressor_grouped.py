import math
import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, classification_report, precision_recall_curve, roc_curve
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal, List, Any, Iterator, Tuple, Optional
from scipy.stats import pearsonr, gaussian_kde, norm, rankdata
import visualkeras
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import shap
import pickle
import pydot
import tensorflow as tf
# import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation, BatchNormalization, Input, ELU, Attention, Reshape, Embedding, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from feature_extraction.classes.attention_layer import AttentionLayer

def assign_aligner(row: pd.Series) -> str:
    code = row['code'].lower()
    not_mafft = ['muscle', 'prank', '_true.fas', 'true_tree.txt', 'bali_phy', 'baliphy', 'original']

    if not any(sub in code for sub in not_mafft):
        return 'mafft'
    if 'muscle' in code:
        return 'muscle'
    elif 'prank' in code:
        return 'prank'
    elif 'bali_phy' in code or 'baliphy' in code:
        return 'baliphy'

    return 'true'

def check_file_type(file_path: str) -> str:
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.parquet':
        return 'parquet'
    elif file_extension == '.csv':
        return 'csv'
    else:
        return 'unknown'

def assign_class_label(group: pd.DataFrame) -> pd.DataFrame:
    max_sop_row = group.loc[group['sop_score'].idxmax()]
    sop_dpos = max_sop_row['dpos_dist_from_true']
    group['class_label'] = (group['dpos_dist_from_true'] < sop_dpos).astype(int)
    return group

def assign_class_label_test(group: pd.DataFrame) -> pd.DataFrame:
    mask = group['code'].str.contains("concat|_alt", case=False, na=False)
    group_without_extra = group[~mask]
    if not group_without_extra.empty:
        max_sop_row = group_without_extra.loc[group_without_extra['sop_score'].idxmax()]
        sop_dpos = max_sop_row['dpos_dist_from_true']
        group['class_label'] = (group['dpos_dist_from_true'] < sop_dpos).astype(int)
        # percentile_20 = group_without_extra['dpos_dist_from_true'].quantile(0.2)
        # group['class_label'] = (group['dpos_dist_from_true'] <= percentile_20).astype(int)
    else:
        group['class_label'] = np.nan
    return group

def rank_percentile_scale_targets(y_true: pd.Series, group_codes: pd.Series) -> pd.Series:
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
        scaled_series.loc[group_df.index] = scaled

    return scaled_series


# class SmoothCDFTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.params_ = {}  # Store mean and std for each feature
#
#     def fit(self, X, y=None):
#         # Convert X to a pandas DataFrame if it's not already
#         X = pd.DataFrame(X)
#
#         # Select only numerical columns (just like how Scalers handle)
#         self.features_ = X.select_dtypes(include=[np.number]).columns.tolist()
#
#         for feature in self.features_:
#             values = X[feature].values
#             mu, sigma = np.mean(values), np.std(values)
#             self.params_[feature] = (mu, sigma)
#
#         return self
#
#     def transform(self, X):
#         # Ensure that the input is a pandas DataFrame
#         X = pd.DataFrame(X)
#
#         # Transform the data
#         X_transformed = X.copy()
#
#         for feature in self.features_:
#             mu, sigma = self.params_[feature]
#             values = X[feature].values
#             # Apply CDF of normal distribution
#             cdf_vals = norm.cdf(values, loc=mu, scale=sigma)
#             X_transformed[f"{feature}_cdf"] = cdf_vals
#
#         return X_transformed
#
#
# class SmoothCDFReplacer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.params_ = {}
#
#     def fit(self, X, y=None):
#         X = pd.DataFrame(X)
#         self.features_ = X.select_dtypes(include=[np.number]).columns.tolist()
#
#         for feature in self.features_:
#             values = X[feature].values
#             mu = np.mean(values)
#             sigma = np.std(values)
#             self.params_[feature] = (mu, sigma)
#
#         return self
#
#     def transform(self, X):
#         X = pd.DataFrame(X).copy()
#
#         for feature in self.features_:
#             mu, sigma = self.params_[feature]
#             values = X[feature].values
#
#             if sigma == 0:
#                 # Avoid NaNs: if all values are the same, assign CDF=0.5
#                 X[feature] = 0.5
#             else:
#                 X[feature] = norm.cdf(values, loc=mu, scale=sigma)
#
#         return X.values


class GroupAwareScaler:
    def __init__(self, global_scaler=None):
        self.global_scaler = global_scaler or StandardScaler()
        self.feature_names = None
        self.group_col = None
        self.fitted = False

    def fit(self, df: pd.DataFrame, group_col: str, feature_cols: list):
        self.feature_names = feature_cols
        self.group_col = group_col
        self.global_scaler.fit(df[feature_cols])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted.")

        df = df.copy()

        # Drop index level if group_col is both in index and columns
        if self.group_col in df.index.names and self.group_col in df.columns:
            df.index = df.index.droplevel(self.group_col)
        elif self.group_col in df.index.names:
            df = df.reset_index()

        # Apply global scaler
        global_scaled = self.global_scaler.transform(df[self.feature_names])

        # Pre-allocate for rank-percentile scaled features
        rank_scaled = np.zeros_like(global_scaled)

        # For each feature, compute per-group ranks in row order
        for i, feature in enumerate(self.feature_names):
            percentiles = np.zeros(len(df))
            for group_value, group_df in df.groupby(self.group_col):
                group_idx = df.index.get_indexer(group_df.index)
                vals = group_df[feature].values
                if len(vals) == 1:
                    p = np.array([0.0])
                else:
                    ranks = rankdata(vals, method="average")
                    p = (ranks - 1) / (len(vals) - 1)
                percentiles[group_idx] = p
            rank_scaled[:, i] = percentiles

        # Combine global_scaled and rank_scaled
        combined = np.concatenate([global_scaled, rank_scaled], axis=1)
        return combined #TODO uncomment this line
        # return rank_scaled

    def fit_transform(self, df: pd.DataFrame, group_col: str, feature_cols: list) -> np.ndarray:
        self.fit(df, group_col, feature_cols)
        return self.transform(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "global_scaler": self.global_scaler,
                "feature_names": self.feature_names,
                "group_col": self.group_col,
                "fitted": self.fitted
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.global_scaler = data["global_scaler"]
            self.feature_names = data["feature_names"]
            self.group_col = data["group_col"]
            self.fitted = data["fitted"]

    def get_feature_names_out(self) -> list:
        """Returns names like ['f1_scaled', ..., 'f1_rank', ...]"""
        scaled = [f"{f}_scaled" for f in self.feature_names]
        ranked = [f"{f}_rank" for f in self.feature_names]
        return scaled + ranked


class BatchGenerator(Sequence):
    def __init__(self, features, true_labels, true_msa_ids, train_msa_ids, val_msa_ids, aligners, batch_size, validation_split=0.2, is_validation=False, repeats=1, mixed_portion=0.3, per_aligner=False, classification_task = False, features_w_names=np.nan):
        self.features = features
        self.true_labels = np.asarray(true_labels)
        self.msa_ids = true_msa_ids  # TRUE MSA IDs (categorical)
        self.batch_size = batch_size
        # self.unique_msa_ids = np.unique(true_msa_ids)[np.unique(true_msa_ids) != "AATF"]  #TODO remove AATF from features file
        self.unique_msa_ids = np.unique(true_msa_ids)
        self.validation_split = validation_split
        self.is_validation = is_validation
        self.val_msa_ids = val_msa_ids
        self.train_msa_ids = train_msa_ids
        self.repeats = repeats
        self.mixed_portion = mixed_portion
        self.per_aligner = per_aligner
        self.aligners = aligners
        self.unique_aligners = np.unique(aligners)[np.unique(aligners) != "true"]
        # self.features_w_names = features_w_names
        self.features_w_names = features_w_names.reset_index(drop=True)

        self.classification_task = classification_task

        if self.is_validation:
            mask = np.isin(self.msa_ids, self.val_msa_ids)

        else:
            mask = np.isin(self.msa_ids, self.train_msa_ids)

        self.features = self.features[mask]
        self.true_labels = self.true_labels[mask]
        self.features_w_names = self.features_w_names[mask]
        self.features_w_names = self.features_w_names.reset_index(drop=True)
        self.msa_ids = self.msa_ids[mask]
        self.unique_msa_ids = np.unique(self.msa_ids)
        self.batches = self._precompute_batches()

    def _split_idx_into_batches(self, idx: np.ndarray) -> Tuple[List[Any], List[Any]]:
        batches: List[Any] = []
        remaining_samples_set: List[Any] = []
        # np.random.shuffle(idx)
        num_samples = len(idx)
        num_full_batches = num_samples // self.batch_size
        remaining_samples = num_samples % self.batch_size
        leaving_out = math.floor(self.mixed_portion * num_full_batches)

        for i in range(num_full_batches - leaving_out): # I want to leave out some batches into the mix of remaining samples
            batch_idx = idx[i * self.batch_size: (i + 1) * self.batch_size]
            # if self.classification_task:
            #     labels = self._precompute_true_labels(batch_idx)
            #     batches.append((self.features[batch_idx], labels))
            # else:
            #     batches.append((self.features[batch_idx], self.true_labels[batch_idx]))
            batches.append((self.features[batch_idx], self.true_labels[batch_idx]))

        if remaining_samples > 0 or leaving_out > 0: # intermixed batches (consisting of the samples from different unique MSA IDs) to make sure that
            remaining_samples_set.extend(idx[(num_full_batches - leaving_out) * self.batch_size:])
        np.random.shuffle(remaining_samples_set)
        np.random.shuffle(batches)
        return batches, remaining_samples_set

    # def _split_sorted_idx_into_batches(self, idx):
    #     if len(idx) > 0:
    #         sorted_features = self.features_w_names.iloc[idx]
    #         sorted_features = sorted_features.sort_values(by='dpos_dist_from_true', ascending=True)
    #         # sorted_features = sorted_features.sort_values(by='dpos_ng_dist_from_true', ascending=True)
    #         sorted_indices = sorted_features.index.values #TODO - check that actual indices are reordered by sorting
    #         batches, remaining_samples_set = self._split_idx_into_batches(sorted_indices)
    #         return batches, remaining_samples_set

    def _precompute_batches(self) -> List[Any]:
        batches: List[Any] = []
        batches_mix: List[Any] = []
        remaining_samples_set: List[Any] = []

        for msa_id in self.unique_msa_ids:
            try:
                for k in range(self.repeats): #testing an option to produce different batch mixes
                    idx = np.where(self.msa_ids == msa_id)[0]
                    if len(idx) > self.batch_size:
                        if self.per_aligner:
                            for aligner in self.unique_aligners:
                                idx_aln = np.intersect1d(np.where(self.aligners == aligner)[0], idx)
                                if len(idx_aln) > self.batch_size:
                                    np.random.shuffle(idx_aln) #TODO check that shuffling here instead of within _split_idx_into_batches doesn't mess with the results
                                    btchs, rem_sam_set = self._split_idx_into_batches(idx_aln)
                                    batches.extend(btchs)
                                    remaining_samples_set.extend(rem_sam_set)
                                else:
                                    continue
                        else:
                            np.random.shuffle(idx) #TODO check that shuffling here instead of within _split_idx_into_batches doesn't mess with the results
                            btchs, rem_sam_set = self._split_idx_into_batches(idx)
                            batches.extend(btchs)
                            remaining_samples_set.extend(rem_sam_set)

                        # add batches with close dpos values' MSAs (sorted by dpos)
                        # btchs, rem_sam_set = self._split_sorted_idx_into_batches(idx)
                        # batches.extend(btchs)
                        # remaining_samples_set.extend(rem_sam_set)

                    # np.random.shuffle(idx)
                    # num_samples = len(idx)
                    # num_full_batches = num_samples // self.batch_size
                    # remaining_samples = num_samples % self.batch_size
                    # leaving_out = math.floor(self.mixed_portion * num_full_batches)
                    #
                    # for i in range(num_full_batches - leaving_out): # I want to leave out some batches into the mix of remaining samples
                    #     batch_idx = idx[i * self.batch_size: (i + 1) * self.batch_size]
                    #     batches.append((self.features[batch_idx], self.true_labels[batch_idx]))
                    #
                    #
                    # if remaining_samples > 0 or leaving_out > 0: # intermixed batches (consisting of the samples from different unique MSA IDs) to make sure that
                    #     remaining_samples_set.extend(idx[(num_full_batches - leaving_out) * self.batch_size:])
                    # np.random.shuffle(remaining_samples_set)
                    # np.random.shuffle(batches)

            except Exception as e:
                print(f"Exception {e}\n")

        remaining_samples_set = np.array(remaining_samples_set)
        np.random.shuffle(remaining_samples_set)

        for i in range(0, len(remaining_samples_set), self.batch_size):
            batch_idx = remaining_samples_set[i: i + self.batch_size]
            # batch_idx = idx[i * self.batch_size: (i + 1) * self.batch_size]
            if len(batch_idx) == self.batch_size:
                # if self.classification_task:
                #     labels = self._precompute_true_labels(batch_idx)
                #     batches_mix.append((self.features[batch_idx], labels))
                # else:
                batches_mix.append((self.features[batch_idx], self.true_labels[batch_idx]))
            # if len(batch_idx) == self.batch_size:  # Ensure full batch size
            #     batches_mix.append((self.features[batch_idx], self.true_labels[batch_idx]))

        final_batches = batches + batches_mix
        np.random.shuffle(final_batches)

        return final_batches

    # def _precompute_true_labels(self, idx):
    #     if len(idx) > 0:
    #         features = self.features_w_names.iloc[idx]
    #         features = features.reset_index(drop=True)
    #         # percentile_10 = features['dpos_dist_from_true'].quantile(0.1)
    #         # labels = (features['dpos_dist_from_true'] <= percentile_10).astype(int)
    #         # labels = labels.to_numpy()
    #
    #         max_sop_row = features.loc[features['sop_score'].idxmax()]
    #         sop_dpos = max_sop_row['dpos_dist_from_true']
    #         # sop_dpos = max_sop_row['dpos_ng_dist_from_true']
    #         # print("Features shape:", features.shape)
    #         # print("Max SOP Row:", max_sop_row)
    #         # print("SoP dpos:", sop_dpos)
    #         labels = (features['dpos_dist_from_true'] < sop_dpos).astype(int)
    #         # labels = (features['dpos_ng_dist_from_true'] < sop_dpos).astype(int)
    #         labels = labels.to_numpy()
    #     else:
    #         labels = np.nan
    #         print("empty batch\n")
    #     return labels
    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> Any:
        return self.batches[idx]

    def on_epoch_end(self) -> None:
        if not self.is_validation:
            self.batches = self._precompute_batches()
        np.random.shuffle(self.batches)

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        for idx in range(len(self)):
            batch_features, batch_labels = self[idx]
            yield (batch_features, batch_labels)
class Regressor:
    '''
    features_file: file with all features and labels
    test_size: portion of the codes to be separated into a test set; all MSAs for that specific code would be on the same side of the train-test split
    mode: 1 is all features, 2 is all except SoP features, 3 is only 2 SoP features'''
    def __init__(self, features_file: str, test_size: float, mode: int = 1, remove_correlated_features: bool = False, predicted_measure: Literal['msa_distance', 'class_label'] = 'msa_distance', i=0) -> None:
        self.features_file: str = features_file
        self.test_size: float = test_size
        self.predicted_measure: Literal['msa_distance', 'class_label'] = predicted_measure
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

        # self.train_codes, self.test_codes = None, None
        # self._prepare_data()


        # df = pd.read_csv(self.features_file)
        # df_extra = pd.read_csv("/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_extra900_features_260225.csv")
        # df = pd.concat([df, df_extra], ignore_index=True)
        # to make sure that all dataset codes are read as strings and not integers
        # df = pd.read_parquet(self.features_file, engine='pyarrow')
        df = pd.DataFrame()
        file_type = check_file_type(self.features_file)
        if file_type == 'parquet':
            df = pd.read_parquet(self.features_file, engine='pyarrow')
        elif file_type == 'csv':
            df = pd.read_csv(self.features_file)
        else:
            print(f"features file is of unknown format\n")

        true_score_name = None
        if self.predicted_measure == 'msa_distance':
            # true_score_name = "dpos_dist_from_true"
            true_score_name = "dpos_ng_dist_from_true"
        elif self.predicted_measure == 'tree_distance':
            true_score_name = 'normalized_rf'
        elif self.predicted_measure == 'class_label':
            true_score_name = 'class_label'

        df['code1'] = df['code1'].astype(str)
        # df = df[df['code1'] != 'AATF'] #TODO delete

        # substrings = ['original', 'concat', '_alt_']
        # mask = df['code'].str.contains('|'.join(substrings), case=False, na=False)
        # df = df[mask] #TODO delete: here I chose only the MSAs we produces with small dpos

        # df['aligner'] = df.apply(assign_aligner, axis=1)
        # df = df[df['aligner'] != 'true'] #removed true MSAs from the data
        # df = pd.get_dummies(df, columns=['aligner'], prefix='aligner') #added one-hot encoding for msa aligner program with the columns names of the form "aligner_mafft", "aligner_..."; the aligner column is automatically replaced/removed

        df['aligner'] = df.apply(assign_aligner, axis=1)

        df = df.groupby('code1').apply(assign_class_label)
        # df = df.reset_index(drop=True)
        # df['class_label'] = df.groupby(['code1', 'aligner']).apply(assign_class_label_test).reset_index(level=[0, 1], drop=True)['class_label']
        df['class_label'] = df['class_label'].astype(int)

        # df = df[df['aligner'] != 'muscle'] #TODO delete: here I remove all muscle msa's

        # df = df[df['class_label'] == 1] #TODO delete: here I chose only the MSAs we produces with small dpos

        df.to_csv('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/features_w_label.csv', index=False)

        #filter BALIPHY ONLY
        # df = df[df['code'].str.contains('bali_phy|BALIPHY', case=False, na=False, regex=True)]
        # df = df[df['code'].str.contains('prank', case=False, na=False, regex=True)]
        # df = df[~df['code'].str.contains('prank|bali_phy|BALIPHY', case=False, na=False, regex=True)]

        # Check for missing values
        print("Missing values in each column:\n", df.isnull().sum())
        corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df[true_score_name])
        # corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df['dpos_ng_dist_from_true'])
        print(f"Pearson Correlation of Normalized SOP and dpos: {corr_coefficient1:.4f}\n", f"P-value of non-correlation: {p_value1:.6f}\n")
        corr_coefficient1, p_value1 = pearsonr(df['sop_score'], df[true_score_name])
        # corr_coefficient1, p_value1 = pearsonr(df['sop_score'], df['dpos_ng_dist_from_true'])
        print(f"Pearson Correlation of SOP and dpos: {corr_coefficient1:.4f}\n",
              f"P-value of non-correlation: {p_value1:.6f}\n")

        # add normalized_rf
        df["normalized_rf"] = df['rf_from_true']/(df['taxa_num']-1)
        # df["class_label"] = np.where(df['dpos_dist_from_true'] <= 0.02, 0, 1)
        # df["class_label2"] = np.where(df['dpos_dist_from_true'] <= 0.015, 0, np.where(df['dpos_dist_from_true'] <= 0.1, 1, 2))


        class_label_counts = df['class_label'].dropna().value_counts()
        print(class_label_counts)

        # class_label2_counts_train = df['class_label_test'].dropna().value_counts()
        # print(class_label2_counts_train)

        df = df.dropna()

        # unique_code1 = df['code1'].unique()
        # encoder = OneHotEncoder(sparse=False)
        # encoded_msa_ids = encoder.fit_transform(unique_code1.reshape(-1, 1))
        # df['true_msa_ids_embed'] = encoded_msa_ids

        # true_score_name = None
        # if self.predicted_measure == 'msa_distance':
        #     # true_score_name = "dpos_dist_from_true"
        #     true_score_name = "dpos_ng_dist_from_true"
        # elif self.predicted_measure == 'tree_distance':
        #     true_score_name = 'normalized_rf'
        # elif self.predicted_measure == 'class_label':
        #     true_score_name = 'class_label'

        self.y = df[true_score_name]

        # all features
        if mode == 1:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'normalised_sop_score', 'aligner', 'dpos_ng_dist_from_true'])
        if mode == 2:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label_test', 'sop_score', 'normalised_sop_score'])
        if mode == 3:
            # self.X = df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]
            # self.X = df[['sop_score']]
            self.X = df[['constant_sites_pct', 'sop_score', 'entropy_mean', 'sp_score_subs_norm', 'sp_ge_count', 'number_of_gap_segments','nj_parsimony_score','msa_len','num_cols_no_gaps','total_gaps','entropy_var','num_unique_gaps','sp_score_gap_e_norm','k_mer_10_mean','av_gaps','n_unique_sites','skew_bl','median_bl','bl_75_pct','avg_unique_gap', 'k_mer_20_var','k_mer_10_top_10_norm', 'gaps_2seq_len3plus', 'gaps_1seq_len3plus', 'num_cols_1_gap', 'single_char_count']]
        if mode == 4:
            self.X = df.drop(
                columns=['dpos_dist_from_true', 'dpos_ng_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'code', 'code1', 'aligner',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len', 'clustal_mid_root', 'clustal_differential_sum'])


        if remove_correlated_features:
            correlation_matrix = self.X.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)] #TODO to uncomment
            # to_drop = ['entropy_pct_75', 'gaps_len_one',
            #  'gaps_len_two', 'gaps_len_three_plus',  'gaps_1seq_len1', 'gaps_1seq_len2',
            #  'gaps_1seq_len3', 'gaps_1seq_len3plus',  'num_cols_1_gap', 'num_cols_2_gaps',
            #  'num_cols_all_gaps_except1', 'sp_match_ratio', 'sp_missmatch_ratio', 'single_char_count',
            #  'double_char_count', 'bl_sum', 'kurtosis_bl', 'bl_std', 'bl_max', 'k_mer_10_max', 'k_mer_10_var',
            #  'k_mer_10_pct_90', 'k_mer_10_norm', 'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_pct_95', 'k_mer_20_pct_90',
            #  'k_mer_20_norm',  'number_of_mismatches', 'sp_score_subs', 'nj_parsimony_sd'] #TODO to comment
            print("Correlated features to drop:", to_drop)
            self.X = self.X.drop(columns=to_drop)

        unique_code1 = df['code1'].unique()

        train_code1, test_code1 = train_test_split(unique_code1, test_size=test_size) #TODO add random state for reproducability
        # train_code1, test_code1 = train_test_split(unique_code1, test_size=test_size, random_state=42)

        print(f"the training set is: {train_code1} \n")
        print(f"the testing set is: {test_code1} \n")

        # Create training and test DataFrames by filtering based on 'code1'
        self.train_df = df[df['code1'].isin(train_code1)]
        self.test_df = df[df['code1'].isin(test_code1)]
        # self.train_codes = train_code1
        # self.test_codes = test_code1

        class_label_counts_train = self.train_df['class_label'].dropna().value_counts()
        print(class_label_counts_train)

        # class_label2_counts_train = self.train_df['class_label2'].dropna().value_counts()
        # print(class_label2_counts_train)

        class_label_counts_test = self.test_df['class_label'].dropna().value_counts()
        print(class_label_counts_test)

        # class_label2_counts_test = self.test_df['class_label2'].dropna().value_counts()
        # print(class_label2_counts_test)


        # all features
        if mode == 1:
            self.X_train = self.train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'normalised_sop_score', 'aligner', 'dpos_ng_dist_from_true'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'normalised_sop_score', 'aligner', 'dpos_ng_dist_from_true'])
        if mode == 2:
            self.X_train = self.train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf','class_label', 'code', 'code1', 'pypythia_msa_difficulty','sop_score', 'normalised_sop_score', 'aligner'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf','class_label', 'code', 'code1', 'pypythia_msa_difficulty', 'sop_score', 'normalised_sop_score', 'aligner'])

        # 2 sop features
        if mode == 3:
            # self.X_train = self.train_df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]
            # self.X_test = self.test_df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]
            self.X_train = self.train_df[['constant_sites_pct', 'sop_score', 'entropy_mean', 'sp_score_subs_norm', 'sp_ge_count', 'number_of_gap_segments','nj_parsimony_score','msa_len','num_cols_no_gaps','total_gaps','entropy_var','num_unique_gaps','sp_score_gap_e_norm','k_mer_10_mean','av_gaps','n_unique_sites','skew_bl','median_bl','bl_75_pct','avg_unique_gap', 'k_mer_20_var','k_mer_10_top_10_norm', 'gaps_2seq_len3plus', 'gaps_1seq_len3plus', 'num_cols_1_gap', 'single_char_count']]
            self.X_test = self.test_df[['constant_sites_pct', 'sop_score', 'entropy_mean', 'sp_score_subs_norm', 'sp_ge_count', 'number_of_gap_segments','nj_parsimony_score','msa_len','num_cols_no_gaps','total_gaps','entropy_var','num_unique_gaps','sp_score_gap_e_norm','k_mer_10_mean','av_gaps','n_unique_sites','skew_bl','median_bl','bl_75_pct','avg_unique_gap', 'k_mer_20_var','k_mer_10_top_10_norm', 'gaps_2seq_len3plus', 'gaps_1seq_len3plus', 'num_cols_1_gap', 'single_char_count']]

        if mode == 4:
            self.X_train = self.train_df.drop(
                columns=['dpos_dist_from_true', 'dpos_ng_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'code', 'code1', 'aligner',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len', 'clustal_mid_root', 'clustal_differential_sum'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'dpos_ng_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'code', 'code1', 'aligner',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len', 'clustal_mid_root', 'clustal_differential_sum']
            )

        if remove_correlated_features:
            self.X_train = self.X_train.drop(columns=to_drop)
            self.X_test = self.X_test.drop(columns=to_drop)

        # self.scaler = MinMaxScaler() #TODO uncomment this line
        # self.scaler = RobustScaler()
        # self.scaler = StandardScaler()
        # self.scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=5000)
        # self.X_train_scaled = self.scaler.fit_transform(self.X_train)  # calculate scaling parameters (fit) #TODO uncomment this line
        # self.X_test_scaled = self.scaler.transform(self.X_test)  # use the same scaling parameters as in train scaling #TODO uncomment this line

        # joblib.dump(self.scaler, f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/scaler_{i}_mode{self.mode}_{self.predicted_measure}.pkl') #TODO uncomment this line

        scaler = GroupAwareScaler(global_scaler=RobustScaler())
        self.X_train_scaled = scaler.fit_transform(self.train_df, group_col="code1", feature_cols=self.X_train.columns)
        self.X_test_scaled = scaler.transform(self.test_df)
        scaler.save(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/scaler_{i}_mode{self.mode}_{self.predicted_measure}.pkl')
        self.final_features_names = scaler.get_feature_names_out()
        # scaler = GroupAwareScaler()
        # scaler.load("group_aware_scaler.pkl")


        self.main_codes_train = self.train_df['code1']
        self.file_codes_train = self.train_df['code']
        self.aligners_train = self.train_df['aligner']
        class_weights = compute_class_weight('balanced', classes=np.unique(self.train_df['class_label']), y=self.train_df['class_label'])
        self.weights = dict(enumerate(class_weights))
        print(self.weights)
        self.main_codes_test = self.test_df['code1']
        self.file_codes_test = self.test_df['code']
        self.aligners_test = self.test_df['aligner']

        corr_coefficient1, p_value1 = pearsonr(self.test_df['normalised_sop_score'], self.test_df[true_score_name])
        # corr_coefficient1, p_value1 = pearsonr(self.test_df['normalised_sop_score'],
        #                                        self.test_df['dpos_ng_dist_from_true'])
        print(f"Pearson Correlation of Normalized SOP and dpos in the TEST set: {corr_coefficient1:.4f}\n",
              f"P-value of non-correlation: {p_value1:.4f}\n")
        corr_coefficient1, p_value1 = pearsonr(self.test_df['sop_score'],
                                               self.test_df[true_score_name])
        # corr_coefficient1, p_value1 = pearsonr(self.test_df['sop_score'],
        #                                         self.test_df['dpos_ng_dist_from_true'])
        print(f"Pearson Correlation of SOP and dpos in the TEST set: {corr_coefficient1:.4f}\n",
              f"P-value of non-correlation: {p_value1:.4f}\n")

        # Set train and test Labels
        self.y_train = self.train_df[true_score_name]
        self.y_test = self.test_df[true_score_name]
        # self.dpos_train = self.train_df['dpos_dist_from_true']

        """ REMOVE this section """
        self.y_train_scaled = rank_percentile_scale_targets(y_true =self.y_train , group_codes = self.main_codes_train) #TODO remove this line
        self.y_test_scaled = rank_percentile_scale_targets(y_true=self.y_test, group_codes=self.main_codes_test)  # TODO remove this line
        self.y_train = self.y_train_scaled # TODO remove this line
        self.y_test = self.y_test_scaled # TODO remove this line
        """ REMOVE this section """

        self.binary_feature = self.train_df['class_label'].astype('float64')

        # Check the size of each set
        print(f"Training set size: {self.train_df.shape}")
        print(f"Test set size: {self.test_df.shape}")

        self.X_train_scaled = self.X_train_scaled.astype('float64')
        self.X_test_scaled = self.X_test_scaled.astype('float64')
        if true_score_name != 'class_label':
            self.y_train = self.y_train.astype('float64')
            self.y_test = self.y_test.astype('float64')
        elif true_score_name == 'class_label':
            self.y_train = self.y_train.astype('int')
            self.y_test = self.y_test.astype('int')

        print(f"Training set size (final): {self.X_train_scaled.shape}")
        print(f"Test set size  (final): {self.X_test_scaled.shape}")

        # writing train set into csv
        x_train_scaled_to_save = pd.DataFrame(self.X_train_scaled)
        # x_train_scaled_to_save.columns = self.X_train.columns #TODO uncomment this line
        x_train_scaled_to_save['code'] = self.file_codes_train.reset_index(drop=True)
        x_train_scaled_to_save['code1'] = self.main_codes_train.reset_index(drop=True)
        x_train_scaled_to_save['class_label'] = self.y_train.reset_index(drop=True) #TODO uncomment this line
        # x_train_scaled_to_save['class_label'] = self.y_train
        # x_train_scaled_to_save['class_label_test'] = ...
        x_train_scaled_to_save.to_csv(
            f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/train_scaled_{i}.csv', index=False)
        self.train_df.to_csv(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/train_unscaled_{i}.csv',
                             index=False)

        # writing test set into csv
        x_test_scaled_to_save = pd.DataFrame(self.X_test_scaled)
        # x_test_scaled_to_save.columns = self.X_test.columns #TODO uncomment this line
        x_test_scaled_to_save['code'] = self.file_codes_test.reset_index(drop=True)
        x_test_scaled_to_save['code1'] = self.main_codes_test.reset_index(drop=True)
        x_test_scaled_to_save['class_label'] = self.y_test.reset_index(drop=True) #TODO uncomment this line
        # x_test_scaled_to_save['class_label'] = self.y_test
        # x_test_scaled_to_save['class_label_test'] = ...
        x_test_scaled_to_save.to_csv(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/test_scaled_{i}.csv',
                                     index=False)
        self.test_df.to_csv(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/test_unscaled_{i}.csv',
                            index=False)

    def deep_learning(self, epochs: int = 50, batch_size: int = 16, validation_split: float = 0.2, verbose: int = 1, learning_rate: float = 0.01, dropout_rate: float = 0.2, l1: float = 1e-5, l2: float = 1e-5, i: int = 0, undersampling: bool = False, repeats: int = 1, mixed_portion: float = 0.3, top_k: int = 4, mse_weight: float = 0, ranking_weight: float = 50, per_aligner: bool = False) -> float:
        history = None
        tf.config.set_visible_devices([], 'GPU') #disable GPU in tensorflow

        def low_score_weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            error = tf.abs(y_true - y_pred)
            weight = tf.maximum(1.0, 1.0 / (y_true + 1e-6))
            weighted_error = weight * error
            return tf.reduce_mean(weighted_error ** 2)

        def weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            weighted_loss = mse_loss * weights
            return weighted_loss

        def weighted_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor, factor: Optional[float] = 7.0) -> tf.Tensor:
            weights = K.exp(-y_true)  # Op1: Higher weights for lower scores
            # weights = 1/(1 + K.exp(factor*(y_true-0.5)))  # Op2: Higher weights for lower scores
            mse_loss = K.mean(weights * K.square(y_true - y_pred))  # Weighted MSE
            return mse_loss

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
            # print_func(y_true, y_pred, paired, sorted_paired)

            true_top_k = sorted_paired[:, :top_k, 0]
            pred_top_k = sorted_paired[:, :top_k, 1]
            # tf.print("true_top_k:", true_top_k)
            # tf.print("pred_top_k:", pred_top_k)

            rank_diff = K.mean(K.square(K.cast(true_top_k - pred_top_k, dtype=tf.float32)))
            # tf.print("Rank Diff:", rank_diff)

            return rank_diff

        # @tf.function
        # def pairwise_rank_loss(y_true, y_pred, margin=0.0, top_k = 4):
        #     n = tf.shape(y_true)[0]
        #
        #     y_true_flat = tf.reshape(y_true, [-1])
        #     _, top_k_indices = tf.math.top_k(-y_true_flat, k=top_k, sorted=True)  # Use negative to get smallest values
        #     mask = tf.reduce_any(tf.equal(tf.reshape(tf.range(n), [-1, 1]), tf.reshape(top_k_indices, [1, -1])), axis=1)
        #
        #     i_indices = tf.reshape(tf.range(n), [-1, 1])
        #     j_indices = tf.reshape(tf.range(n), [1, -1])
        #     i_indices_flat = tf.reshape(i_indices, [-1])
        #     j_indices_flat = tf.reshape(j_indices, [-1])
        #     y_true_i = tf.gather(y_true, i_indices_flat)
        #     y_true_j = tf.gather(y_true, j_indices_flat)
        #     y_pred_i = tf.gather(y_pred, i_indices_flat)
        #     y_pred_j = tf.gather(y_pred, j_indices_flat)
        #
        #     y_true_diff = tf.cast(y_true_i < y_true_j, tf.float32)
        #     pairwise_loss = tf.maximum(0.0, y_pred_i - y_pred_j + margin)
        #
        #     loss = tf.reduce_sum(pairwise_loss * y_true_diff * tf.cast(tf.reshape(mask, [-1, 1]), tf.float32))
        #
        #     return loss

        # Combine MSE loss with rank-based loss
        def mse_with_rank_loss(y_true: tf.Tensor, y_pred: tf.Tensor, top_k: int = 4, mse_weight: float = 1, ranking_weight: float = 0.3) -> tf.Tensor:

            mse_loss = K.mean(K.square(K.cast(y_true - y_pred, dtype=tf.float32)))  # MSE loss
            # mse_loss = tf.keras.losses.MSE(y_true, y_pred)
            top_k_rank_loss = rank_loss(y_true, y_pred, top_k)
            # rank_loss = pairwise_rank_loss(y_true, y_pred, margin=1.0, top_k=top_k)
            mse_weight = tf.cast(mse_weight, dtype=tf.float32)
            ranking_weight = tf.cast(ranking_weight, dtype=tf.float32)
            top_k_rank_loss = tf.cast(top_k_rank_loss, dtype=tf.float32)
            total_loss = mse_weight * mse_loss + ranking_weight * top_k_rank_loss

            return total_loss

        @tf.function
        def min_score_penalty_loss(y_true: tf.Tensor, y_pred: tf.Tensor, mse_weight: float = 1.0, min_penalty_weight: float = 50.0) -> tf.Tensor:
            # mse_loss = K.mean(K.square(y_true - y_pred))
            #absolute error
            mse_loss = K.mean(K.abs(y_true - y_pred))

            min_true_index = tf.argmin(y_true, axis=0)

            min_pred = tf.gather(y_pred, min_true_index)
            min_true = tf.gather(y_true, min_true_index)

            # min_penalty = tf.maximum(0.0,
            #                          min_pred - min_true)  # penalize if prediction is greater than true minimum
            # absolute error of top1
            min_penalty = tf.abs(min_pred - min_true)
            # min_penalty = K.square(min_pred - min_true)

            total_loss = mse_weight * mse_loss + min_penalty_weight * min_penalty

            return total_loss

        # non-negative regression msa_distance task
        if self.predicted_measure == 'msa_distance':
            model = Sequential()
            model.add(Input(shape=(self.X_train_scaled.shape[1],)))

            #first hidden
            model.add(
                Dense(128, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2)))
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

            # second hidden
            model.add(
                Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2)))
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

            # third hidden
            model.add(Dense(16, kernel_initializer=GlorotUniform(),kernel_regularizer=regularizers.l2(l2=l2)))
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

            # # # fourth hidden
            # model.add(
            #     Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2)))
            # model.add(LeakyReLU(negative_slope=0.01))
            # model.add(BatchNormalization())
            # model.add(Dropout(dropout_rate))
            #
            # # # # fifth hidden
            # model.add(
            #     Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            # model.add(LeakyReLU(negative_slope=0.01))
            # model.add(BatchNormalization())
            # model.add(Dropout(dropout_rate))

            model.add(Dense(1, activation='sigmoid'))  #limits output to 0 to 1 range

            optimizer = Adam(learning_rate=learning_rate)
            # optimizer = RMSprop(learning_rate=learning_rate)

            # model.compile(optimizer=optimizer, loss='mean_squared_error')
            # model.compile(optimizer=optimizer, loss=low_score_weighted_mse)
            # model.compile(optimizer=optimizer, loss=mse_with_rank_loss)
            # model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: weighted_mse_loss(y_true, y_pred))
            # model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: weighted_mse(y_true, y_pred, weights))
            model.compile(optimizer=optimizer, loss = lambda y_true, y_pred: mse_with_rank_loss(y_true, y_pred, top_k=top_k, mse_weight=mse_weight,
                                                        ranking_weight=ranking_weight))
            # model.compile(optimizer=optimizer,
            #               loss=lambda y_true, y_pred: min_score_penalty_loss(y_true, y_pred, mse_weight=1.0, min_penalty_weight=50.0))

            unique_train_codes = self.main_codes_train.unique()
            train_msa_ids, val_msa_ids = train_test_split(unique_train_codes, test_size=0.2)
            print(f"the training set is: {train_msa_ids} \n")
            print(f"the validation set is: {val_msa_ids} \n")
            # x_train_scaled_with_names = pd.DataFrame(self.X_train_scaled)
            # x_train_scaled_with_names.columns = self.X_train.columns
            batch_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train,
                                             true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids, val_msa_ids=val_msa_ids, aligners =self.aligners_train, batch_size=batch_size,
                                             validation_split=validation_split, is_validation=False, repeats=repeats, mixed_portion=mixed_portion, per_aligner=per_aligner, features_w_names=self.train_df)

            val_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train,
                                           true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids, val_msa_ids=val_msa_ids, aligners = self.aligners_train,
                                           batch_size=batch_size, validation_split=validation_split, is_validation=True, repeats=repeats, mixed_portion=mixed_portion, per_aligner=per_aligner, features_w_names=self.train_df)

            # Callback 1: early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-5)
            # Callback 2: learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',  # to monitor
                patience=3,  # number of epochs with no improvement before reducing the learning rate
                verbose=1,
                factor=0.5,  # factor by which the learning rate will be reduced
                min_lr=1e-6,  # lower bound on the learning rate
                min_delta=1e-5  # the threshold for val loss improvement - to identify the plateau
            )
            callbacks = [
                early_stopping,
                lr_scheduler
            ]
            history = model.fit(batch_generator, epochs=epochs, validation_data=val_generator, verbose=verbose,
                                    callbacks=callbacks)

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        epochs = range(1, len(history.history['loss']) + 1)  # Integer epoch numbers
        plt.xticks(ticks=epochs)  # Set the ticks to integer epoch numbers

        plt.legend()
        plt.savefig(fname=f'./out/loss_graph_{i}_mode{self.mode}_{self.predicted_measure}.png', format='png')
        plt.show()
        plt.close()

        # visualize model architecture
        plot_model(model, to_file=f'./out/model_architecture_{i}_mode{self.mode}_{self.predicted_measure}.png', show_shapes=True, show_layer_names=True,
                   show_layer_activations=True)
        model.save(f'./out/regressor_model_{i}_mode{self.mode}_{self.predicted_measure}.keras')
        plot_model(model, to_file='./out/model_architecture.dot', show_shapes=True, show_layer_names=True)

        # substrings = ['original', 'concat']
        # X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
        # mask = X_test_scaled_with_names['code'].str.contains('|'.join(substrings), case=False, na=False)
        # self.X_test_scaled = self.X_test_scaled[~mask]
        loss = model.evaluate(self.X_test_scaled, self.y_test)
        print(f"Test Loss: {loss}")

        self.y_pred = model.predict(self.X_test_scaled)
        self.y_pred = np.ravel(self.y_pred)  # flatten multi-dimensional array into one-dimensional
        self.y_pred = self.y_pred.astype('float64')

        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred
        })

        df_res.to_csv(f'./out/prediction_DL_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")

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
            joblib.dump(explainer,
                        f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/explainer_{i}_mode{self.mode}_{self.predicted_measure}.pkl')
            joblib.dump(shap_values,
                        f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/shap_values__{i}_mode{self.mode}_{self.predicted_measure}.pkl')
            matplotlib.use('Agg')

            feature_names = [
                a + ": " + str(b) for a, b in zip(X_test_subset.columns, np.abs(shap_values.values).mean(0).round(3))
            ]

            shap.summary_plot(shap_values, X_test_subset, max_display=40, feature_names=feature_names)
            # shap.summary_plot(shap_values, X_test_scaled_with_names, max_display=30, feature_names=feature_names)
            plt.savefig(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/summary_plot_{i}.png', dpi=300,
                        bbox_inches='tight')
            # plt.show()
            plt.close()

            shap.plots.waterfall(shap_values[0], max_display=40)
            plt.savefig(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/waterfall_plot_{i}.png', dpi=300,
                        bbox_inches='tight')
            # plt.show()
            plt.close()

            shap.force_plot(shap_values[0], X_test_subset[0], matplotlib=True, show=False)
            plt.savefig(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/force_plot_{i}.png')
            # plt.show()
            plt.close()

            shap.plots.bar(shap_values, max_display=40)
            plt.savefig(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/bar_plot_{i}.png', dpi=300,
                        bbox_inches='tight')
            # plt.show()
            plt.close()
        except Exception as e:
            print(f"Did not manage to get features importance\n")

        return mse


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
            plt.savefig(fname=f'./out/features_importances_{i}_mode{self.mode}_{self.predicted_measure}.png', format='png')
            plt.show()
            plt.close()

    def dl_classifier(self, epochs: int = 50, batch_size: int = 16, validation_split: float = 0.2, verbose: int = 1, learning_rate: float =0.0001, dropout_rate: float = 0.2, l1: float = 1e-5, l2: float = 1e-5, i: int = 0, undersampling: bool = False, repeats: int = 1, mixed_portion: float = 0.1, top_k: int = 4, mse_weight: float = 0, ranking_weight: float = 50, threshold: float = 0.5, per_aligner: bool = False) -> float:

        # def weighted_binary_crossentropy(w0=1.0, w1=1.0):
        #     def loss(y_true, y_pred):
        #         epsilon = tf.keras.backend.epsilon()
        #         y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Prevent log(0)
        #
        #         # Compute the binary cross-entropy loss with weights
        #         loss = - (w1 * y_true * tf.keras.backend.log(y_pred) +
        #                   2 * w0 * (1 - y_true) * tf.keras.backend.log(1 - y_pred))
        #
        #         return tf.reduce_mean(loss)  # Mean loss over the batch
        #
        #     return loss

        def weighted_binary_crossentropy(w0: float = 1.0, w1: float = 1.0) -> callable:
            def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                weights = y_true * w1 + (1 - y_true) * w0 #would it help me to make class1 5 times more important?
                return tf.reduce_mean(loss_fn(y_true, y_pred) * weights)

            return loss


        model = Sequential()
        model.add(Input(shape=(self.X_train_scaled.shape[1],)))

        # first hidden
        model.add(Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2)))
        # model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
        model.add(Activation('relu'))
        # model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))  # Dropout for regularization

        # second hidden
        model.add(Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2)))
        # model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))  # Dropout for regularization

        # third hidden
        model.add(Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2)))
        # model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the third hidden layer
        # model.add(ELU())
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))  # Dropout for regularization

        model.add(Dense(1, activation='sigmoid'))  # limits output to 0 to 1 range


        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = dict(enumerate(class_weights))
        print(class_weight_dict)

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(w0=class_weight_dict[0], w1=class_weight_dict[1]),
                      metrics=['accuracy', metrics.AUC(), metrics.AUC(name='auc_weighted')])
        # model.compile(optimizer=optimizer, loss='binary_crossentropy',
        #               metrics=['accuracy', metrics.AUC(), metrics.AUC(name='auc_weighted')])


        unique_train_codes = self.main_codes_train.unique()
        train_msa_ids, val_msa_ids = train_test_split(unique_train_codes, test_size=0.2)
        print(f"the training set is: {train_msa_ids} \n")
        print(f"the validation set is: {val_msa_ids} \n")
        batch_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train,
                                         true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids, val_msa_ids=val_msa_ids, aligners = self.aligners_train, batch_size=batch_size,
                                         validation_split=validation_split, is_validation=False, repeats=repeats, mixed_portion=mixed_portion, per_aligner=per_aligner, classification_task=True, features_w_names=self.train_df)

        val_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train,
                                       true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids, val_msa_ids=val_msa_ids, aligners = self.aligners_train,
                                       batch_size=batch_size, validation_split=validation_split, is_validation=True, repeats=repeats, mixed_portion=mixed_portion, per_aligner=per_aligner, classification_task=True, features_w_names=self.train_df)


        # 1. Implement early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        # 2. learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',  # Metric to monitor
            patience=3,  # Number of epochs with no improvement to wait before reducing the learning rate
            verbose=1,  # Print messages when learning rate is reduced
            factor=0.7,  # Factor by which the learning rate will be reduced
            min_lr=1e-5  # Lower bound on the learning rate
        )
        callbacks = [early_stopping, lr_scheduler]
        history = model.fit(batch_generator, epochs=epochs, validation_data=val_generator, verbose=verbose,
                            callbacks=callbacks)
        # history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose, callbacks=[early_stopping, lr_scheduler])

        plot_model(model, to_file=f'./out/classifier_model_architecture_{i}_mode{self.mode}_{self.predicted_measure}.png',
                   show_shapes=True, show_layer_names=True,
                   show_layer_activations=True)
        model.save(f'./out/classifer_model_{i}_mode{self.mode}_{self.predicted_measure}.keras')
        plot_model(model, to_file='./out/classifier_model_architecture.dot', show_shapes=True, show_layer_names=True)

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Set integer ticks on the x-axis
        epochs = range(1, len(history.history['loss']) + 1)  # Integer epoch numbers
        plt.xticks(ticks=epochs)  # Set the ticks to integer epoch numbers

        plt.legend()
        plt.savefig(fname=f'./out/classification_loss_graph_{i}_mode{self.mode}_{self.predicted_measure}.png', format='png')
        plt.show()
        plt.close()

        self.y_prob = model.predict(self.X_test_scaled).flatten()

        if self.y_prob is not None:
            auc = roc_auc_score(self.y_test, self.y_prob)
            print(f"AUC-ROC: {auc:.4f}")
            auc_pr = average_precision_score(self.y_test, self.y_prob)
            print(f"AUC-PR: {auc_pr:.4f}")
        else:
            auc = np.nan
            auc_pr = np.nan

        self.y_pred = (self.y_prob >= threshold).astype(int)
        print(classification_report(self.y_test, self.y_pred))

        # explain features importance
        # X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
        # explainer = shap.Explainer(model, X_test_scaled_with_names)
        # shap_values = explainer(X_test_scaled_with_names)
        # joblib.dump(explainer,
        #             f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/explainer_{i}_mode{self.mode}_{self.predicted_measure}.pkl')
        # joblib.dump(shap_values,
        #             f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/shap_values__{i}_mode{self.mode}_{self.predicted_measure}.pkl')

        # Create a DataFrame
        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred,
            'probabilities': self.y_prob
        })

        # Save the DataFrame to a CSV file
        df_res.to_csv(f'./out/prediction_DL_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Plot Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: 0', 'Pred: 1'],
                    yticklabels=['True: 0', 'True: 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(fname=f'./out/confusion_matrix.png', format='png')
        plt.show()

        # Plot precision-recall curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_prob)
        target_thresholds = [0.3, 0.4, 0.5, 0.52, 0.55, 0.57, 0.6, 0.7]
        plt.plot(recall, precision, color='b', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')

        for target_threshold in target_thresholds:
            threshold_idx = (abs(thresholds - target_threshold)).argmin()
            threshold_recall = recall[threshold_idx]
            threshold_precision = precision[threshold_idx]
            plt.scatter(threshold_recall, threshold_precision, label=f'Threshold {target_threshold}', s=100, marker='x')

        plt.legend()
        plt.savefig(fname=f'./out/precision_recall.png', format='png')
        plt.show()

        roc_auc = roc_auc_score(self.y_test, self.y_prob)
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(fname=f'./out/ROC_curve.png', format='png')
        plt.show()

        mask = self.file_codes_test.str.contains("concat|_alt", regex=True)
        self.y_pred_filtered = self.y_pred[~mask]
        self.y_prob_filtered = self.y_prob[~mask]
        self.y_test_filtered = self.y_test[~mask]

        # Confusion Matrix #2
        cm = confusion_matrix(self.y_test_filtered, self.y_pred_filtered)
        print("Confusion Matrix:")
        print(cm)

        # Plot Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: 0', 'Pred: 1'],
                    yticklabels=['True: 0', 'True: 1'])
        plt.title('Confusion Matrix2')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(fname=f'./out/confusion_matrix2.png', format='png')
        plt.show()

        return auc