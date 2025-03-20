import math
import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, classification_report, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from typing import Literal
from scipy.stats import pearsonr, gaussian_kde
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer, LabelEncoder, OneHotEncoder
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

def assign_aligner(row):
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

def check_file_type(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.parquet':
        return 'parquet'
    elif file_extension == '.csv':
        return 'csv'
    else:
        return 'unknown'

def assign_class_label(group):
    max_sop_row = group.loc[group['sop_score'].idxmax()]
    sop_dpos = max_sop_row['dpos_dist_from_true']
    group['class_label'] = (group['dpos_dist_from_true'] < sop_dpos).astype(int)
    return group

# def assign_class_label(group):
#     mask = group['code'].str.contains("concat|_alt", case=False, na=False)
#     group_without_extra = group[~mask]
#     percentile_15 = group_without_extra['dpos_dist_from_true'].quantile(0.15)
#     group['class_label'] = (group['dpos_dist_from_true'] < percentile_15).astype(int)
#     return group

def assign_class_label_test(group):
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

# from tensorflow.keras.layers import Layer
# class LowScoreMaskLayer(Layer):
#     def __init__(self, threshold=0.5, **kwargs):
#         super(LowScoreMaskLayer, self).__init__(**kwargs)
#         self.threshold = threshold
#     def call(self, inputs):
#         classification_output = inputs[0]  # Get classification output
#         # Generate the mask where classification output > threshold
#         low_score_mask = K.cast(K.greater(classification_output, self.threshold), dtype='float32')
#         return low_score_mask


from tensorflow.keras.layers import Layer
# class WeightedRegressionLoss(Layer):
#     def __init__(self, alpha=0.5, **kwargs):
#         super(WeightedRegressionLoss, self).__init__(**kwargs)
#         self.alpha = alpha
#
#     def call(self, inputs):
#         y_true, regression_output, classification_output = inputs
#
#         # Convert classification output to a weight (classification probability)
#         classification_weight = K.cast(classification_output, dtype=tf.float32)
#         classification_weight = classification_weight * self.alpha + (1 - self.alpha)
#
#         # Compute regression loss
#         regression_error = K.abs(y_true - regression_output)
#         weighted_error = regression_error * classification_weight
#
#         return K.mean(weighted_error)


from tensorflow.keras.losses import Loss
class WeightedRegressionLoss(Loss):
    def __init__(self, alpha=0.5, name="weighted_regression_loss"):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Retrieve classification output dynamically
        classification_output = self.get_classification_output(y_pred)

        # Compute classification weight (alpha-weighted probabilities)
        classification_weight = K.cast(classification_output, dtype=tf.float32)
        classification_weight = classification_weight * self.alpha + (1 - self.alpha)

        # Compute weighted MSE
        regression_error = K.square(y_true - y_pred)
        weighted_error = regression_error * classification_weight

        return K.mean(weighted_error)

    def get_classification_output(self, y_pred):
        """Tries to fetch classification output dynamically from the model during training."""
        for layer in y_pred._keras_history[0].model.layers:
            if layer.name == "classification_output":
                return layer.output
        raise ValueError("Classification output not found in the model.")

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config

class BatchGenerator(Sequence):
    def __init__(self, features, true_labels, true_class_labels, true_msa_ids, train_msa_ids, val_msa_ids, aligners, batch_size, validation_split=0.2, is_validation=False, repeats=1, mixed_portion=0.3, per_aligner=False, classification_task = False, features_w_names=np.nan):
        self.features = features
        self.true_labels = np.asarray(true_labels)
        self.true_class_labels = np.asarray(true_class_labels)
        self.msa_ids = true_msa_ids  # TRUE MSA IDs (categorical)
        self.batch_size = batch_size
        self.unique_msa_ids = np.unique(true_msa_ids)[np.unique(true_msa_ids) != "AATF"]  #TODO remove AATF from features file
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
        self.true_class_labels = self.true_class_labels[mask]
        self.features_w_names = self.features_w_names[mask]
        self.features_w_names = self.features_w_names.reset_index(drop=True)
        self.msa_ids = self.msa_ids[mask]
        self.unique_msa_ids = np.unique(self.msa_ids)
        self.batches = self._precompute_batches()

    def _split_idx_into_batches(self, idx):
        batches = []
        remaining_samples_set = []
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
            batches.append((self.features[batch_idx], (self.true_labels[batch_idx], self.true_class_labels[batch_idx])))

        if remaining_samples > 0 or leaving_out > 0: # intermixed batches (consisting of the samples from different unique MSA IDs) to make sure that
            remaining_samples_set.extend(idx[(num_full_batches - leaving_out) * self.batch_size:])
        np.random.shuffle(remaining_samples_set)
        np.random.shuffle(batches)
        return batches, remaining_samples_set

    def _split_sorted_idx_into_batches(self, idx):
        if len(idx) > 0:
            sorted_features = self.features_w_names.iloc[idx]
            sorted_features = sorted_features.sort_values(by='dpos_dist_from_true', ascending=True)
            sorted_indices = sorted_features.index.values #TODO - check that actual indices are reordered by sorting
            batches, remaining_samples_set = self._split_idx_into_batches(sorted_indices)
            return batches, remaining_samples_set

    def _precompute_batches(self):
        batches = []
        batches_mix = []
        remaining_samples_set = []

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
                batches_mix.append((self.features[batch_idx], (self.true_labels[batch_idx],self.true_class_labels[batch_idx])))
            # if len(batch_idx) == self.batch_size:  # Ensure full batch size
            #     batches_mix.append((self.features[batch_idx], self.true_labels[batch_idx]))

        final_batches = batches + batches_mix
        np.random.shuffle(final_batches)

        return final_batches

    def _precompute_true_labels(self, idx):
        if len(idx) > 0:
            features = self.features_w_names.iloc[idx]
            features = features.reset_index(drop=True)
            # percentile_10 = features['dpos_dist_from_true'].quantile(0.1)
            # labels = (features['dpos_dist_from_true'] <= percentile_10).astype(int)
            # labels = labels.to_numpy()

            max_sop_row = features.loc[features['sop_score'].idxmax()]
            sop_dpos = max_sop_row['dpos_dist_from_true']
            # print("Features shape:", features.shape)
            # print("Max SOP Row:", max_sop_row)
            # print("SoP dpos:", sop_dpos)
            labels = (features['dpos_dist_from_true'] < sop_dpos).astype(int)
            labels = labels.to_numpy()
        else:
            labels = np.nan
            print("empty batch\n")
        return labels
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def on_epoch_end(self):
        if not self.is_validation:
            self.batches = self._precompute_batches()
        np.random.shuffle(self.batches)

    def __iter__(self):
        for idx in range(len(self)):
            batch_features, batch_labels = self[idx]
            yield (batch_features, batch_labels)
class Regressor:
    '''
    features_file: file with all features and labels
    test_size: portion of the codes to be separated into a test set; all MSAs for that specific code would be on the same side of the train-test split
    mode: 1 is all features, 2 is all except SoP features, 3 is only 2 SoP features'''
    def __init__(self, features_file: str, test_size: float, mode: int = 1, remove_correlated_features: bool = False, predicted_measure: Literal['msa_distance', 'class_label'] = 'msa_distance', i=0) -> None:
        self.features_file = features_file
        self.test_size = test_size
        self.predicted_measure = predicted_measure
        self.mode = mode
        # self.num_estimators = n_estimators
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X, self.y, self.y_pred = None, None, None
        self.prediction = None
        self.main_codes_train = None # these are the codes we can use for batch generation
        self.file_codes_train = None
        self.main_codes_test = None
        self.file_codes_test = None
        # self.train_codes, self.test_codes = None, None


        # df = pd.read_csv(self.features_file)
        # df_extra = pd.read_csv("/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_extra900_features_260225.csv")
        # df = pd.concat([df, df_extra], ignore_index=True)
        # to make sure that all dataset codes are read as strings and not integers
        # df = pd.read_parquet(self.features_file, engine='pyarrow')
        file_type = check_file_type(self.features_file)
        if file_type == 'parquet':
            df = pd.read_parquet(self.features_file, engine='pyarrow')
        elif file_type =='csv':
            df = pd.read_csv(self.features_file)
        else:
            print(f"features file is of unknown format\n")

        df['code1'] = df['code1'].astype(str)
        df = df[df['code1'] != 'AATF'] #TODO delete

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

        # df = df[df['class_label'] == 1] #TODO delete: here I chose only the MSAs we produces with small dpos

        df.to_csv('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/features_w_label.csv', index=False)

        #filter BALIPHY ONLY
        # df = df[df['code'].str.contains('bali_phy|BALIPHY', case=False, na=False, regex=True)]
        # df = df[df['code'].str.contains('prank', case=False, na=False, regex=True)]
        # df = df[~df['code'].str.contains('prank|bali_phy|BALIPHY', case=False, na=False, regex=True)]

        # Check for missing values
        print("Missing values in each column:\n", df.isnull().sum())
        corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df['dpos_dist_from_true'])
        print(f"Pearson Correlation of Normalized SOP and dpos: {corr_coefficient1:.4f}\n", f"P-value of non-correlation: {p_value1:.6f}\n")
        corr_coefficient1, p_value1 = pearsonr(df['sop_score'], df['dpos_dist_from_true'])
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

        if self.predicted_measure == 'msa_distance':
            true_score_name = "dpos_dist_from_true"
        elif self.predicted_measure == 'tree_distance':
            true_score_name = 'normalized_rf'
        elif self.predicted_measure == 'class_label':
            true_score_name = 'class_label'

        self.y = df[true_score_name]

        # all features
        if mode == 1:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'normalised_sop_score', 'aligner'])
        if mode == 2:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label_test', 'sop_score', 'normalised_sop_score'])
        if mode == 3:
            # self.X = df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]
            self.X = df[['sop_score']]
        if mode == 4:
            self.X = df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'code', 'code1',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'sp_score_gap_e_norm','single_char_count', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len'])


        if remove_correlated_features:
            correlation_matrix = self.X.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
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
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty','class_label', 'normalised_sop_score', 'aligner'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty','class_label', 'normalised_sop_score', 'aligner'])
        if mode == 2:
            self.X_train = self.train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf','class_label', 'code', 'code1', 'pypythia_msa_difficulty','sop_score', 'normalised_sop_score', 'aligner'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf','class_label', 'code', 'code1', 'pypythia_msa_difficulty', 'sop_score', 'normalised_sop_score', 'aligner'])

        # 2 sop features
        if mode == 3:
            # self.X_train = self.train_df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]
            # self.X_test = self.test_df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]
            self.X_train = self.train_df[['sop_score']]
            self.X_test = self.test_df[['sop_score']]

        if mode == 4:
            self.X_train = self.train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'code', 'code1',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'sp_score_gap_e_norm','single_char_count', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'code',
                         'code1',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min', 'gaps_len_two',
                         'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
                         'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
                         'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
                         'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus',
                         'sp_score_gap_e_norm', 'sp_score_gap_e_norm', 'single_char_count', 'double_char_count',
                         'k_mer_10_max', 'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
                         'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90',
                         'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1',
                         'seq_min_len']
            )

        if remove_correlated_features:
            self.X_train = self.X_train.drop(columns=to_drop)
            self.X_test = self.X_test.drop(columns=to_drop)

        self.scaler = MinMaxScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train) #calculate scaling parameters (fit)
        self.X_test_scaled = self.scaler.transform(self.X_test) #use the same scaling parameters as in train scaling
        joblib.dump(self.scaler, f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/scaler_{i}_mode{self.mode}_{self.predicted_measure}.pkl')

        self.main_codes_train = self.train_df['code1']
        self.file_codes_train = self.train_df['code']
        self.aligners_train = self.train_df['aligner']
        class_weights = compute_class_weight('balanced', classes=np.unique(self.train_df['class_label']), y=self.train_df['class_label'])
        self.weights = dict(enumerate(class_weights))
        print(self.weights)
        self.main_codes_test = self.test_df['code1']
        self.file_codes_test = self.test_df['code']
        self.aligners_test = self.test_df['aligner']

        corr_coefficient1, p_value1 = pearsonr(self.test_df['normalised_sop_score'], self.test_df['dpos_dist_from_true'])
        print(f"Pearson Correlation of Normalized SOP and dpos in the TEST set: {corr_coefficient1:.4f}\n",
              f"P-value of non-correlation: {p_value1:.4f}\n")
        corr_coefficient1, p_value1 = pearsonr(self.test_df['sop_score'],
                                               self.test_df['dpos_dist_from_true'])
        print(f"Pearson Correlation of SOP and dpos in the TEST set: {corr_coefficient1:.4f}\n",
              f"P-value of non-correlation: {p_value1:.4f}\n")

        # Set train and test Labels
        self.y_train = self.train_df[true_score_name]
        self.y_test = self.test_df[true_score_name]
        # self.dpos_train = self.train_df['dpos_dist_from_true']
        self.class_label_train = self.train_df['class_label']
        self.class_label_test = self.test_df['class_label']

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
        x_train_scaled_to_save.columns = self.X_train.columns
        x_train_scaled_to_save['code'] = self.file_codes_train.reset_index(drop=True)
        x_train_scaled_to_save['code1'] = self.main_codes_train.reset_index(drop=True)
        # x_train_scaled_to_save['class_label'] = self.y_train.reset_index(drop=True)
        x_train_scaled_to_save['class_label'] = self.class_label_train.reset_index(drop=True)
        # x_train_scaled_to_save['class_label_test'] = ...
        x_train_scaled_to_save.to_csv(
            f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/train_scaled_{i}.csv', index=False)
        self.train_df.to_csv(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/train_unscaled_{i}.csv',
                             index=False)

        # writing test set into csv
        x_test_scaled_to_save = pd.DataFrame(self.X_test_scaled)
        x_test_scaled_to_save.columns = self.X_test.columns
        x_test_scaled_to_save['code'] = self.file_codes_test.reset_index(drop=True)
        x_test_scaled_to_save['code1'] = self.main_codes_test.reset_index(drop=True)
        x_test_scaled_to_save['class_label'] = self.y_test.reset_index(drop=True)
        # x_test_scaled_to_save['class_label_test'] = ...
        x_test_scaled_to_save.to_csv(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/test_scaled_{i}.csv',
                                     index=False)
        self.test_df.to_csv(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/test_unscaled_{i}.csv',
                            index=False)

    def deep_learning(self, epochs=50, batch_size=16, validation_split=0.2, verbose=1, learning_rate=0.01, dropout_rate=0.2, l1=1e-5, l2=1e-5, i=0, undersampling = False, repeats = 1, mixed_portion = 0.3, top_k = 4, mse_weight=0, ranking_weight=50, per_aligner=False):
        history = None
        tf.config.set_visible_devices([], 'GPU') #disable GPU in tensorflow

        def final_prediction(main_output, low_score_output, y_true, threshold=0.03):
            return tf.where(y_true < threshold, low_score_output, main_output)

        # def low_score_weighted_mse(y_true, y_pred):
        #     error = tf.abs(y_true - y_pred)
        #     weight = tf.maximum(1.0, 1.0 / (y_true + 1e-6))
        #     weighted_error = weight * error
        #     return tf.reduce_mean(weighted_error ** 2)

        def low_score_weighted_log_error(y_true, y_pred):
            error = tf.abs(y_true - y_pred)
            weight = tf.maximum(1.0, tf.math.log(y_true + 1e-6))  # Logarithmic weighting based on y_true
            weighted_error = weight * error
            return tf.reduce_mean(weighted_error ** 2)

        def low_score_weighted_exp_error(y_true, y_pred, exponent=6.0):
            error = tf.abs(y_true - y_pred)
            weight = tf.maximum(1.0, tf.exp(-y_true) + 1e-6)  # Exponentially decaying weight based on y_true
            weighted_error = weight * error
            return tf.reduce_mean(weighted_error ** exponent)

        def rank_loss(y_true, y_pred, top_k):
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

        def mse_with_rank_loss(y_true, y_pred, top_k=4, mse_weight=1, ranking_weight=50):

            mse_loss = K.mean(K.square(K.cast(y_true - y_pred, dtype=tf.float32)))  # MSE loss
            # mse_loss = tf.keras.losses.MSE(y_true, y_pred)
            top_k_rank_loss = rank_loss(y_true, y_pred, top_k)
            # rank_loss = pairwise_rank_loss(y_true, y_pred, margin=1.0, top_k=top_k)
            mse_weight = tf.cast(mse_weight, dtype=tf.float32)
            ranking_weight = tf.cast(ranking_weight, dtype=tf.float32)
            top_k_rank_loss = tf.cast(top_k_rank_loss, dtype=tf.float32)
            total_loss = mse_weight * mse_loss + ranking_weight * top_k_rank_loss

            return total_loss

        def weighted_binary_crossentropy(w0=1.0, w1=1.0):
            def loss(y_true, y_pred):
                loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                weights = y_true * w1 + (1 - y_true) * w0 #would it help me to make class1 5 times more important?
                return tf.reduce_mean(loss_fn(y_true, y_pred) * weights)

            return loss

        def weighted_regression_loss(alpha=0.5):
            def loss(y_true, y_pred):
                # Extract classification probability from y_pred (assuming y_pred contains both outputs)
                regression_output, classification_output = y_pred[:, 0], y_pred[:, 1]

                # Ensure classification_output is float
                classification_weight = K.cast(classification_output, dtype=tf.float32)

                # Apply weighting based on classification probability
                classification_weight = classification_weight * alpha + (1 - alpha)

                # Compute regression loss
                regression_error = K.abs(y_true - regression_output)
                weighted_error = regression_error * classification_weight

                return K.mean(weighted_error)

            return loss

        # def regression_loss_with_classification_weight(y_true, y_pred, classification_output):
        #     # classification_output = K.get_value(model.get_layer('classification_output').output)
        #     return weighted_regression_loss(y_true, y_pred, classification_output, alpha=0.5)


        def classification_first_stage_loss(y_true, y_pred, classification_output):
            '''masks all samples that were classified as "class 0" / probability lower than 0.5 threshold/
            only calculates the error for the samples classified as "class 1"'''

            tf.compat.v1.enable_eager_execution()
            @tf.function
            def print_func(y_true, y_pred, regression_error, filtered_error):
                tf.print("low_score_mask:", low_score_mask)
                tf.print("low_score_mask shape:", K.int_shape(low_score_mask))
                tf.print("y_true:", y_true)
                tf.print("y_true shape:", K.int_shape(y_true))
                tf.print("y_pred:", y_pred)
                tf.print("y_pred shape:", K.int_shape(y_pred))
                tf.print("regression_error:", regression_error)
                tf.print("regression_error shape:", K.int_shape(regression_error))
                tf.print("filtered_error:", filtered_error)
                tf.print("filtered_error shape:", K.int_shape(filtered_error))
                return y_true, y_pred, regression_error, filtered_error

            low_score_mask_layer = LowScoreMaskLayer(threshold=0.5)
            low_score_mask = low_score_mask_layer(
                [classification_output])

            y_true = K.cast(y_true, dtype=tf.float32)
            y_pred = K.cast(y_pred, dtype=tf.float32)
            regression_error = K.abs(y_true - y_pred)
            # low_score_mask = K.reshape(low_score_mask, K.shape(regression_error))
            # regression_error = K.abs(K.cast(y_true - y_pred, dtype=tf.float32))
            filtered_error = regression_error * low_score_mask
            print_func(y_true, y_pred, regression_error, filtered_error)
            # filtered_error = K.cast(regression_error * low_score_mask, dtype=tf.float32)
            # mean_error = tf.reduce_mean(K.cast(filtered_error,dtype=tf.float32))
            # mean_error = K.mean(K.cast(filtered_error,dtype=tf.float32))
            mean_error = K.mean(filtered_error)
            return mean_error

        def create_model(input_shape, mse_weight, ranking_weight, l2, dropout_rate,
                         learning_rate):

            input_layer = Input(shape=(input_shape,))

            # Shared layers
            x = Dense(128, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2))(
                input_layer)
            x = LeakyReLU(negative_slope=0.01)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            x = Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2))(x)
            x = LeakyReLU(negative_slope=0.01)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            x = Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2))(x)
            x = LeakyReLU(negative_slope=0.01)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            # x = Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2))(x)
            # x = LeakyReLU(negative_slope=0.01)(x)
            # x = BatchNormalization()(x)
            # x = Dropout(dropout_rate)(x)

            # Main regression (mse) head
            regression_head = Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2))(x)
            regression_head = LeakyReLU(negative_slope=0.01)(regression_head)
            regression_head = BatchNormalization()(regression_head)
            regression_head = Dropout(dropout_rate)(regression_head)
            regression_output = Dense(1, activation='sigmoid', name='regression_output')(regression_head)

            # Classification head
            classification_head = Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2))(x)
            classification_head = LeakyReLU(negative_slope=0.01)(classification_head)
            classification_head = BatchNormalization()(classification_head)
            classification_head = Dropout(dropout_rate)(classification_head)
            classification_output = Dense(1, activation='sigmoid', name='classification_output')(classification_head)

            weighted_loss_output = WeightedRegressionLoss(alpha=0.5, name='weighted_regression_loss')([input_layer, regression_output, classification_output])


            # Define the model
            model = Model(inputs=input_layer, outputs=[regression_output, classification_output])
            # model = Model(inputs=input_layer, outputs=[regression_output, classification_output, weighted_loss_output])

            class_weights = compute_class_weight('balanced', classes=np.unique(self.class_label_train), y=self.class_label_train)
            class_weight_dict = dict(enumerate(class_weights))
            print(class_weight_dict)

            # Compile the model with custom loss functions
            # model.compile(optimizer=Adam(learning_rate=learning_rate),
            #               loss={'main_output': 'mse', 'low_score_output': low_score_weighted_exp_error},
            #               loss_weights={'main_output': mse_weight,
            #                             'low_score_output': ranking_weight})
            # model.compile(optimizer=Adam(learning_rate=learning_rate),
            #               loss={'regression_output': mse_with_rank_loss, 'classification_output': weighted_binary_crossentropy(w0=class_weight_dict[0], w1=5*class_weight_dict[1])},
            #               loss_weights={'regression_output': mse_weight,
            #                             'classification_output': ranking_weight})

            # model.compile(optimizer=Adam(learning_rate=learning_rate),
            #               loss={'regression_output': lambda y_true, y_pred: y_pred,
            #                     'classification_output': weighted_binary_crossentropy(w0=class_weight_dict[0],
            #                                                                           w1=5 * class_weight_dict[1]),
            #                     'weighted_regression_loss': lambda y_true, y_pred: y_pred}, loss_weights={'regression_output': 1,
            #                             'classification_output': 1, 'weighted_regression_loss': 1})

            model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss={'regression_output': WeightedRegressionLoss,
                                'classification_output': weighted_binary_crossentropy(w0=class_weight_dict[0],
                                                                                      w1=5 * class_weight_dict[1])},
                          loss_weights={'regression_output': 1,
                                        'classification_output': 1})

            # model.compile(optimizer=Adam(learning_rate=learning_rate),
            #               loss={'regression_output': lambda y_true, y_pred: classification_first_stage_loss(y_true, y_pred, classification_output),
            #                     'classification_output': weighted_binary_crossentropy(w0=class_weight_dict[0],
            #                                                                           w1=5 * class_weight_dict[1])})

            return model

        model = create_model(input_shape=self.X_train_scaled.shape[1], mse_weight=mse_weight, ranking_weight=ranking_weight, l2=l2, dropout_rate=dropout_rate,
                         learning_rate=learning_rate)

        unique_train_codes = self.main_codes_train.unique()
        train_msa_ids, val_msa_ids = train_test_split(unique_train_codes, test_size=0.2)
        print(f"the training set is: {train_msa_ids} \n")
        print(f"the validation set is: {val_msa_ids} \n")
        # x_train_scaled_with_names = pd.DataFrame(self.X_train_scaled)
        # x_train_scaled_with_names.columns = self.X_train.columns
        batch_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train, true_class_labels = self.class_label_train,
                                         true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids, val_msa_ids=val_msa_ids, aligners =self.aligners_train, batch_size=batch_size,
                                         validation_split=validation_split, is_validation=False, repeats=repeats, mixed_portion=mixed_portion, per_aligner=per_aligner, features_w_names=self.train_df)

        val_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train, true_class_labels = self.class_label_train,
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

        # history = model.fit(self.X_train_scaled, [self.y_train, self.y_train],
        #   epochs=epochs,
        #   batch_size=batch_size,
        #   validation_split=validation_split,
        #   verbose=verbose,
        #   callbacks=callbacks)

        # history = model.fit(self.X_train_scaled, [self.y_train, self.class_label_train],
        #   epochs=epochs,
        #   batch_size=batch_size,
        #   validation_split=validation_split,
        #   verbose=verbose,
        #   callbacks=callbacks)

        history = model.fit(batch_generator,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=val_generator,
          verbose=verbose,
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

        loss = model.evaluate(self.X_test_scaled, [self.y_test, self.class_label_test])
        print(f"Test Loss: {loss}")

        self.y_pred, self.class_prob = model.predict(self.X_test_scaled)

        # Get predictions from both heads
        # main_predictions, low_score_predictions = model.predict(self.X_test_scaled)


        self.y_pred = np.ravel(self.y_pred)  # flatten
        self.y_pred = self.y_pred.astype('float64')
        self.class_prob = np.ravel(self.class_prob)  # flatten
        self.class_prob = self.class_prob.astype('float64')

        self.class_pred = (self.class_prob >= 0.55).astype(int)


        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred,
            'predicted_class_prob': self.class_prob,
            'predicted_class_pred': self.class_pred,
        })

        df_res.to_csv(f'./out/prediction_DL_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error 1: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")

        print(classification_report(self.class_label_test, self.class_pred))

        return mse


    def plot_results(self, model_name: Literal["svr", "rf", "knn-r", "gbr", "dl"], mse: float, i: int) -> None:
        # FIRST PREDICTION
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


#class PREDICTION
        # Confusion Matrix
        cm = confusion_matrix(self.class_label_test, self.class_pred)
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
        precision, recall, thresholds = precision_recall_curve(self.class_label_test, self.class_pred)
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

        roc_auc = roc_auc_score(self.class_label_test, self.class_prob)
        fpr, tpr, _ = roc_curve(self.class_label_test, self.class_prob)

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
