import math

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
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

class BatchGenerator(Sequence):
    def __init__(self, features, true_labels, true_msa_ids, train_msa_ids, val_msa_ids, batch_size, validation_split=0.2, is_validation=False, repeats=1, mixed_portion=0.3):
        self.features = features
        self.true_labels = np.asarray(true_labels)
        self.msa_ids = true_msa_ids  # TRUE MSA IDs (categorical)
        self.batch_size = batch_size
        self.unique_msa_ids = np.unique(true_msa_ids)  # Get unique TRUE MSA IDs
        self.validation_split = validation_split
        self.is_validation = is_validation
        self.val_msa_ids = val_msa_ids
        self.train_msa_ids = train_msa_ids
        self.repeats = repeats
        self.mixed_portion = mixed_portion

        # np.random.shuffle(self.unique_msa_ids) #can't do that as the order will be different in training and in validation

        # split_idx = int(len(self.unique_msa_ids) * (1 - self.validation_split))

        if self.is_validation:
            # self.val_msa_ids = self.unique_msa_ids[split_idx:]
            self.features = self.features[np.isin(self.msa_ids, self.val_msa_ids)]
            self.true_labels = self.true_labels[np.isin(self.msa_ids, self.val_msa_ids)]
            self.msa_ids = self.msa_ids[np.isin(self.msa_ids, self.val_msa_ids)]
        else:
            # self.train_msa_ids = self.unique_msa_ids[:split_idx]
            self.features = self.features[np.isin(self.msa_ids, self.train_msa_ids)]
            self.true_labels = self.true_labels[np.isin(self.msa_ids, self.train_msa_ids)]
            self.msa_ids = self.msa_ids[np.isin(self.msa_ids, self.train_msa_ids)]

        self.unique_msa_ids = np.unique(self.msa_ids)
        self.batches = self._precompute_batches()

    def _precompute_batches(self):
        batches = []
        batches_mix = []
        remaining_samples_set = []
        # leaving_out = 5

        for msa_id in self.unique_msa_ids:
            try:
                for k in range(self.repeats): #testing an option to produce different batch mixes
                    idx = np.where(self.msa_ids == msa_id)[0]
                    np.random.shuffle(idx)
                    num_samples = len(idx)
                    num_full_batches = num_samples // self.batch_size
                    remaining_samples = num_samples % self.batch_size
                    leaving_out = math.floor(self.mixed_portion * num_full_batches)

                    for i in range(num_full_batches - leaving_out): # I want to leave out some batches into the mix of remaining samples
                        batch_idx = idx[i * self.batch_size: (i + 1) * self.batch_size]
                        batches.append((self.features[batch_idx], self.true_labels[batch_idx]))

                    if remaining_samples > 0 or leaving_out > 0: # intermixed batches (consisting of the samples from different unique MSA IDs) to make sure that
                        remaining_samples_set.extend(idx[(num_full_batches - leaving_out) * self.batch_size:])
                    # np.random.shuffle(remaining_samples_set)
                    # np.random.shuffle(batches)

            except Exception as e:
                print(f"Exception {e}\n")

        remaining_samples_set = np.array(remaining_samples_set)
        np.random.shuffle(remaining_samples_set)

        for i in range(0, len(remaining_samples_set), self.batch_size):
            batch_idx = remaining_samples_set[i: i + self.batch_size]
            if len(batch_idx) == self.batch_size:  # Ensure full batch size
                batches_mix.append((self.features[batch_idx], self.true_labels[batch_idx]))

        final_batches = batches + batches_mix
        np.random.shuffle(final_batches)

        return final_batches

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
    def __init__(self, features_file: str, test_size: float, mode: int = 1, remove_correlated_features: bool = False, predicted_measure: Literal['msa_distance', 'tree_distance', 'class_label'] = 'msa_distance' ,i=0) -> None:
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
        df = pd.read_parquet(self.features_file, engine='pyarrow')
        df['code1'] = df['code1'].astype(str)

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
        df["class_label"] = np.where(df['dpos_dist_from_true'] <= 0.02, 0, 1)
        df["class_label2"] = np.where(df['dpos_dist_from_true'] <= 0.015, 0, np.where(df['dpos_dist_from_true'] <= 0.1, 1, 2))

        # df['aligner'] = df.apply(assign_aligner, axis=1)
        # df = df[df['aligner'] != 'true'] #removed true MSAs from the data
        # df = pd.get_dummies(df, columns=['aligner'], prefix='aligner') #added one-hot encoding for msa aligner program with the columns names of the form "aligner_mafft", "aligner_..."; the aligner column is automatically replaced/removed

        class_label_counts = df['class_label'].dropna().value_counts()
        print(class_label_counts)

        class_label2_counts_train = df['class_label2'].dropna().value_counts()
        print(class_label2_counts_train)

        df = df.dropna()

        unique_code1 = df['code1'].unique()
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
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2', 'normalised_sop_score'])
        if mode == 2:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2', 'sop_score', 'normalised_sop_score'])
        if mode == 3:
            self.X = df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]
        if mode == 4:
            self.X = df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'class_label2', 'code', 'code1',
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

        # Split the unique 'code1' into training and test sets
        train_code1, test_code1 = train_test_split(unique_code1, test_size=0.2)
        print(f"the training set is: {train_code1} \n")
        print(f"the testing set is: {test_code1} \n")

        # Create training and test DataFrames by filtering based on 'code1'
        self.train_df = df[df['code1'].isin(train_code1)]
        self.test_df = df[df['code1'].isin(test_code1)]
        # self.train_codes = train_code1
        # self.test_codes = test_code1

        class_label_counts_train = self.train_df['class_label'].dropna().value_counts()
        print(class_label_counts_train)

        class_label2_counts_train = self.train_df['class_label2'].dropna().value_counts()
        print(class_label2_counts_train)

        class_label_counts_test = self.test_df['class_label'].dropna().value_counts()
        print(class_label_counts_test)

        class_label2_counts_test = self.test_df['class_label2'].dropna().value_counts()
        print(class_label2_counts_test)


        # all features
        if mode == 1:
            self.X_train = self.train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty','class_label', 'class_label2', 'normalised_sop_score'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty','class_label', 'class_label2', 'normalised_sop_score'])
        if mode == 2:
            self.X_train = self.train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf','class_label', 'class_label2', 'code', 'code1', 'pypythia_msa_difficulty','sop_score', 'normalised_sop_score'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf','class_label', 'class_label2', 'code', 'code1', 'pypythia_msa_difficulty', 'sop_score', 'normalised_sop_score'])

        # 2 sop features
        if mode == 3:
            self.X_train = self.train_df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]
            self.X_test = self.test_df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]

        if mode == 4:
            self.X_train = self.train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'class_label2', 'code', 'code1',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'sp_score_gap_e_norm','single_char_count', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'class_label2', 'code',
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
        class_weights = compute_class_weight('balanced', classes=np.unique(self.train_df['class_label2']), y=self.train_df['class_label2'])
        self.weights = dict(enumerate(class_weights))
        print(self.weights)
        self.main_codes_test = self.test_df['code1']
        self.file_codes_test = self.test_df['code']

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

    def deep_learning(self, epochs=50, batch_size=16, validation_split=0.2, verbose=1, learning_rate=0.01, dropout_rate=0.2, l1=1e-5, l2=1e-5, i=0, undersampling = False, repeats = 1, mixed_portion = 0.3, top_k = 4, mse_weight=0, ranking_weight=50):
        history = None

        def weighted_mse(y_true, y_pred, weights):
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            weighted_loss = mse_loss * weights
            return weighted_loss

        def weighted_mse_loss(y_true, y_pred, factor):
            weights = K.exp(-y_true)  # Op1: Higher weights for lower scores
            # weights = 1/(1 + K.exp(factor*(y_true-0.5)))  # Op2: Higher weights for lower scores
            mse_loss = K.mean(weights * K.square(y_true - y_pred))  # Weighted MSE
            return mse_loss

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

        @tf.function
        def pairwise_rank_loss(y_true, y_pred, margin=0.0, top_k = 3):
            n = tf.shape(y_true)[0]

            y_true_flat = tf.reshape(y_true, [-1])
            _, top_k_indices = tf.math.top_k(-y_true_flat, k=top_k, sorted=True)  # Use negative to get smallest values
            mask = tf.reduce_any(tf.equal(tf.reshape(tf.range(n), [-1, 1]), tf.reshape(top_k_indices, [1, -1])), axis=1)

            i_indices = tf.reshape(tf.range(n), [-1, 1])
            j_indices = tf.reshape(tf.range(n), [1, -1])
            i_indices_flat = tf.reshape(i_indices, [-1])
            j_indices_flat = tf.reshape(j_indices, [-1])
            y_true_i = tf.gather(y_true, i_indices_flat)
            y_true_j = tf.gather(y_true, j_indices_flat)
            y_pred_i = tf.gather(y_pred, i_indices_flat)
            y_pred_j = tf.gather(y_pred, j_indices_flat)

            y_true_diff = tf.cast(y_true_i < y_true_j, tf.float32)
            pairwise_loss = tf.maximum(0.0, y_pred_i - y_pred_j + margin)

            loss = tf.reduce_sum(pairwise_loss * y_true_diff * tf.cast(tf.reshape(mask, [-1, 1]), tf.float32))

            return loss

        # Combine MSE loss with rank-based loss
        def mse_with_rank_loss(y_true, y_pred, top_k=3, mse_weight=1, ranking_weight=0.3):

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
        def min_score_penalty_loss(y_true, y_pred, mse_weight=1.0, min_penalty_weight=50.0):
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

            # fourth hidden
            model.add(
                Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l2(l2=l2)))
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

            model.add(Dense(1, activation='sigmoid'))  #limits output to 0 to 1 range

            optimizer = Adam(learning_rate=learning_rate)
            # optimizer = RMSprop(learning_rate=learning_rate)

            # model.compile(optimizer=optimizer, loss='mean_squared_error')
            # model.compile(optimizer=optimizer, loss='mean_absolute_error')
            # model.compile(optimizer=optimizer, loss=mse_with_rank_loss)
            # model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: weighted_mse_loss(y_true, y_pred, factor=7))
            # model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: weighted_mse(y_true, y_pred, weights))
            model.compile(optimizer=optimizer, loss = lambda y_true, y_pred: mse_with_rank_loss(y_true, y_pred, top_k=top_k, mse_weight=mse_weight,
                                                        ranking_weight=ranking_weight))
            # model.compile(optimizer=optimizer,
            #               loss=lambda y_true, y_pred: min_score_penalty_loss(y_true, y_pred, mse_weight=1.0, min_penalty_weight=50.0))

            unique_train_codes = self.main_codes_train.unique()
            train_msa_ids, val_msa_ids = train_test_split(unique_train_codes, test_size=0.2)
            print(f"the training set is: {train_msa_ids} \n")
            print(f"the validation set is: {val_msa_ids} \n")
            batch_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train,
                                             true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids, val_msa_ids=val_msa_ids, batch_size=batch_size,
                                             validation_split=validation_split, is_validation=False, repeats=repeats, mixed_portion=mixed_portion)

            val_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train,
                                           true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids, val_msa_ids=val_msa_ids,
                                           batch_size=batch_size, validation_split=validation_split, is_validation=True, repeats=repeats, mixed_portion=mixed_portion)

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
            plt.savefig('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/summary_plot.png', dpi=300,
                        bbox_inches='tight')
            # plt.show()
            plt.close()

            shap.plots.waterfall(shap_values[0], max_display=40)
            plt.savefig('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/waterfall_0_plot.png', dpi=300,
                        bbox_inches='tight')
            # plt.show()
            plt.close()

            shap.force_plot(shap_values[0], X_test_subset[0], matplotlib=True, show=False)
            plt.savefig('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/force_0_plot.png')
            # plt.show()
            plt.close()

            shap.plots.bar(shap_values, max_display=40)
            plt.savefig('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/bar_plot.png', dpi=300,
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