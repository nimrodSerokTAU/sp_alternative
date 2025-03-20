import itertools
import random

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
from tensorflow.keras.layers import Dense, Lambda, Dropout, LeakyReLU, Activation, BatchNormalization, Input, ELU, Attention, Reshape, Embedding, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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
from sklearn.utils import shuffle
from tensorflow.keras.metrics import AUC, Precision

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

class MSA_PairwiseBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, features, true_labels, true_msa_ids, train_msa_ids, val_msa_ids, msa_files, batch_size=32, validation_split=0.2, is_validation=False):
        self.features = features
        self.true_labels = true_labels
        self.msa_ids = true_msa_ids
        self.unique_msa_ids = np.unique(true_msa_ids)[
            np.unique(true_msa_ids) != "AATF"]  # TODO remove AATF from features file
        self.val_msa_ids = val_msa_ids
        self.train_msa_ids = train_msa_ids
        self.msa_files = msa_files
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.is_validation = is_validation


        # split_idx = int(len(self.unique_msa_ids) * (1 - self.validation_split))

        if not is_validation:
            # self.train_msa_ids = self.unique_msa_ids[:split_idx]
            self.features = self.features[np.isin(self.msa_ids, self.train_msa_ids)]
            self.true_labels = self.true_labels[np.isin(self.msa_ids, self.train_msa_ids)]
            self.msa_files = self.msa_files[np.isin(self.msa_ids, self.train_msa_ids)]
            self.msa_ids = self.msa_ids[np.isin(self.msa_ids, self.train_msa_ids)]

        else:
            # self.val_msa_ids = self.unique_msa_ids[split_idx:]
            self.features = self.features[np.isin(self.msa_ids, self.val_msa_ids)]
            self.true_labels = self.true_labels[np.isin(self.msa_ids, self.val_msa_ids)]
            self.msa_files = self.msa_files[np.isin(self.msa_ids, self.val_msa_ids)]
            self.msa_ids = self.msa_ids[np.isin(self.msa_ids, self.val_msa_ids)]

        # self.unique_msa_ids = np.unique(self.msa_ids)
        self.unique_msa_ids = np.unique(self.msa_ids)[
            np.unique(self.msa_ids) != "AATF"]
        self.pairs = self._create_pairs()
        self.batches = self._precompute_batches()
        df = pd.DataFrame(self.assert_,
                          columns=["Code1", "Index_1", "Index_2", "Label", "True_Label_1", "True_Label_2", "MSA_File_1",
                                   "MSA_File_2"])

        # Save the DataFrame to a CSV file
        df.to_csv("/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/pairs.csv", index=False)

    def _precompute_batches(self):
        batches = []
        num_full_batches = len(self.pairs) // self.batch_size

        for index in range(num_full_batches):
            batch_pairs = self.pairs[index * self.batch_size: (index + 1) * self.batch_size]
            batch_features_1 = []
            batch_features_2 = []
            batch_labels = []
            for i, j, label in batch_pairs:
                batch_features_1.append(self.features[i])
                batch_features_2.append(self.features[j])
                batch_labels.append(label)
            batch_features_1 = np.array(batch_features_1)
            batch_features_2 = np.array(batch_features_2)
            batch_labels = np.array(batch_labels)
            batch_features_1 = tf.convert_to_tensor(batch_features_1, dtype=tf.float32) #TODO do I really need to convert it to tensors? np array should be fine too...
            batch_features_2 = tf.convert_to_tensor(batch_features_2, dtype=tf.float32)
            batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
            batches.append(((batch_features_1, batch_features_2), batch_labels))
        np.random.shuffle(batches)
        return batches

# create pairs for 150 initial MSAs only
    def _create_pairs(self):
        pairs = []
        self.assert_ = []
        for msa_id in self.unique_msa_ids:
            try:
                # condition_msa_id = (self.msa_ids == msa_id)
                # condition_concat = self.msa_files.str.contains('concat')
                # combined_condition = condition_msa_id & condition_concat
                # idx = np.where(combined_condition)[0]
                # if len(idx) >= 40:
                #     random_indices = np.random.choice(idx, size=40, replace=False)
                # else:
                #     random_indices = idx
                # for i in range(len(random_indices)-1):
                #     for j in range(i+1, len(random_indices)):
                #         dist_i = self.true_labels.iloc[random_indices[i]]
                #         dist_j = self.true_labels.iloc[random_indices[j]]
                #         if dist_i < dist_j:
                #             pairs.append((random_indices[i], random_indices[j], 0))  # i is closer (first sample is closer)
                #         else:
                #             pairs.append((random_indices[i], random_indices[j], 1))  # j is closer or equal (second sample is closer)

                idx = np.where(self.msa_ids == msa_id)[0]
                # TEMP_features, TEMP_labels = self.features[idx], self.true_labels[idx]
                # if len(idx) >= 100:
                #     random_indices = np.random.choice(idx, size=100, replace=False)
                # else:
                #     random_indices = idx
                all_pairs = list(itertools.combinations(idx, 2))
                sampled_pairs = random.sample(all_pairs, 10000)
                for pair in sampled_pairs:
                    dist_i = self.true_labels.iloc[pair[0]]
                    dist_j = self.true_labels.iloc[pair[1]]
                    if dist_i < dist_j:
                        pairs.append((pair[0], pair[1], 0))  # i is closer (first sample is closer)
                    else:
                        pairs.append(
                            (pair[0], pair[1], 1))  # j is closer or equal (second sample is closer)
                        self.assert_.append((msa_id, pair[0], pair[1], 1, self.true_labels.iloc[pair[0]], self.true_labels.iloc[pair[1]], self.msa_files.iloc[pair[0]], self.msa_files.iloc[pair[1]]))
                # for i in range(len(random_indices)-1):
                #     for j in range(i+1, len(random_indices)):
                #         dist_i = self.true_labels.iloc[random_indices[i]]
                #         dist_j = self.true_labels.iloc[random_indices[j]]
                #         if dist_i < dist_j:
                #             pairs.append((random_indices[i], random_indices[j], 0))  # i is closer (first sample is closer)
                #         else:
                #             pairs.append((random_indices[i], random_indices[j], 1))  # j is closer or equal (second sample is closer)
            except Exception as e:
                # print(f"Exception1 {msa_id}, {random_indices[i]}, {random_indices[j]}: {e}\n")
                print(f"Exception1 {msa_id}: {e}\n")
        np.random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def on_epoch_end(self):
        if not self.is_validation:
            self.pairs = shuffle(self.pairs)
            self.batches = self._precompute_batches()
        np.random.shuffle(self.batches) # on validation we don't need to compute batches, just shuffle the validation batches created for 1st epoch

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


        df = pd.read_csv(self.features_file)
        # to make sure that all dataset codes are read as strings and not integers
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
            self.X = df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score','normalised_sop_score']]
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
            self.X_train = self.train_df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score','normalised_sop_score']]
            self.X_test = self.test_df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score','normalised_sop_score']]

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

        # writing train set into csv
        x_train_scaled_to_save = pd.DataFrame(self.X_train_scaled)
        x_train_scaled_to_save.columns = self.X_train.columns
        x_train_scaled_to_save['code'] = self.file_codes_train.reset_index(drop=True)
        x_train_scaled_to_save['code1'] = self.main_codes_train.reset_index(drop=True)
        x_train_scaled_to_save['class_label'] = self.y_train.reset_index(drop=True)
        # x_train_scaled_to_save['class_label_test'] = ...
        x_train_scaled_to_save.to_csv(
            '/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/train_scaled.csv', index=False)
        self.train_df.to_csv('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/train_unscaled.csv',
                             index=False)

        # writing test set into csv
        x_test_scaled_to_save = pd.DataFrame(self.X_test_scaled)
        x_test_scaled_to_save.columns = self.X_test.columns
        x_test_scaled_to_save['code'] = self.file_codes_test.reset_index(drop=True)
        x_test_scaled_to_save['code1'] = self.main_codes_test.reset_index(drop=True)
        x_test_scaled_to_save['class_label'] = self.y_test.reset_index(drop=True)
        # x_test_scaled_to_save['class_label_test'] = ...
        x_test_scaled_to_save.to_csv('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/test_scaled.csv',
                                     index=False)
        self.test_df.to_csv('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/test_unscaled.csv',
                            index=False)

    def deep_learning(self, epochs=50, batch_size=16, validation_split=0.2, verbose=1, learning_rate=0.01, dropout_rate=0.2, l1=1e-5, l2=1e-4, i=0, undersampling = False, threshold=0.50):
        history = None
        tf.config.set_visible_devices([], 'GPU')
        # tf.config.experimental_run_functions_eagerly(True)
        # @tf.function
        def _get_true_label(i, j):
            dist_i = self.y_test[i]
            dist_j = self.y_test[j]
            if dist_i < dist_j:
                label = 0
            else:
                label = 1
            return label

        def _get_test_pairs():
            pairs = []
            unique_msa_ids = np.unique(self.main_codes_test)
            try:
                for msa_id in unique_msa_ids:
                    idx = np.where(self.main_codes_test == msa_id)[0]
                    if len(idx) >= 50:
                        random_indices = np.random.choice(idx, size=50, replace=False)
                    else:
                        random_indices = idx
                    for i in range(len(random_indices) - 1):
                        for j in range(i + 1, len(random_indices)):
                            pairs.append((random_indices[i], random_indices[j]))
            except Exception as e:
                print(f"Exception {msa_id}, {random_indices[i]}, {random_indices[j]}: {e}\n")
            return pairs

        # def contrastive_loss(y_true, y_pred, margin=1.0):
        #     y_true = tf.cast(y_true, tf.float32)
        #     return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

        def focal_loss(gamma=2., alpha=0.25):
            """
            Focal Loss for binary classification.
            Parameters:
                gamma (float): Focusing parameter, typically set to 2.
                alpha (float): Balancing factor to adjust class weights (typically 0.25 for imbalanced datasets).
            Returns:
                A loss function that can be used in model compilation.
            """

            def loss(y_true, y_pred):
                y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
                cross_entropy = -y_true * K.log(y_pred)
                loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
                return K.mean(loss, axis=-1)

            return loss

        def inverse_focal_loss(alpha=0.25, gamma=2.0):
            def loss(y_true, y_pred):
                y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
                cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred) #cross-netropy changed in case if the minority class is overrepresented
                loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
                return K.mean(loss, axis=-1)

            return loss

        # def weighted_binary_crossentropy(y_true, y_pred, weight=1):
        #     return K.mean(weight * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred), axis=-1)

        def build_siamese_model(input_dim):
            input_1 = Input(shape=(input_dim,))
            input_2 = Input(shape=(input_dim,))

            # Shared layers for both inputs (same weights for both)
            shared = Sequential([
                Dense(64, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
                LeakyReLU(alpha=0.1),
                # Activation('swish'),
                Dropout(dropout_rate),
                BatchNormalization(),

                Dense(16, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
                # LeakyReLU(alpha=0.1),
                Activation('swish'),
                Dropout(dropout_rate),
                BatchNormalization(),

                # Dense(32, kernel_regularizer=regularizers.l2(l2)),
                # LeakyReLU(alpha=0.1),
                # Dropout(dropout_rate),
                # BatchNormalization(),
                #
                # Dense(32, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
                # LeakyReLU(alpha=0.1),
                Activation('swish'),
                # Dropout(dropout_rate),
                BatchNormalization()
            ])
            x1 = shared(input_1)
            x2 = shared(input_2)

            distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([x1, x2]) #abs difference between two embeddings
            # def euclidean_distance(vectors):
            #     x, y = vectors
            #     return K.sqrt(K.sum(K.square(x - y), axis=-1))
            #
            # distance = Lambda(euclidean_distance)([x1, x2])

            # def l1_distance(vectors):
            #     x, y = vectors
            #     return K.abs(x - y)
            #
            # distance = Lambda(l1_distance)([x1, x2])

            output = Dense(1, activation='sigmoid')(distance) # layer with sigmoid activation to get a single value between 0 a

            model = Model(inputs=[input_1, input_2], outputs=output)
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision()])
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision()])
            # model.compile(optimizer='adam', loss=inverse_focal_loss(gamma=2., alpha=0.25), metrics=['accuracy', AUC(), Precision()])
            # model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy, metrics=['accuracy', AUC(), Precision()])

            return model

        unique_train_codes = self.main_codes_train.unique()
        train_msa_ids, val_msa_ids = train_test_split(unique_train_codes, test_size=0.2)
        print(f"the training set is: {train_msa_ids} \n")
        print(f"the validation set is: {val_msa_ids} \n")

        # train_generator = MSA_PairwiseBatchGenerator(self.X_train_scaled, self.y_train, batch_size=batch_size)
        train_generator = MSA_PairwiseBatchGenerator(features=self.X_train_scaled, true_labels=self.y_train, true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids, val_msa_ids=val_msa_ids, msa_files=self.file_codes_train, batch_size=batch_size,
                                                     validation_split=validation_split, is_validation=False)
        val_generator = MSA_PairwiseBatchGenerator(features=self.X_train_scaled, true_labels=self.y_train, true_msa_ids=self.main_codes_train, train_msa_ids=train_msa_ids, val_msa_ids=val_msa_ids, msa_files=self.file_codes_train, batch_size=batch_size,
                                                   validation_split=validation_split, is_validation=True)

        # Callback 1: early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-5)
        # Callback 2: learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',  # to monitor
            patience=3,  # number of epochs with no improvement before reducing the learning rate
            verbose=1,
            factor=0.5,  # factor by which the learning rate will be reduced
            min_lr=1e-6  # lower bound on the learning rate
        )

        checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [early_stopping, lr_scheduler, checkpoint]

        # Build the Siamese model
        input_dim = self.X_train_scaled.shape[1]  # Number of features
        # with tf.device('/CPU:0'):
        model = build_siamese_model(input_dim)

        # Train the model
        history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=verbose, callbacks=callbacks)

        # Plot training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Evaluate on test data
        # test_pairs = [(i, j) for i in range(len(self.X_test_scaled)) for j in range(i + 1, len(self.X_test_scaled))]
        test_pairs = _get_test_pairs()
        y_true, y_pred = [], []
        for i, j in test_pairs:
            features1 = self.X_test_scaled[i:i + 1]
            features2 = self.X_test_scaled[j:j + 1]
            score = model.predict((features1, features2))  # Predict similarity
            predicted_label = 1 if score > threshold else 0
            true_label = _get_true_label(i, j)

            y_true.append(true_label)
            y_pred.append(predicted_label)

        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"Test Accuracy: {accuracy}")

        return accuracy