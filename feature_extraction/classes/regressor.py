import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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

import pydot
import tensorflow as tf
# import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation, BatchNormalization, Input, ELU, Attention, Reshape
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from feature_extraction.classes.attention_layer import AttentionLayer

def assign_aligner(row):
    code = row['code'].lower()
    no_aligners = ['muscle', 'prank', '_true.fas', 'true_tree.txt', 'bali_phy', 'baliphy', 'original']

    if not any(sub in code for sub in no_aligners):
        return 'mafft'
    if 'muscle' in code:
        return 'muscle'
    elif 'prank' in code:
        return 'prank'
    elif 'bali_phy' in code or 'baliphy' in code:
        return 'baliphy'

    return 'true'

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
        self.main_codes_train = None
        self.file_codes_train = None
        self.main_codes_test = None
        self.file_codes_test = None


        df = pd.read_csv(self.features_file)
        # to make sure that all dataset codes are read as strings and not integers
        df['code1'] = df['code1'].astype(str)

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
        # df = df[df['aligner'] != 'true']
        # df = pd.get_dummies(df, columns=['aligner'], prefix='aligner')


        class_label_counts = df['class_label'].dropna().value_counts()
        print(class_label_counts)

        class_label2_counts_train = df['class_label2'].dropna().value_counts()
        print(class_label2_counts_train)

        # Handle missing values (if any)
        # Example: Filling missing values with the mean (for numerical columns)
        # df['orig_tree_ll'] = df['orig_tree_ll'].fillna(df['orig_tree_ll'].mean())
        # remove NaNs?
        df = df.dropna()

        if self.predicted_measure == 'msa_distance':
            true_score_name = "dpos_dist_from_true"
            # true_score_name = "MEAN_RES_PAIR_SCORE"
            # true_score_name = "MEAN_COL_SCORE"
            # df['MEAN_RES_PAIR_SCORE'] = df['MEAN_RES_PAIR_SCORE'].astype(float)
            # df['MEAN_COL_SCORE'] = df['MEAN_COL_SCORE'].astype(float)
        elif self.predicted_measure == 'tree_distance':
            # true_score_name = "rf_from_true"
            true_score_name = 'normalized_rf'
        elif self.predicted_measure == 'class_label':
            true_score_name = 'class_label'

        self.y = df[true_score_name]


        # all features
        if mode == 1:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2', 'normalised_sop_score'])
            # self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2', 'entropy_mean', 'entropy_pct_75', 'msa_len', 'seq_max_len', 'seq_min_len', 'total_gaps', 'gaps_len_one', 'gaps_len_two', 'gaps_len_three_plus', 'num_unique_gaps', 'gaps_1seq_len1', 'gaps_1seq_len2', 'gaps_1seq_len3plus', 'num_cols_no_gaps', 'num_cols_1_gap', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'sp_score_subs_norm', 'sp_score_gap_e_norm', 'sp_match_ratio', 'sp_missmatch_ratio', 'double_char_count', 'bl_sum', 'kurtosis_bl', 'bl_std', 'bl_max', 'k_mer_10_max', 'k_mer_10_var', 'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_norm', 'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_norm', 'number_of_gap_segments', 'number_of_mismatches', 'henikoff_with_gaps', 'henikoff_without_gaps', 'clustal_differential_sum'])
        # all features except 2 features of SoP
        # if mode == 2:
        #     self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'sop_score', 'normalised_sop_score', 'sp_score_subs_norm', 'sp_score_gap_e_norm',
        #     'sp_match_ratio', 'sp_missmatch_ratio'])
        if mode == 2:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2', 'sop_score', 'normalised_sop_score'])

        # only 2 features of SoP
        if mode == 3:
            self.X = df[['sp_ge_count', 'sp_score_subs', 'number_of_gap_segments', 'sop_score']]

        if mode == 4: #test removing features
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


        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size)

        if remove_correlated_features:
            correlation_matrix = self.X.corr().abs()

            # Create a mask to select upper triangle of the correlation matrix
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

            # Get columns to drop based on correlation threshold
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

            print(to_drop)

            # Drop correlated columns
            self.X = self.X.drop(columns=to_drop)

            print(self.X)


        # Get unique 'code1' values
        unique_code1 = df['code1'].unique()

        # Split the unique 'code1' into training and test sets
        train_code1, test_code1 = train_test_split(unique_code1, test_size=0.2)
        print(f"the training set is: {train_code1} \n")
        print(f"the testing set is: {test_code1} \n")

        # Create training and test DataFrames by filtering based on 'code1'
        self.train_df = df[df['code1'].isin(train_code1)]
        self.test_df = df[df['code1'].isin(test_code1)]

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
            # self.X_train =self.train_df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2', 'entropy_mean', 'entropy_pct_75', 'msa_len', 'seq_max_len', 'seq_min_len', 'total_gaps', 'gaps_len_one', 'gaps_len_two', 'gaps_len_three_plus', 'num_unique_gaps', 'gaps_1seq_len1', 'gaps_1seq_len2', 'gaps_1seq_len3plus', 'num_cols_no_gaps', 'num_cols_1_gap', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'sp_score_subs_norm', 'sp_score_gap_e_norm', 'sp_match_ratio', 'sp_missmatch_ratio', 'double_char_count', 'bl_sum', 'kurtosis_bl', 'bl_std', 'bl_max', 'k_mer_10_max', 'k_mer_10_var', 'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_norm', 'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_norm', 'number_of_gap_segments', 'number_of_mismatches', 'henikoff_with_gaps', 'henikoff_without_gaps', 'clustal_differential_sum'])
            # self.X_test = self.test_df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2', 'entropy_mean', 'entropy_pct_75', 'msa_len', 'seq_max_len', 'seq_min_len', 'total_gaps', 'gaps_len_one', 'gaps_len_two', 'gaps_len_three_plus', 'num_unique_gaps', 'gaps_1seq_len1', 'gaps_1seq_len2', 'gaps_1seq_len3plus', 'num_cols_no_gaps', 'num_cols_1_gap', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'sp_score_subs_norm', 'sp_score_gap_e_norm', 'sp_match_ratio', 'sp_missmatch_ratio', 'double_char_count', 'bl_sum', 'kurtosis_bl', 'bl_std', 'bl_max', 'k_mer_10_max', 'k_mer_10_var', 'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_norm', 'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_norm', 'number_of_gap_segments', 'number_of_mismatches', 'henikoff_with_gaps', 'henikoff_without_gaps', 'clustal_differential_sum'])
        # all features except 2 sop
        # if mode == 2:
        #     self.X_train = train_df.drop(
        #         columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty','sop_score', 'normalised_sop_score','sp_score_subs_norm', 'sp_score_gap_e_norm',
        #     'sp_match_ratio', 'sp_missmatch_ratio'])
        #     self.X_test = test_df.drop(
        #         columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'sop_score', 'normalised_sop_score', 'sp_score_subs_norm', 'sp_score_gap_e_norm',
        #     'sp_match_ratio', 'sp_missmatch_ratio'])
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

        self.scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # self.X_train_scaled = self.X_train
        # self.X_test_scaled = self.X_test
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # saving the scaler that was used for training
        # joblib.dump(self.scaler, f'./out/scaler_{i}_mode{self.mode}_{self.predicted_measure}.pkl')
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

        # corr_coefficient1, p_value1 = pearsonr(self.test_df['normalised_sop_score'], self.test_df['MEAN_COL_SCORE'])
        # print(f"Pearson Correlation of SOP and MEAN_COL_SCORE in the TEST set: {corr_coefficient1:.4f}\n",
        #       f"P-value of non-correlation: {p_value1:.4f}\n")

        # Set train and test Labels
        self.y_train = self.train_df[true_score_name]
        self.y_test = self.test_df[true_score_name]

        self.binary_feature = self.train_df['class_label'].astype('float64')
        # self.binary_feature_scaled = self.scaler.transform(self.y_train)
        # self.binary_feature_scaled = self.binary_feature_scaled.astype('float64')

        # Check the size of each set
        print(f"Training set size: {self.train_df.shape}")
        print(f"Test set size: {self.test_df.shape}")

        # check for NaNs and inf
        # print(np.any(np.isnan(self.X_train_scaled)))
        # print(np.any(np.isinf(self.X_train_scaled)))
        # print(np.any(np.isnan(self.y_train)))
        # print(np.any(np.isinf(self.y_train)))
        # print(np.any(np.isnan(self.X_test_scaled)))
        # print(np.any(np.isinf(self.X_test_scaled)))
        # print(np.any(np.isnan(self.y_test)))
        # print(np.any(np.isinf(self.y_test)))
        # nan_indices = np.where(np.isnan(self.X_train_scaled))
        # print("NaN indices:", nan_indices)

        # num_nans = np.sum(np.isnan(self.X_train_scaled))
        # print(f"Total NaN values: {num_nans}")

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


    def random_forest(self, n_estimators: int = 100, i: int = 0, n_jobs=-1, cv=5) -> float:
        self.regressor = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)

        cv_scores = cross_val_score(self.regressor, self.X_train_scaled, self.y_train, cv=cv,
                                    scoring='neg_mean_squared_error')

        # cross_val_score returns negative MSE by default, convert to positive
        cv_scores = -cv_scores
        cv_mean_mse = cv_scores.mean()
        cv_std_mse = cv_scores.std()
        print(f"Cross-Validation Mean MSE: {cv_mean_mse:.4f}")
        print(f"Cross-Validation Std MSE: {cv_std_mse:.4f}")

        self.regressor.fit(self.X_train_scaled, self.y_train)

        y_train_pred = self.regressor.predict(self.X_train_scaled)
        self.y_pred = self.regressor.predict(self.X_test_scaled)

        if self.predicted_measure == "tree_distance":
            self.y_pred = np.round(self.y_pred).astype(int)

        train_accuracy = mean_squared_error(self.y_train, y_train_pred)
        test_accuracy = mean_squared_error(self.y_test, self.y_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred
        })

        df_res.to_csv(f'./out/rf_prediction_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        import pickle
        with open(f'./out/random_forest_model_{i}.pkl', 'wb') as file:
            pickle.dump(self.regressor, file)

        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")
        return mse

    # def gradient_boost(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3) -> float:
    #     # Create and fit the Gradient Boosting Regressor
    #     self.regressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    #     self.regressor.fit(self.X_train, self.y_train)
    #
    #     # Make predictions
    #     self.y_pred = self.regressor.predict(self.X_test)
    #
    #     # Evaluate the model
    #     mse = mean_squared_error(self.y_test, self.y_pred)
    #     print(f"Mean Squared Error: {mse:.4f}")
    #     return mse

    # def support_vector(self, kernel: str = 'rbf', c_param: int = 100, gamma: float = 0.1) -> float:
    #     # Create and fit the Support Vector Regressor
    #     self.regressor = SVR(kernel=kernel, C=c_param, gamma=gamma)
    #     self.regressor.fit(self.X_train, self.y_train)
    #
    #     # Make predictions
    #     self.y_pred = self.regressor.predict(self.X_test)
    #
    #     # Evaluate the model
    #     mse = mean_squared_error(self.y_test, self.y_pred)
    #     print(f"Mean Squared Error: {mse:.4f}")
    #     return mse
    #
    # def k_nearest_neighbors(self, n_neighbors: int = 5) -> float:
    #     # Create and fit the K-Nearest Neighbors Regressor
    #     self.regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    #     self.regressor.fit(self.X_train, self.y_train)
    #
    #     # Make predictions
    #     self.y_pred = self.regressor.predict(self.X_test)
    #
    #     # Evaluate the model
    #     mse = mean_squared_error(self.y_test, self.y_pred)
    #     print(f"Mean Squared Error: {mse:.4f}")
    #     return mse

    def deep_learning(self, i=0, epochs=50, batch_size=16, validation_split=0.2, verbose=1, learning_rate=0.01, dropout_rate=0.2, l1=1e-4, l2=1e-4, undersampling = False):
        history = None

        def weighted_mse(y_true, y_pred, weights):
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            weighted_loss = mse_loss * weights
            return weighted_loss

        def weighted_mse_loss(y_true, y_pred):
            # Simple weight based on the true values
            weights = K.exp(-y_true)  # Example: Higher weights for lower scores
            mse_loss = K.mean(weights * K.square(y_true - y_pred))  # Weighted MSE
            return mse_loss

        def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
            # Focal loss penalizes predictions that are farther from the true value
            error = K.abs(y_true - y_pred)
            loss = alpha * K.pow(1 - error, gamma) * K.square(error)
            return K.mean(loss)

        # Custom rank-based loss function using tf.argsort
        # Custom rank-based loss function using tf.argsort
        # def rank_loss(y_true, y_pred, top_k=100):
        #     true_rank = tf.argsort(y_true, axis=-1, direction='ASCENDING')
        #     pred_rank = tf.argsort(y_pred, axis=-1, direction='ASCENDING')
        #     true_top_k = true_rank[:, :top_k]
        #     pred_top_k = pred_rank[:, :top_k]
        #
        #     # rank_diff = tf.abs(true_rank - pred_rank)
        #     rank_diff = tf.reduce_sum(tf.abs(true_top_k - pred_top_k), axis=-1)
        #
        #     return K.mean(rank_diff)
        #     # return tf.reduce_min(rank_diff)

        def rank_loss(y_true, y_pred, top_k):

            paired = tf.stack([y_true, y_pred], axis=-1)
            sorted_paired = tf.sort(paired, axis=-2, direction='ASCENDING', name='sort_true')

            true_top_k = sorted_paired[:, :top_k, 0]
            pred_top_k = sorted_paired[:, :top_k, 1]
            # rank_diff = tf.reduce_mean(K.square(true_top_k - pred_top_k), axis=-1)
            rank_diff = K.mean(K.square(K.cast(true_top_k - pred_top_k, dtype=tf.float32)))

            return rank_diff

        def mse_with_rank_loss(y_true, y_pred, top_k=20, mse_weight=0.5, ranking_weight=0.5):
            mse_loss = K.mean(K.square(K.cast(y_true - y_pred, dtype=tf.float32)))  # MSE loss
            # mse_loss = tf.keras.losses.MSE(y_true, y_pred)
            top_k_rank_loss = rank_loss(y_true, y_pred, top_k)

            mse_weight = tf.cast(mse_weight, dtype=tf.float32)
            ranking_weight = tf.cast(ranking_weight, dtype=tf.float32)
            top_k_rank_loss = tf.cast(top_k_rank_loss, dtype=tf.float32)

            total_loss = mse_weight * mse_loss + ranking_weight * top_k_rank_loss

            return total_loss


        # Define pruning schedule
        # pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        #     initial_sparsity=0.0,  # Start with no pruning
        #     final_sparsity=0.5,  # Final sparsity (50% of weights will be pruned)
        #     begin_step=200,  # Pruning starts after 2000 steps
        #     end_step=400,  # Pruning ends after 10000 steps
        #     frequency=100  # Apply pruning every 100 steps
        # )
        # prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude


        # mode for non-negative regression msa_distance task
        if self.predicted_measure == 'msa_distance':
            model = Sequential()
            model.add(Input(shape=(self.X_train_scaled.shape[1],)))

            #first hidden
            # model.add(Dense(128, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-5)))
            # model.add(Dense(128, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
            model.add(
                Dense(128, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2)))
            # model.add(prune_low_magnitude(
            #     Dense(128, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-5)),
            #     pruning_schedule=pruning_schedule))
            model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
            # model.add(Activation('relu'))
            # model.add(ELU())
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))  # Dropout for regularization

            # # # second new hidden
            # model.add(Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
            model.add(
                Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2)))
            # model.add(prune_low_magnitude(
            #     Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-5)),
            #     pruning_schedule=pruning_schedule))
            model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
            # model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))  # Dropout for regularization

            # second hidden
            model.add(Dense(16, kernel_initializer=GlorotUniform(),kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2)))
            model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
            # model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))  # Dropout for regularization

            # third hidden
            # model.add(prune_low_magnitude(
            #     Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-5)),
            #     pruning_schedule=pruning_schedule))
            # model.add(Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
            # model.add(
            #     Dense(16, kernel_initializer=GlorotUniform(),
            #           kernel_regularizer=regularizers.l2(0.0005)))
            model.add(
                Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2)))
            model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the third hidden layer
            # model.add(ELU())
            # model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))  # Dropout for regularization

            # model.add(Dense(1, activation='exponential')) #exponential ensures no negative values
            # model.add(Dense(1, activation='softplus'))  #ensures non-negative values
            model.add(Dense(1, activation='sigmoid'))  #limits output to 0 to 1 range

            optimizer = Adam(learning_rate=learning_rate)
            # optimizer = RMSprop(learning_rate=learning_rate)

            # model.compile(optimizer=optimizer, loss='mean_squared_error')
            # model.compile(optimizer=optimizer, loss=mse_with_rank_loss)
            model.compile(optimizer=optimizer, loss = lambda y_true, y_pred: mse_with_rank_loss(y_true, y_pred, top_k=8, mse_weight=1,
                                                        ranking_weight=10))
            # model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: quantile_loss(0.02, y_true, y_pred))
            # model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: weighted_mse(y_true, y_pred, weights))

            #set call-backs
            # 1. Implement early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            # 2. learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',  # Metric to monitor
                patience=3,  # Number of epochs with no improvement to wait before reducing the learning rate
                verbose=1,  # Print messages when learning rate is reduced
                factor=0.7,  # Factor by which the learning rate will be reduced
                min_lr=1e-6  # Lower bound on the learning rate
            )

            # pruning_callbacks = [
            #     tfmot.sparsity.keras.UpdatePruningStep(),  # Callback to update pruning schedule during training
            #     early_stopping,
            #     lr_scheduler
            # ]

            callbacks = [
                early_stopping,
                lr_scheduler
            ]

            if undersampling == True:
                # weights = np.where(self.y_train < 0.2, 7, 1)
                # Define thresholds and weights
                threshold_low = 0.015
                threshold_high = 0.1
                w_low = self.weights[0] # Weight for the lower tail (values < threshold_low)
                w_high = self.weights[2]  # Weight for the upper tail (values > threshold_high)
                w_mid = self.weights[1]  # Weight for the middle range (between threshold_low and threshold_high)
                weights = np.where(self.y_train < threshold_low, w_low,
                                   np.where(self.y_train > threshold_high, w_high, w_mid))
                history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose, callbacks=callbacks, sample_weight=weights)
            else:
                history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size,
                                    validation_split=validation_split, verbose=verbose,
                                    callbacks=callbacks)
                # history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size,
                #                     validation_split=validation_split, verbose=verbose,
                #                     callbacks=pruning_callbacks)

            # After training, you can optionally strip the pruning wrappers
            # model = tfmot.sparsity.keras.strip_pruning(model)

        # # mode for non-negative regression tree_distance task
        # elif self.predicted_measure == 'tree_distance':
        #     model = Sequential()
        #     model.add(Input(shape=(self.X_train_scaled.shape[1],)))
        #
        #     # first hidden
        #     model.add(Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4)))
        #     model.add(LeakyReLU(negative_slope=0.01))
        #     model.add(BatchNormalization())
        #     model.add(Dropout(0.2))  # Dropout for regularization
        #
        #     # second hidden
        #     model.add(Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4)))
        #     # model.add(ELU())
        #     model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
        #     model.add(BatchNormalization())
        #     model.add(Dropout(0.2))  # Dropout for regularization
        #
        #     # third hidden
        #     model.add(Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4)))
        #     # model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
        #     model.add(ELU())
        #     model.add(BatchNormalization())
        #     model.add(Dropout(0.2))  # Dropout for regularization
        #
        #     # model.add(Dense(1, activation='exponential')) #exponential ensures no negative values
        #     model.add(Dense(1))  #
        #
        #     optimizer = Adam(learning_rate=0.0001)
        #     model.compile(optimizer=optimizer, loss='mean_squared_error')
        #
        #     # set call-backs
        #     # 1. Implement early stopping
        #     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        #     # 2. learning rate scheduler
        #     lr_scheduler = ReduceLROnPlateau(
        #         monitor='val_loss',  # Metric to monitor
        #         patience=3,  # Number of epochs with no improvement to wait before reducing the learning rate
        #         verbose=1,  # Print messages when learning rate is reduced
        #         factor=0.5,  # Factor by which the learning rate will be reduced
        #         min_lr=1e-6  # Lower bound on the learning rate
        #     )
        #
        #     history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size,
        #                         validation_split=validation_split, verbose=verbose,
        #                         callbacks=[early_stopping])

        # Plotting training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Set integer ticks on the x-axis
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
        # Save the model architecture as a Dot file
        plot_model(model, to_file='./out/model_architecture.dot', show_shapes=True, show_layer_names=True)
        # visualkeras.layered_view(model, to_file='./out/output.png',legend=True, draw_funnel=False, show_dimension=True).show()

        # Evaluate the model
        loss = model.evaluate(self.X_test_scaled, self.y_test)
        print(f"Test Loss: {loss}")

        # Make predictions
        self.y_pred = model.predict(self.X_test_scaled)
        self.y_pred = np.ravel(self.y_pred)  # flatten multi-dimensional array into one-dimensional
        self.y_pred = self.y_pred.astype('float64')

        # get integers predictions of RF distance
        if self.predicted_measure == "tree_distance":
            self.y_pred = np.round(self.y_pred).astype(int)


        # Create a DataFrame
        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred
        })

        # Save the DataFrame to a CSV file
        df_res.to_csv(f'./out/prediction_DL_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")

        # # explain features importance
        # X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
        # X_test_subset = X_test_scaled_with_names.sample(n=500, random_state=42)  # Take a sample of 500 rows
        # explainer = shap.Explainer(model, X_test_subset)
        # shap_values = explainer(X_test_subset)
        # # explainer = shap.Explainer(model, X_test_scaled_with_names)
        # # shap_values = explainer(X_test_scaled_with_names)
        # joblib.dump(explainer,
        #             f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/explainer_{i}_mode{self.mode}_{self.predicted_measure}.pkl')
        # joblib.dump(shap_values,
        #             f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/shap_values__{i}_mode{self.mode}_{self.predicted_measure}.pkl')

        return mse

    # def deep_learning_with_attention(self, epochs=30, batch_size=16, validation_split=0.2, verbose=1, learning_rate=0.01, i=0, undersampling = False):
    #     history = None
    #
    #     # mode for non-negative regression msa_distance task
    #     if self.predicted_measure == 'msa_distance':
    #         # Define input layer
    #         inputs = Input(shape=(self.X_train_scaled.shape[1],))
    #
    #         # First hidden layer
    #         x = Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4))(inputs)
    #         x = LeakyReLU(negative_slope=0.01)(x)
    #         x = BatchNormalization()(x)
    #         x = Dropout(0.2)(x)
    #
    #         # Second hidden layer
    #         x = Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4))(x)
    #         x = LeakyReLU(negative_slope=0.01)(x)
    #         x = BatchNormalization()(x)
    #         x = Dropout(0.2)(x)
    #
    #         # Third hidden layer
    #         x = Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4))(x)
    #         x = LeakyReLU(negative_slope=0.01)(x)
    #         x = BatchNormalization()(x)
    #         x = Dropout(0.2)(x)
    #
    #         # Attention layer
    #         # Assuming AttentionLayer is a custom class. The output and attention weights will be unpacked.
    #         attention_output, attn_weights = AttentionLayer()(x)
    #
    #         # Reshape the attention output to be 2D (batch_size, features) for the next Dense layer
    #         attention_output_reshaped = Reshape((-1,))(attention_output)
    #
    #         # Output layer (sigmoid)
    #         output = Dense(1, activation='sigmoid')(attention_output_reshaped)
    #
    #         # Define the model
    #         model = Model(inputs=inputs, outputs=output)
    #
    #         # Create a sub-model for accessing the attention weights directly
    #         attn_model = Model(inputs=inputs, outputs=[attention_output, attn_weights])
    #
    #         optimizer = Adam(learning_rate=learning_rate)
    #         model.compile(optimizer=optimizer, loss='mean_squared_error')
    #
    #         #set call-backs
    #         # 1. Implement early stopping
    #         early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    #         # 2. learning rate scheduler
    #         lr_scheduler = ReduceLROnPlateau(
    #             monitor='val_loss',  # Metric to monitor
    #             patience=3,  # Number of epochs with no improvement to wait before reducing the learning rate
    #             verbose=1,  # Print messages when learning rate is reduced
    #             factor=0.7,  # Factor by which the learning rate will be reduced
    #             min_lr=1e-5  # Lower bound on the learning rate
    #         )
    #         if undersampling == True:
    #             # weights = np.where(self.y_train < 0.2, 7, 1)
    #             # Define thresholds and weights
    #             threshold_low = 0.2
    #             threshold_high = 0.8
    #             w_low = 5 # Weight for the lower tail (values < threshold_low)
    #             w_high = 2  # Weight for the upper tail (values > threshold_high)
    #             w_mid = 1  # Weight for the middle range (between threshold_low and threshold_high)
    #             weights = np.where(self.y_train < threshold_low, w_low,
    #                                np.where(self.y_train > threshold_high, w_high, w_mid))
    #             history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose, callbacks=[early_stopping, lr_scheduler], sample_weight=weights)
    #         else:
    #             history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size,
    #                                 validation_split=validation_split, verbose=verbose,
    #                                 callbacks=[early_stopping, lr_scheduler])
    #
    #
    #     # Plotting training and validation loss
    #     plt.plot(history.history['loss'], label='Training Loss')
    #     plt.plot(history.history['val_loss'], label='Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #
    #     # Set integer ticks on the x-axis
    #     epochs = range(1, len(history.history['loss']) + 1)  # Integer epoch numbers
    #     plt.xticks(ticks=epochs)  # Set the ticks to integer epoch numbers
    #
    #     plt.legend()
    #     plt.savefig(fname=f'./out/loss_graph_{i}_mode{self.mode}_{self.predicted_measure}.png', format='png')
    #     plt.show()
    #     plt.close()
    #
    #     # visualize model architecture
    #     plot_model(model, to_file=f'./out/model_architecture_{i}_mode{self.mode}_{self.predicted_measure}.png', show_shapes=True, show_layer_names=True,
    #                show_layer_activations=True)
    #     model.save(f'./out/regressor_model_{i}_mode{self.mode}_{self.predicted_measure}.keras')
    #     # Save the model architecture as a Dot file
    #     plot_model(model, to_file='./out/model_architecture.dot', show_shapes=True, show_layer_names=True)
    #     # visualkeras.layered_view(model, to_file='./out/output.png',legend=True, draw_funnel=False, show_dimension=True).show()
    #
    #     # Evaluate the model
    #     loss = model.evaluate(self.X_test_scaled, self.y_test)
    #     print(f"Test Loss: {loss}")
    #
    #     # Make predictions
    #     self.y_pred = model.predict(self.X_test_scaled)
    #     self.y_pred = np.ravel(self.y_pred)  # flatten multi-dimensional array into one-dimensional
    #     self.y_pred = self.y_pred.astype('float64')
    #
    #     # get integers predictions of RF distance
    #     if self.predicted_measure == "tree_distance":
    #         self.y_pred = np.round(self.y_pred).astype(int)
    #
    #
    #     # Create a DataFrame
    #     df_res = pd.DataFrame({
    #         'code1': self.main_codes_test,
    #         'code': self.file_codes_test,
    #         'predicted_score': self.y_pred
    #     })
    #
    #     # Save the DataFrame to a CSV file
    #     df_res.to_csv(f'./out/prediction_DL_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)
    #
    #     # Evaluate the model
    #     mse = mean_squared_error(self.y_test, self.y_pred)
    #     print(f"Mean Squared Error: {mse:.4f}")
    #     corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
    #     print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")
    #
    #     # # To visualize the attention weights for a specific input sample:
    #     # sample_input = self.X_train_scaled[0]  # Use the first sample or any other sample index
    #     # sample_input = np.expand_dims(sample_input, axis=0)  # Add batch dimension
    #     #
    #     # # Get both the attention output and attention weights by calling predict on attn_model
    #     # attn_output_val, attn_weights_val = attn_model.predict(sample_input)
    #     # print(attn_weights_val)
    #
    #     # explain features importance
    #     X_test_scaled_with_names = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
    #     explainer = shap.Explainer(model, X_test_scaled_with_names)
    #     shap_values = explainer(X_test_scaled_with_names)
    #     # explainer = shap.Explainer(model, self.X_test_scaled)
    #     # shap_values = explainer(self.X_test_scaled)
    #     joblib.dump(explainer,
    #                 f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/explainer_{i}_mode{self.mode}_{self.predicted_measure}.pkl')
    #     joblib.dump(shap_values,
    #                 f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/shap_values__{i}_mode{self.mode}_{self.predicted_measure}.pkl')
    #
    #     return mse

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

    def random_forest_classification(self, n_estimators: int = 100, i: int = 0) -> float:

        # class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        # class_weight_dict = dict(enumerate(class_weights))
        # print(class_weight_dict)
        # # class_weights = {0: 10, 1: 1}
        # self.regressor = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight_dict, random_state=42)
        # self.regressor.fit(self.X_train_scaled, self.y_train)


        undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        self.X_train_resampled, self.y_train_resampled = undersample.fit_resample(self.X_train_scaled, self.y_train)
        self.regressor = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=42)
        self.regressor.fit(self.X_train_resampled, self.y_train_resampled)

        # Make predictions
        # y_train_pred = self.regressor.predict(self.X_train)
        # self.y_pred = self.regressor.predict(self.X_test_scaled)
        self.y_prob = self.regressor.predict_proba(self.X_test_scaled)[:, 1]  # Probability for class 1
        self.y_pred = (self.y_prob > 0.7).astype(int)

        if self.y_prob is not None:
            auc = roc_auc_score(self.y_test, self.y_prob)
            print(f"AUC-ROC: {auc:.4f}")
            auc_pr = average_precision_score(self.y_test, self.y_prob)
            print(f"AUC-PR: {auc_pr:.4f}")

        print(classification_report(self.y_test, self.y_pred))

        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred,
            'probabilities': self.y_prob
        })

        # Save the DataFrame to a CSV file
        df_res.to_csv(f'./out/rf_prediction_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        # save the model
        import pickle

        # Assuming 'model' is your trained RandomForestRegressor
        with open(f'./out/random_forest_classifier_model_{i}.pkl', 'wb') as file:
            pickle.dump(self.regressor, file)

        # Precision, Recall, F1-Score
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # AUC-ROC (Only if you have probability scores)
        if self.y_prob is not None:
            auc = roc_auc_score(self.y_test, self.y_prob)
            print(f"AUC-ROC: {auc:.4f}")

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
        plt.show()

        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)

        # Plot precision-recall curve
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
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
        plt.show()

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
        plt.title(f'Random Forest Binary Classifier Feature Importances', fontsize=20)
        plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        plt.xlabel('Importance', fontsize=18)
        plt.ylabel('Features', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(fname=f'./out/features_importances_{i}_mode{self.mode}_{self.predicted_measure}.png', format='png')
        plt.show()
        plt.close()

        return auc

    # def xgb_classification(self, i=0, scale_pos_weight=10):
    #
    #     self.regressor = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
    #     self.regressor.fit(self.X_train, self.y_train)
    #
    #     # Make predictions
    #     # y_train_pred = self.regressor.predict(self.X_train)
    #     self.y_pred = self.regressor.predict(self.X_test)
    #     accuracy = accuracy_score(self.y_test, self.y_pred)
    #     print(f'Accuracy: {accuracy:.4f}')
    #     self.y_prob = self.regressor.predict_proba(self.X_test)[:, 1]  # Probability for class 1
    #
    #     df_res = pd.DataFrame({
    #         'code1': self.main_codes_test,
    #         'code': self.file_codes_test,
    #         'predicted_score': self.y_pred
    #     })
    #
    #     # Save the DataFrame to a CSV file
    #     df_res.to_csv(f'./out/xgb_prediction_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)
    #
    #     # save the model
    #     import pickle
    #
    #     # Assuming 'model' is your trained RandomForestRegressor
    #     with open(f'./out/xgb_classifier_model_{i}.pkl', 'wb') as file:
    #         pickle.dump(self.regressor, file)
    #
    #     # Precision, Recall, F1-Score
    #     precision = precision_score(self.y_test, self.y_pred)
    #     recall = recall_score(self.y_test, self.y_pred)
    #     f1 = f1_score(self.y_test, self.y_pred)
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall: {recall:.4f}")
    #     print(f"F1-Score: {f1:.4f}")
    #
    #     # AUC-ROC (Only if you have probability scores)
    #     if self.y_prob is not None:
    #         auc = roc_auc_score(self.y_test, self.y_prob)
    #         print(f"AUC-ROC: {auc:.4f}")
    #
    #     # Confusion Matrix
    #     cm = confusion_matrix(self.y_test, self.y_pred)
    #     print("Confusion Matrix:")
    #     print(cm)
    #
    #     # Plot Confusion Matrix
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: 0', 'Pred: 1'],
    #                 yticklabels=['True: 0', 'True: 1'])
    #     plt.title('Confusion Matrix')
    #     plt.xlabel('Predicted')
    #     plt.ylabel('True')
    #     plt.show()
    #     return auc
    #
    # def catboost_classification(self, i=0, class_weights={0: 1, 1: 10}, iterations=1000, learning_rate=0.05):
    #
    #     self.regressor = CatBoostClassifier(class_weights=class_weights, iterations=iterations, learning_rate=learning_rate)
    #     self.regressor.fit(self.X_train_scaled, self.y_train)
    #
    #     # Make predictions
    #     # y_train_pred = self.regressor.predict(self.X_train)
    #     self.y_pred = self.regressor.predict(self.X_test_scaled)
    #     accuracy = accuracy_score(self.y_test, self.y_pred)
    #     print(f'Accuracy: {accuracy:.4f}')
    #     self.y_prob = self.regressor.predict_proba(self.X_test_scaled)[:, 1]  # Probability for class 1
    #
    #     df_res = pd.DataFrame({
    #         'code1': self.main_codes_test,
    #         'code': self.file_codes_test,
    #         'predicted_score': self.y_pred,
    #         'probabilities': self.y_prob
    #     })
    #
    #     # Save the DataFrame to a CSV file
    #     df_res.to_csv(f'./out/catboost_prediction_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)
    #
    #     # save the model
    #     import pickle
    #
    #     # Assuming 'model' is your trained RandomForestRegressor
    #     with open(f'./out/catboost_classifier_model_{i}.pkl', 'wb') as file:
    #         pickle.dump(self.regressor, file)
    #
    #     # Precision, Recall, F1-Score
    #     precision = precision_score(self.y_test, self.y_pred)
    #     recall = recall_score(self.y_test, self.y_pred)
    #     f1 = f1_score(self.y_test, self.y_pred)
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall: {recall:.4f}")
    #     print(f"F1-Score: {f1:.4f}")
    #
    #     # AUC-ROC (Only if you have probability scores)
    #     if self.y_prob is not None:
    #         auc = roc_auc_score(self.y_test, self.y_prob)
    #         print(f"AUC-ROC: {auc:.4f}")
    #         auc_pr = average_precision_score(self.y_test, self.y_prob)
    #         print(f"AUC-PR: {auc_pr:.4f}")
    #
    #     # Confusion Matrix
    #     cm = confusion_matrix(self.y_test, self.y_pred)
    #     print("Confusion Matrix:")
    #     print(cm)
    #
    #     # Plot Confusion Matrix
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: 0', 'Pred: 1'],
    #                 yticklabels=['True: 0', 'True: 1'])
    #     plt.title('Confusion Matrix')
    #     plt.xlabel('Predicted')
    #     plt.ylabel('True')
    #     plt.show()
    #     return auc

    def dl_classifier(self, epochs=30, batch_size=64, validation_split=0.2, verbose=1, learning_rate=0.01, i=0):

        # def custom_loss(w0=5.0, w1=1.0):
        #     def loss_cl(y_true, y_pred):
        #         # Convert y_pred to a binary outcome
        #         y_pred = tf.round(y_pred)
        #
        #         # Define custom penalties: higher penalty for class "0"
        #         loss_0 = 2.0 * tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred,
        #                                                                 pos_weight=w0)  # Class 0 has higher weight
        #         loss_1 = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred,
        #                                                           pos_weight=w1)  # Class 1 has a smaller weight
        #
        #         # Combine the loss for both classes
        #         loss = loss_0 * tf.cast(tf.equal(y_true, 0), tf.float32) + loss_1 * tf.cast(tf.equal(y_true, 1), tf.float32)
        #         return tf.reduce_mean(loss)
        #     return loss_cl
        def focal_loss(gamma=1.5, alpha=0.75):
            def focal_loss_fixed(y_true, y_pred):
                epsilon = K.epsilon()
                y_true = K.clip(y_true, epsilon, 1. - epsilon)
                y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

                cross_entropy = -y_true * K.log(y_pred)
                loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
                return K.sum(loss, axis=1)

            return focal_loss_fixed

        def weighted_binary_crossentropy(w0=1.0, w1=1.0):
            def loss(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Prevent log(0)

                # Compute the binary cross-entropy loss with weights
                loss = - (w1 * y_true * tf.keras.backend.log(y_pred) +
                          2 * w0 * (1 - y_true) * tf.keras.backend.log(1 - y_pred))

                return tf.reduce_mean(loss)  # Mean loss over the batch

            return loss


        model = Sequential()
        model.add(Input(shape=(self.X_train_scaled.shape[1],)))

        # first hidden
        model.add(Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4)))
        # model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
        model.add(Activation('relu'))
        # model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.2))  # Dropout for regularization

        # second hidden
        model.add(Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4)))
        # model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))  # Dropout for regularization

        # third hidden
        model.add(Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4)))
        # model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the third hidden layer
        # model.add(ELU())
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))  # Dropout for regularization

        # model.add(Dense(1, activation='exponential')) #exponential ensures no negative values
        # model.add(Dense(1, activation='softplus'))  #ensures non-negative values
        model.add(Dense(1, activation='sigmoid'))  # limits output to 0 to 1 range


        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = dict(enumerate(class_weights))
        print(class_weight_dict)

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(w0=class_weight_dict[0], w1=class_weight_dict[1]),
                      metrics=['accuracy', metrics.AUC(), metrics.AUC(name='auc_weighted')])
        # model.compile(optimizer=optimizer,
        #               loss=custom_loss(w0=class_weight_dict[0], w1=class_weight_dict[1]),
        #               metrics=['accuracy'])

        # set call-backs
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

        history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose, callbacks=[early_stopping, lr_scheduler])

        # visualize model architecture
        plot_model(model, to_file=f'./out/classifier_model_architecture_{i}_mode{self.mode}_{self.predicted_measure}.png',
                   show_shapes=True, show_layer_names=True,
                   show_layer_activations=True)
        model.save(f'./out/classifer_model_{i}_mode{self.mode}_{self.predicted_measure}.keras')
        # Save the model architecture as a Dot file
        plot_model(model, to_file='./out/classifier_model_architecture.dot', show_shapes=True, show_layer_names=True)

        # Plotting training and validation loss
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

        self.y_pred = (self.y_prob >= 0.6).astype(int)
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
        plt.show()

        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)

        # Plot precision-recall curve
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
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
        plt.show()

        return auc