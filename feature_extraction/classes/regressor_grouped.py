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
    def __init__(self, features, true_labels, true_msa_ids, batch_size,validation_split=0.2, is_validation=False):
        self.features = features
        self.true_labels = true_labels
        self.msa_ids = true_msa_ids  # TRUE MSA IDs (categorical)
        self.batch_size = batch_size
        self.unique_msa_ids = np.unique(true_msa_ids)  # Get unique TRUE MSA IDs
        self.validation_split = validation_split
        self.is_validation = is_validation

        # num_samples = len(self.features)
        split_idx = int(len(self.unique_msa_ids) * (1 - self.validation_split))

        if self.is_validation:
            self.val_msa_ids = self.unique_msa_ids[split_idx:]  # First part is validation
            # self.train_msa_ids = self.unique_msa_ids[split_idx:]  # The rest is training
        else:
            # self.val_msa_ids = self.unique_msa_ids[split_idx:]  # First part is validation
            self.train_msa_ids = self.unique_msa_ids[:split_idx]

        if self.is_validation:
            self.features = self.features[np.isin(self.msa_ids, self.val_msa_ids)]
            self.true_labels = self.true_labels[np.isin(self.msa_ids, self.val_msa_ids)]
            self.msa_ids = self.msa_ids[np.isin(self.msa_ids, self.val_msa_ids)]
        else:
            # Filter for training data
            self.features = self.features[np.isin(self.msa_ids, self.train_msa_ids)]
            self.true_labels = self.true_labels[np.isin(self.msa_ids, self.train_msa_ids)]
            self.msa_ids = self.msa_ids[np.isin(self.msa_ids, self.train_msa_ids)]

        self.unique_msa_ids = np.unique(self.msa_ids)

    def __len__(self):
        # return len(self.unique_msa_ids)
        batch_count = 0
        for msa_id in self.unique_msa_ids:
            batch_count += len(self.features[self.msa_ids == msa_id]) // self.batch_size  # Whole batches per MSA ID
            # If there are remaining samples that don't fill a batch, count that as 1 more batch
            if len(self.features[self.msa_ids == msa_id]) % self.batch_size != 0:
                batch_count += 1
        return batch_count

    def __getitem__(self, idx):
        # msa_id = self.unique_msa_ids[idx]
        #
        # batch_features = self.features[self.msa_ids == msa_id]
        # batch_labels = self.true_labels[self.msa_ids == msa_id]
        #
        # if len(batch_features) < self.batch_size:
        #     repeat_times = (self.batch_size // len(batch_features)) + 1
        #     batch_features = np.tile(batch_features, (repeat_times, 1))[:self.batch_size]
        #     batch_labels = np.tile(batch_labels, (repeat_times, 1))[:self.batch_size]
        #
        # return batch_features, batch_labels
        batch_start = 0
        for msa_id in self.unique_msa_ids:
            msa_id_samples = len(self.features[self.msa_ids == msa_id])
            # Check how many batches we have for this MSA ID
            num_batches_for_msa_id = msa_id_samples // self.batch_size
            if msa_id_samples % self.batch_size != 0:
                num_batches_for_msa_id += 1

            # Check if the batch index falls within the range of batches for this MSA ID
            if idx < batch_start + num_batches_for_msa_id:
                # We're in the right MSA ID, calculate the batch index within this ID's samples
                local_idx = idx - batch_start
                batch_features, batch_labels = self.get_batch_for_msa_id(msa_id, local_idx)
                return batch_features, batch_labels
            batch_start += num_batches_for_msa_id

    def get_batch_for_msa_id(self, msa_id, batch_idx):
        # Get all samples for the MSA ID
        batch_features = self.features[self.msa_ids == msa_id]
        batch_labels = self.true_labels[self.msa_ids == msa_id]

        # Number of full batches for this MSA ID
        num_full_batches = len(batch_features) // self.batch_size

        if batch_idx < num_full_batches:
            # Standard batch (full batch of size `self.batch_size`)
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            return batch_features[start_idx:end_idx], batch_labels[start_idx:end_idx]
        else:
            # Last batch (fewer than `self.batch_size` samples)
            start_idx = batch_idx * self.batch_size
            return batch_features[start_idx:], batch_labels[start_idx:]

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

        df['aligner'] = df.apply(assign_aligner, axis=1)
        df = df[df['aligner'] != 'true'] #removed true MSAs from the data
        df = pd.get_dummies(df, columns=['aligner'], prefix='aligner') #added one-hot encoding for msa aligner program with the columns names of the form "aligner_mafft", "aligner_..."

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

        self.scaler = MinMaxScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
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


    def random_forest(self, n_estimators: int = 100, i: int = 0) -> float:
        # Create and fit the Random Forest Regressor
        self.regressor = RandomForestRegressor(n_estimators=n_estimators)
        self.regressor.fit(self.X_train, self.y_train)

        # Make predictions
        y_train_pred = self.regressor.predict(self.X_train)
        self.y_pred = self.regressor.predict(self.X_test)

        if self.predicted_measure == "tree_distance":
            self.y_pred = np.round(self.y_pred).astype(int)

        # Calculate accuracy
        train_accuracy = mean_squared_error(self.y_train, y_train_pred)
        test_accuracy = mean_squared_error(self.y_test, self.y_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Create a DataFrame
        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred
        })

        # Save the DataFrame to a CSV file
        df_res.to_csv(f'./out/rf_prediction_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        # Assuming 'model' is your trained RandomForestRegressor
        with open(f'./out/random_forest_model_{i}.pkl', 'wb') as file:
            pickle.dump(self.regressor, file)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")
        return mse


    def deep_learning(self, epochs=50, batch_size=16, validation_split=0.2, verbose=1, learning_rate=0.01, i=0, undersampling = False):
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

        def rank_loss(y_true, y_pred, top_k):

            paired = tf.stack([y_true, y_pred], axis=-1)
            sorted_paired = tf.sort(paired, axis=-2, direction='ASCENDING', name='_')

            true_top_k = sorted_paired[:, :top_k, 0]
            pred_top_k = sorted_paired[:, :top_k, 1]

            rank_diff = K.mean(K.square(K.cast(true_top_k - pred_top_k, dtype=tf.float32)))

            return rank_diff

        # Combine MSE loss with rank-based loss
        def mse_with_rank_loss(y_true, y_pred, top_k=3, mse_weight=0.5, ranking_weight=0.5):
            mse_loss = K.mean(K.square(K.cast(y_true - y_pred, dtype=tf.float32)))  # MSE loss
            # mse_loss = tf.keras.losses.MSE(y_true, y_pred)
            top_k_rank_loss = rank_loss(y_true, y_pred, top_k)

            mse_weight = tf.cast(mse_weight, dtype=tf.float32)
            # rank_loss_value = rank_loss(y_true, y_pred)
            ranking_weight = tf.cast(ranking_weight, dtype=tf.float32)
            # rank_loss_value = tf.cast(rank_loss_value, dtype=tf.float32)
            top_k_rank_loss = tf.cast(top_k_rank_loss, dtype=tf.float32)

            total_loss = mse_weight * mse_loss + ranking_weight * top_k_rank_loss

            # return mse_loss + rank_weight * rank_loss_value  # Combine them with a weight
            return total_loss


        # non-negative regression msa_distance task
        if self.predicted_measure == 'msa_distance':
            model = Sequential()
            model.add(Input(shape=(self.X_train_scaled.shape[1],)))

            #first hidden
            model.add(
                Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            # second hidden
            model.add(
                Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            # third hidden
            model.add(Dense(16, kernel_initializer=GlorotUniform(),kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            # fourth hidden
            model.add(
                Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Dense(1, activation='sigmoid'))  #limits output to 0 to 1 range

            optimizer = Adam(learning_rate=learning_rate)
            # optimizer = RMSprop(learning_rate=learning_rate)

            batch_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train,
                                             true_msa_ids=self.main_codes_train, batch_size=batch_size, validation_split=validation_split, is_validation=False)

            val_generator = BatchGenerator(features=self.X_train_scaled, true_labels=self.y_train, true_msa_ids=self.main_codes_train,
                                             batch_size=batch_size, validation_split=validation_split, is_validation=True)

            # model.compile(optimizer=optimizer, loss='mean_squared_error')
            # model.compile(optimizer=optimizer, loss=mse_with_rank_loss)
            # model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: weighted_mse(y_true, y_pred, weights))
            # model.compile(optimizer=optimizer, loss = lambda y_true, y_pred: mse_with_rank_loss(y_true, y_pred, top_k=20, mse_weight=0.5,
            #                                             ranking_weight=0.5))
            model.compile(optimizer=optimizer,
                          loss=lambda y_true, y_pred: mse_with_rank_loss(y_true, y_pred, top_k=3, mse_weight=0.5,
                                                                         ranking_weight=0.5))

            # Callback 1: early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            # Callback 2: learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',  # Metric to monitor
                patience=3,  # Number of epochs with no improvement to wait before reducing the learning rate
                verbose=1,  # Print messages when learning rate is reduced
                factor=0.7,  # Factor by which the learning rate will be reduced
                min_lr=1e-6  # Lower bound on the learning rate
            )

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
                # history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose, callbacks=callbacks, sample_weight=weights)
                history = model.fit(batch_generator, epochs=epochs, validation_data=val_generator, verbose=verbose,
                                    callbacks=callbacks, sample_weight=weights)
            else:
                # history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size,
                #                     validation_split=validation_split, verbose=verbose,
                #                     callbacks=callbacks)
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

        loss = model.evaluate(self.X_test_scaled, self.y_test)
        print(f"Test Loss: {loss}")

        self.y_pred = model.predict(self.X_test_scaled)
        self.y_pred = np.ravel(self.y_pred)  # flatten multi-dimensional array into one-dimensional
        self.y_pred = self.y_pred.astype('float64')

        if self.predicted_measure == "tree_distance":
            self.y_pred = np.round(self.y_pred).astype(int)

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