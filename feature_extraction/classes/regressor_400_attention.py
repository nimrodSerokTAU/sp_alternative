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
from tensorflow.keras.layers import Dense, Lambda, Dropout, LeakyReLU, Activation, BatchNormalization, Input, ELU, Attention, Reshape, Embedding, Concatenate, Flatten, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
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
from tensorflow.keras.metrics import AUC

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
    def __init__(self, features, true_labels, msa_ids, msa_files, batch_size=32, validation_split=0.2, is_validation=False):
        self.features = features
        self.true_labels = true_labels
        self.msa_ids = msa_ids
        self.msa_files = msa_files
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.is_validation = is_validation
        self.unique_msa_ids = np.unique(msa_ids)

        split_idx = int(len(self.unique_msa_ids) * (1 - self.validation_split))

        if not is_validation:
            self.train_msa_ids = self.unique_msa_ids[:split_idx]
            self.features = self.features[np.isin(self.msa_ids, self.train_msa_ids)]
            self.true_labels = self.true_labels[np.isin(self.msa_ids, self.train_msa_ids)]
            self.msa_files = self.msa_files[np.isin(self.msa_ids, self.train_msa_ids)]
            self.msa_ids = self.msa_ids[np.isin(self.msa_ids, self.train_msa_ids)]
        else:
            self.val_msa_ids = self.unique_msa_ids[split_idx:]
            self.features = self.features[np.isin(self.msa_ids, self.val_msa_ids)]
            self.true_labels = self.true_labels[np.isin(self.msa_ids, self.val_msa_ids)]
            self.msa_files = self.msa_files[np.isin(self.msa_ids, self.val_msa_ids)]
            self.msa_ids = self.msa_ids[np.isin(self.msa_ids, self.val_msa_ids)]

        self.unique_msa_ids = np.unique(self.msa_ids)
        self.pairs = self._create_pairs()
        self.batches = self._precompute_batches()

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
            batch_features_1 = tf.convert_to_tensor(batch_features_1, dtype=tf.float32)
            batch_features_2 = tf.convert_to_tensor(batch_features_2, dtype=tf.float32)
            batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
            batches.append(((batch_features_1, batch_features_2), batch_labels))
        np.random.shuffle(batches)
        return batches

    def _create_pairs(self):
        pairs = []

        for msa_id in self.unique_msa_ids:
            try:
                condition_msa_id = (self.msa_ids == msa_id)
                condition_concat = self.msa_files.str.contains('concat')
                combined_condition = condition_msa_id & condition_concat
                idx = np.where(combined_condition)[0]
                if len(idx) >= 150:
                    random_indices = np.random.choice(idx, size=150, replace=False)
                else:
                    random_indices = idx
                for i in range(len(random_indices)-1):
                    for j in range(i+1, len(random_indices)):
                        dist_i = self.true_labels.iloc[random_indices[i]]
                        dist_j = self.true_labels.iloc[random_indices[j]]
                        if dist_i < dist_j:
                            pairs.append((random_indices[i], random_indices[j], 0))  # i is closer (first sample is closer)
                        else:
                            pairs.append((random_indices[i], random_indices[j], 1))  # j is closer or equal (second sample is closer)

                idx = np.where(self.msa_ids == msa_id)[0]
                if len(idx) >= 150:
                    random_indices = np.random.choice(idx, size=150, replace=False)
                else:
                    random_indices = idx
                    # continue
                for i in range(len(random_indices)-1):
                    for j in range(i+1, len(random_indices)):
                        dist_i = self.true_labels.iloc[random_indices[i]]
                        dist_j = self.true_labels.iloc[random_indices[j]]
                        if dist_i < dist_j:
                            pairs.append((random_indices[i], random_indices[j], 0))  # i is closer (first sample is closer)
                        else:
                            pairs.append((random_indices[i], random_indices[j], 1))  # j is closer or equal (second sample is closer)
            except Exception as e:
                print(f"Exception1 {msa_id}, {random_indices[i]}, {random_indices[j]}: {e}\n")
        np.random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def on_epoch_end(self):
        self.pairs = shuffle(self.pairs)
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

    def prepare_data(self, X_scaled, y, main_codes, file_codes, num_samples=5, msa_per_sample=400):
        X_final = []
        y_final = []
        files_final = []
        codes_final = []

        unique_codes = np.unique(main_codes)

        for code in unique_codes:
            idx = np.where(main_codes == code)[0]
            X_code = X_scaled[idx]
            y_code = y.iloc[idx]
            files = file_codes.iloc[idx]

            if len(X_code) >= msa_per_sample:
                for _ in range(num_samples):
                    sampled_idx = np.random.choice(len(X_code), msa_per_sample, replace=False)
                    X_sample = X_code[sampled_idx]
                    y_sample = y_code.iloc[sampled_idx].values
                    files_sample = files.iloc[sampled_idx]

                    best_idx = np.argmin(y_sample)
                    label = np.zeros(msa_per_sample)
                    label[best_idx] = 1

                    X_final.append(X_sample)
                    y_final.append(label)
                    files_final.append(files_sample)
                    codes_final.append(code)

        X_final = np.array(X_final)
        y_final = np.array(y_final)
        files_final = np.array(files_final)
        codes_final = np.array(codes_final)

        return X_final, y_final, files_final, codes_final


    def deep_learning(self, epochs=50, batch_size=16, validation_split=0.2, verbose=1, learning_rate=0.01, dropout_rate=0.2, l1=1e-5, l2=1e-4, attention_size=64, num_heads=4, i=0, undersampling = False):
        history = None
        msa_per_sample = 200
        num_samples = 40
        X_train_final, y_train_final, files_train_final, codes_train_final = self.prepare_data(self.X_train_scaled, self.y_train, self.main_codes_train, self.file_codes_train, num_samples=num_samples, msa_per_sample=msa_per_sample)
        X_test_final, y_test_final, files_test_final, codes_test_final = self.prepare_data(self.X_test_scaled, self.y_test, self.main_codes_test, self.file_codes_test, num_samples=num_samples, msa_per_sample=msa_per_sample)

        print(f"X_train_final shape: {X_train_final.shape}")  # (num_samples, 400, num_features)
        print(f"y_train_final shape: {y_train_final.shape}")  # (num_samples, 400)
        print(f"X_test_final shape: {X_test_final.shape}")  # (num_samples, 400, num_features)
        print(f"y_test_final shape: {y_test_final.shape}")  # (num_samples, 400)

        def build_attention_model(num_MSA, num_features):
            input_layer = Input(shape=(num_MSA, num_features))

            # attention_output = Attention()([input_layer, input_layer])
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=attention_size)(input_layer, input_layer)
            attention_output = BatchNormalization()(attention_output)

            x = GlobalAveragePooling1D()(attention_output)
            x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
            x = Dropout(dropout_rate)(x)

            x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
            # x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
            # x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

            output_layer = Dense(num_MSA, activation='softmax')(x)

            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
            return model


        number_of_features = self.X_train_scaled.shape[1]  # Number of features
        model = build_attention_model(msa_per_sample, number_of_features)
        # model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        history = model.fit(X_train_final, y_train_final, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split, callbacks=[early_stopping, lr_scheduler])

        test_loss, test_acc = model.evaluate(X_test_final, y_test_final)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

        self.y_pred = model.predict(X_test_final)
        # self.y_pred = np.ravel(self.y_pred)  # flatten multi-dimensional array into one-dimensional
        # self.y_pred = self.y_pred.astype('float64')
        self.predicted_best_msa = np.argmax(self.y_pred, axis=1)
        self.true_best_msa = np.argmax(y_test_final, axis=1)
        correct_predictions = np.sum(self.predicted_best_msa == self.true_best_msa)
        print(f"Correct Predictions: {correct_predictions}/{len(self.predicted_best_msa)}")

        # plot training progress (loss, accuracy) over epochs
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(fname=f'./out/loss_graph_{i}_mode{self.mode}_{self.predicted_measure}.png', format='png')
        plt.show()
        plt.close()

        plot_model(model, to_file=f'./out/model_architecture_{i}_mode{self.mode}_{self.predicted_measure}.png',
                   show_shapes=True, show_layer_names=True,
                   show_layer_activations=True)
        model.save(f'./out/regressor_model_{i}_mode{self.mode}_{self.predicted_measure}.keras')
        # Save the model architecture as a Dot file
        plot_model(model, to_file='./out/model_architecture.dot', show_shapes=True, show_layer_names=True)

        # df_res = pd.DataFrame({
        #     'code1': codes_test_final,
        #     'code': files_test_final,
        #     'predicted_score': self.y_pred
        # })
        #
        # Save the DataFrame to a CSV file
        # df_res.to_csv(f'./out/prediction_DL_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        return correct_predictions

    def plot_results(self, model_name: Literal["svr", "rf", "knn-r", "gbr", "dl"], mse: float, i: int) -> None:

        plt.figure(figsize=(12, 6))
        plt.hist(self.predicted_best_msa, bins=30, alpha=0.6, label='Predicted Best MSA')
        plt.hist(self.true_best_msa, bins=30, alpha=0.6, label='True Best MSA')
        plt.title('Histogram of Predicted vs True Best MSA')
        plt.xlabel('MSA Index')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()