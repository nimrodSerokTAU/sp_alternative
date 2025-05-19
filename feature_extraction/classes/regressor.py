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
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation, BatchNormalization, Input, ELU, Attention, Reshape, Embedding, Concatenate, Flatten
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
from feature_extraction.classes.batch_generator import BatchGenerator

def _assign_aligner(row: pd.Series) -> str:
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
        scaled_series.loc[group_df.index] = scaled

    return scaled_series

def _check_missing_values(df: pd.DataFrame, verbose) -> pd.DataFrame:
    if verbose == 1:
        print("Missing values in each column:\n", df.isnull().sum())
    df = df.dropna()

    return df

def _print_correlations(df: pd.DataFrame, true_score_name: str) -> None:
    corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df[true_score_name])
    print(f"Pearson Correlation of Normalized SOP and {true_score_name}: {corr_coefficient1:.4f}\n",
          f"P-value of non-correlation: {p_value1:.6f}\n")
    corr_coefficient1, p_value1 = pearsonr(df['sop_score'], df[true_score_name])
    print(f"Pearson Correlation of SOP and {true_score_name}: {corr_coefficient1:.4f}\n",
          f"P-value of non-correlation: {p_value1:.6f}\n")
def _assign_true_score_name(predicted_measure: str) -> str:
    if predicted_measure == 'msa_distance':
        # true_score_name = "dpos_dist_from_true"
        true_score_name = "dpos_ng_dist_from_true"
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


class Regressor:
    '''
    features_file: file with all features and labels
    test_size: portion of the codes to be separated into a test set; all MSAs for that specific code would be on the same side of the train-test split
    mode: 1 is all features (default), 2 is all except SoP features, 3 is chosen list of features
    remove_correlated_features: if the highly correlated features should be removed (boolean, default value: False)
    predicted_measure: 'msa_distance' is a default measure
    scale_labels: y-labels by default are also rank-percentile scaled
    '''
    def __init__(self, features_file: str, test_size: float, mode: int = 1, remove_correlated_features: bool = False, predicted_measure: Literal['msa_distance', 'class_label'] = 'msa_distance', i: int = 0, scale_labels: bool = True, verbose: int = 1) -> None:
        self.verbose = verbose
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
        self.remove_correlated_features: bool = remove_correlated_features

        df = _read_features_into_df(self.features_file)
        self.true_score_name = _assign_true_score_name(self.predicted_measure)

        # df["conserved_col_pct"] = df['num_cols_no_gaps'] / df['msa_len']
        df['aligner'] = df.apply(_assign_aligner, axis=1)
        df.to_csv('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/features_w_aligner.csv', index=False)

        df = _check_missing_values(df, self.verbose)
        if self.verbose == 1:
            _print_correlations(df, self.true_score_name)

        self._split_into_training_test(df, test_size)

        self._finalize_features(df)
        self._scale(scale_labels = scale_labels, i = i)

        if self.verbose == 1:
            _print_correlations(self.test_df, self.true_score_name)
        self._save_scaled(i)



    def _scale(self, scale_labels: bool = True, i: int = 0):
        scaler = GroupAwareScaler(global_scaler=RobustScaler())
        self.X_train_scaled = scaler.fit_transform(self.train_df, group_col="code1", feature_cols=self.X_train.columns)
        self.X_test_scaled = scaler.transform(self.test_df)
        scaler.save(
            f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/scaler_{i}_mode{self.mode}_{self.predicted_measure}.pkl')
        self.final_features_names = scaler.get_feature_names_out()
        # scaler = GroupAwareScaler()
        # scaler.load("group_aware_scaler.pkl")

        """ SCALED y-labels """
        self.y_train_scaled = _rank_percentile_scale_targets(y_true =self.y_train , group_codes = self.main_codes_train) #TODO remove this line
        self.y_test_scaled = _rank_percentile_scale_targets(y_true=self.y_test, group_codes=self.main_codes_test)  # TODO remove this line
        self.y_train = self.y_train_scaled # TODO remove this line
        self.y_test = self.y_test_scaled # TODO remove this line
        """ SCALED y-labels """

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
        x_test_scaled_to_save.to_csv(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/test_scaled_{i}.csv',
                                     index=False)
        self.test_df.to_csv(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/test_unscaled_{i}.csv',
                            index=False)


    def _split_into_training_test(self, df, test_size):
        self.unique_code1 = df['code1'].unique()
        self.train_code1, self.test_code1 = train_test_split(self.unique_code1,
                                                   test_size=test_size)  # TODO add random state for reproducability
        # train_code1, test_code1 = train_test_split(unique_code1, test_size=test_size, random_state=42)

        if self.verbose == 1:
            print(f"the training set is: {self.train_code1} \n")
            print(f"the testing set is: {self.test_code1} \n")

        # Create training and test DataFrames by filtering based on 'code1'
        self.train_df = df[df['code1'].isin(self.train_code1)]
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
        columns_to_drop_dft = ['dpos_dist_from_true', 'rf_from_true', 'code', 'code1',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'aligner', 'dpos_ng_dist_from_true']
        columns_to_drop_extended = columns_to_drop_dft + ['sop_score']
        columns_to_choose = ['constant_sites_pct', 'sop_score', 'entropy_mean', 'sp_score_subs_norm', 'sp_ge_count',
                         'number_of_gap_segments', 'nj_parsimony_score', 'msa_len', 'num_cols_no_gaps', 'total_gaps',
                         'entropy_var', 'num_unique_gaps', 'sp_score_gap_e_norm', 'k_mer_10_mean', 'av_gaps',
                         'n_unique_sites', 'skew_bl', 'median_bl', 'bl_75_pct', 'avg_unique_gap', 'k_mer_20_var',
                         'k_mer_10_top_10_norm', 'gaps_2seq_len3plus', 'gaps_1seq_len3plus', 'num_cols_1_gap',
                         'single_char_count']

        self.y = df[self.true_score_name]

        if self.mode == 1:
            self.X = df.drop(
                columns=columns_to_drop_dft)
            self.X_train = self.train_df.drop(
                columns=columns_to_drop_dft)
            self.X_test = self.test_df.drop(
                columns=columns_to_drop_dft)

        if self.mode == 2:
            self.X = df.drop(
                columns=columns_to_drop_extended)
            self.X_train = self.train_df.drop(
                columns=columns_to_drop_extended)
            self.X_test = self.test_df.drop(
                columns=columns_to_drop_extended)
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

    def deep_learning(self, epochs: int = 50, batch_size: int = 16, validation_split: float = 0.2, verbose: int = 1, learning_rate: float = 0.01, dropout_rate: float = 0.2, l1: float = 1e-5, l2: float = 1e-5, i: int = 0, undersampling: bool = False, repeats: int = 1, mixed_portion: float = 0.3, top_k: int = 4, mse_weight: float = 1, ranking_weight: float = 50, per_aligner: bool = False) -> float:
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

            model.add(Dense(1, activation='sigmoid'))  #limits output to 0 to 1 range

            optimizer = Adam(learning_rate=learning_rate)
            # optimizer = RMSprop(learning_rate=learning_rate)

            # model.compile(optimizer=optimizer, loss='mean_squared_error')
            model.compile(optimizer=optimizer, loss = lambda y_true, y_pred: mse_with_rank_loss(y_true, y_pred, top_k=top_k, mse_weight=mse_weight,
                                                        ranking_weight=ranking_weight))

            unique_train_codes = self.main_codes_train.unique()
            train_msa_ids, val_msa_ids = train_test_split(unique_train_codes, test_size=0.2)
            if self.verbose == 1:
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

        df_res.to_csv(f'./out/prediction_DL_{i}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        mse = mean_squared_error(self.y_test, self.y_pred)
        if self.verbose == 1:
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
