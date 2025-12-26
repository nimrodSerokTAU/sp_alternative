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


class RegressorPortion:
    '''
    features_file: file with all features and labels
    test_size: portion of the codes to be separated into a test set; all MSAs for that specific code would be on the same side of the train-test split
    mode: 1 is all features, 2 is all except SoP features, 3 is only 2 SoP features'''
    def __init__(self, features_file: str, test_size: float, mode: int = 1, predicted_measure: Literal['msa_distance', 'tree_distance', 'class_label'] = 'msa_distance', i=0, portion=1) -> None:
        self.features_file = features_file
        self.test_size = test_size
        self.predicted_measure = predicted_measure
        self.mode = mode
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X, self.y, self.y_pred = None, None, None
        self.prediction = None
        self.main_codes_train = None
        self.file_codes_train = None
        self.main_codes_test = None
        self.file_codes_test = None
        self.portion = portion


        df = pd.read_csv(self.features_file)
        # to make sure that all dataset codes are read as strings and not integers
        df['code1'] = df['code1'].astype(str)
        # Check for missing values
        print("Missing values in each column:\n", df.isnull().sum())
        corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df['dpos_dist_from_true'])
        print(f"Pearson Correlation of SOP and dpos: {corr_coefficient1:.4f}\n", f"P-value of non-correlation: {p_value1:.6f}\n")

        # add normalized_rf
        df["normalized_rf"] = df['rf_from_true']/(df['taxa_num']-1)
        df["class_label"] = np.where(df['dpos_dist_from_true'] <= 0.02, 0, 1)
        df["class_label2"] = np.where(df['dpos_dist_from_true'] <= 0.01, 0, np.where(df['dpos_dist_from_true'] <= 0.05, 1, 2))

        class_label_counts = df['class_label'].dropna().value_counts()
        print(class_label_counts)

        class_label2_counts_train = df['class_label2'].dropna().value_counts()
        print(class_label2_counts_train)

        df = df.dropna()

        if self.predicted_measure == 'msa_distance':
            true_score_name = "dpos_dist_from_true"
        elif self.predicted_measure == 'tree_distance':
            # true_score_name = "rf_from_true"
            true_score_name = 'normalized_rf'
        elif self.predicted_measure == 'class_label':
            true_score_name = 'class_label'

        self.y = df[true_score_name]

        # all features
        if mode == 1:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2'])

        if mode == 2:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2', 'sop_score', 'normalised_sop_score'])

        # only 2 features of SoP
        if mode == 3:
            self.X = df[['normalised_sop_score', 'henikoff_with_gaps', 'sp_missmatch_ratio', 'k_mer_10_norm','henikoff_without_gaps','clustal_mid_root','clustal_differential_sum','number_of_gap_segments','num_cols_no_gaps']]

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


        # Get unique 'code1' values
        unique_code1 = df['code1'].unique()
        unique_code1 = np.random.choice(unique_code1, size=int(len(unique_code1) * portion), replace=False)

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
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty','class_label', 'class_label2'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty','class_label', 'class_label2'])

        if mode == 2:
            self.X_train = self.train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf','class_label', 'class_label2', 'code', 'code1', 'pypythia_msa_difficulty','sop_score', 'normalised_sop_score'])
            self.X_test = self.test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf','class_label', 'class_label2', 'code', 'code1', 'pypythia_msa_difficulty', 'sop_score', 'normalised_sop_score'])

        # 2 sop features
        if mode == 3:
            self.X_train = self.train_df[['normalised_sop_score', 'henikoff_with_gaps', 'sp_missmatch_ratio', 'k_mer_10_norm','henikoff_without_gaps','clustal_mid_root','clustal_differential_sum','number_of_gap_segments','num_cols_no_gaps']]
            self.X_test = self.test_df[['normalised_sop_score', 'henikoff_with_gaps', 'sp_missmatch_ratio', 'k_mer_10_norm','henikoff_without_gaps','clustal_mid_root','clustal_differential_sum','number_of_gap_segments','num_cols_no_gaps']]


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

        # saving the scaler that was used for training
        joblib.dump(self.scaler, f'./out/scaler_{self.portion}_mode{self.mode}_{self.predicted_measure}.pkl')

        self.main_codes_train = self.train_df['code1']
        self.file_codes_train = self.train_df['code']
        class_weights = compute_class_weight('balanced', classes=np.unique(self.train_df['class_label2']), y=self.train_df['class_label2'])
        self.weights = dict(enumerate(class_weights))
        print(self.weights)
        self.main_codes_test = self.test_df['code1']
        self.file_codes_test = self.test_df['code']

        corr_coefficient1, p_value1 = pearsonr(self.test_df['normalised_sop_score'], self.test_df['dpos_dist_from_true'])
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


    def deep_learning(self, epochs=50, batch_size=16, validation_split=0.2, verbose=1, learning_rate=0.01, i=0, undersampling = False):
        history = None

        def weighted_mse(y_true, y_pred, weights):
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            weighted_loss = mse_loss * weights
            return weighted_loss


        # mode for non-negative regression msa_distance task
        if self.predicted_measure == 'msa_distance':
            model = Sequential()
            model.add(Input(shape=(self.X_train_scaled.shape[1],)))

            #first hidden
            # model.add(Dense(128, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-5)))
            model.add(Dense(128, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.0005)))
            # model.add(prune_low_magnitude(
            #     Dense(128, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-5)),
            #     pruning_schedule=pruning_schedule))
            model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
            # model.add(Activation('relu'))
            # model.add(ELU())
            model.add(BatchNormalization())
            model.add(Dropout(0.2))  # Dropout for regularization

            # # # second new hidden
            model.add(Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.0005)))
            # model.add(prune_low_magnitude(
            #     Dense(64, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-5)),
            #     pruning_schedule=pruning_schedule))
            model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
            # model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))  # Dropout for regularization

            # second hidden
            # model.add(Dense(16, kernel_initializer=GlorotUniform(),kernel_regularizer=l2(1e-4)))
            # model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
            # model.add(Activation('relu'))
            # model.add(BatchNormalization())
            # model.add(Dropout(0.2))  # Dropout for regularization

            # third hidden
            # model.add(prune_low_magnitude(
            #     Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-5)),
            #     pruning_schedule=pruning_schedule))
            model.add(Dense(16, kernel_initializer=GlorotUniform(), kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.0005)))
            model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the third hidden layer
            # model.add(ELU())
            # model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))  # Dropout for regularization

            # model.add(Dense(1, activation='exponential')) #exponential ensures no negative values
            # model.add(Dense(1, activation='softplus'))  #ensures non-negative values
            model.add(Dense(1, activation='sigmoid'))  #limits output to 0 to 1 range

            optimizer = Adam(learning_rate=learning_rate)
            # optimizer = RMSprop(learning_rate=learning_rate)

            model.compile(optimizer=optimizer, loss='mean_squared_error')
            # model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: quantile_loss(0.02, y_true, y_pred))
            # model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: weighted_mse(y_true, y_pred, weights))

            #set call-backs
            # 1. Implement early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            # 2. learning rate scheduler
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
                threshold_low = 0.01
                threshold_high = 0.05
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

        # Plotting training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Set integer ticks on the x-axis
        epochs = range(1, len(history.history['loss']) + 1)  # Integer epoch numbers
        plt.xticks(ticks=epochs)  # Set the ticks to integer epoch numbers

        plt.legend()
        plt.savefig(fname=f'./out/loss_graph_{self.portion}_mode{self.mode}_{self.predicted_measure}.png', format='png')
        plt.show()
        plt.close()

        # visualize model architecture
        plot_model(model, to_file=f'./out/model_architecture_{self.portion}_mode{self.mode}_{self.predicted_measure}.png', show_shapes=True, show_layer_names=True,
                   show_layer_activations=True)
        model.save(f'./out/regressor_model_{self.portion}_mode{self.mode}_{self.predicted_measure}.keras')
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
        df_res.to_csv(f'./out/prediction_DL_{self.portion}_mode{self.mode}_{self.predicted_measure}.csv', index=False)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")

        return mse, corr_coefficient


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
        plt.savefig(fname=f'./out/regression_results_{self.portion}_mode{self.mode}_{self.predicted_measure}.png', format='png')
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
        plt.savefig(fname=f'./out/regression_results_{self.portion}_mode{self.mode}_{self.predicted_measure}2.png', format='png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    correlations = []
    mse_values = []
    for portion in [0.1,0.2,0.4]:
        regressor = RegressorPortion("./out/orthomam_features_251224.csv", 0.2, mode=1,
                              predicted_measure='msa_distance', i=0, portion=portion)
        mse, corr_coefficient = regressor.deep_learning(i=0, epochs=50, batch_size=32, learning_rate=1e-4, undersampling = False)
        mse_values.append(mse)
        correlations.append(corr_coefficient)
        regressor.plot_results("dl", mse, 0)
    print(mse_values)
    print(correlations)


