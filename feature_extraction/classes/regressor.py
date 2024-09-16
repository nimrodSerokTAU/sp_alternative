import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from typing import Literal
from scipy.stats import pearsonr

import pydot
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation, BatchNormalization, Input, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

class Regressor:
    '''
    features_file: file with all features and labels
    test_size: portion of the codes to be separated into a test set; all MSAs for that specific code would be on the same side of the train-test split
    mode: 1 is all features, 2 is all except SoP features, 3 is only 2 SoP features'''
    def __init__(self, features_file: str, test_size: float, mode: int = 1, predicted_measure: Literal['msa_distance', 'tree_distance'] = 'msa_distance') -> None:
        self.features_file = features_file
        self.test_size = test_size
        self.predicted_measure = predicted_measure
        # self.num_estimators = n_estimators
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X, self.y, self.y_pred = None, None, None
        self.prediction = None
        self.main_codes_train = None
        self.file_codes_train = None
        self.main_codes_test = None
        self.file_codes_test = None


        df = pd.read_csv(self.features_file)
        # Check for missing values
        print("Missing values in each column:\n", df.isnull().sum())
        corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df['dpos_dist_from_true'])
        print(f"Pearson Correlation of SOP and dpos: {corr_coefficient1:.4f}\n", f"P-value of non-correlation: {p_value1:.4f}\n")

        # Handle missing values (if any)
        # Example: Filling missing values with the mean (for numerical columns)
        # df['orig_tree_ll'] = df['orig_tree_ll'].fillna(df['orig_tree_ll'].mean())

        if self.predicted_measure == 'msa_distance':
            true_score_name = "dpos_dist_from_true"
        elif self.predicted_measure == 'tree_distance':
            true_score_name = "rf_from_true"

        self.y = df[true_score_name]

        # all features
        if mode == 1:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'code', 'code1', 'taxa_num', 'pypythia_msa_difficulty'])

        # all features except 2 features of SoP
        if mode == 2:
            self.X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'code', 'code1', 'taxa_num', 'pypythia_msa_difficulty', 'sop_score', 'normalised_sop_score'])

        # only 2 features of SoP
        if mode == 3:
            self.X = df[['sop_score', 'normalised_sop_score']]


        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size)

        # Get unique 'code1' values
        unique_code1 = df['code1'].unique()

        # Split the unique 'code1' into training and test sets
        train_code1, test_code1 = train_test_split(unique_code1, test_size=0.2)
        print(f"the training set is: {train_code1} \n")
        print(f"the testing set is: {test_code1} \n")

        # Create training and test DataFrames by filtering based on 'code1'
        train_df = df[df['code1'].isin(train_code1)]
        test_df = df[df['code1'].isin(test_code1)]

        scaler = MinMaxScaler()
        # all features
        if mode == 1:
            self.X_train = train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'code', 'code1', 'taxa_num', 'pypythia_msa_difficulty'])
            self.X_test = test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'code', 'code1', 'taxa_num', 'pypythia_msa_difficulty'])

        # all features except 2 sop
        if mode == 2:
            self.X_train = train_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'code', 'code1', 'taxa_num', 'pypythia_msa_difficulty','sop_score', 'normalised_sop_score'])
            self.X_test = test_df.drop(
                columns=['dpos_dist_from_true', 'rf_from_true', 'code', 'code1', 'taxa_num', 'pypythia_msa_difficulty', 'sop_score', 'normalised_sop_score'])

        # 2 sop features
        if mode == 3:
            self.X_train = train_df[['sop_score', 'normalised_sop_score']]
            self.X_test = test_df[['sop_score', 'normalised_sop_score']]

        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.fit_transform(self.X_test)
        self.main_codes_train = train_df['code1']
        self.file_codes_train = train_df['code']
        self.main_codes_test = test_df['code1']
        self.file_codes_test = test_df['code']

        corr_coefficient1, p_value1 = pearsonr(test_df['normalised_sop_score'], test_df['dpos_dist_from_true'])
        print(f"Pearson Correlation of SOP and dpos in the TEST set: {corr_coefficient1:.4f}\n",
              f"P-value of non-correlation: {p_value1:.4f}\n")

        # Set train and test Labels
        # self.y_train = train_df["dpos_dist_from_true"]
        # self.y_test = test_df["dpos_dist_from_true"]
        self.y_train = train_df[true_score_name]
        self.y_test = test_df[true_score_name]

        # Check the size of each set
        print(f"Training set size: {train_df.shape}")
        print(f"Test set size: {test_df.shape}")

    def random_forest(self, n_estimators: int = 100, i: int = 0) -> float:
        # Create and fit the Random Forest Regressor
        self.regressor = RandomForestRegressor(n_estimators=n_estimators)
        self.regressor.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.regressor.predict(self.X_test)

        # Create a DataFrame
        df_res = pd.DataFrame({
            'Code': self.main_codes_test,
            'File': self.file_codes_test,
            'PredictedValue': self.y_pred
        })

        # Save the DataFrame to a CSV file
        df_res.to_csv(f'/Users/kpolonsky/Downloads/TEST/Features/prediction_{i}.csv', index=False)


        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")
        return mse

    def gradient_boost(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3) -> float:
        # Create and fit the Gradient Boosting Regressor
        self.regressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        self.regressor.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.regressor.predict(self.X_test)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    def support_vector(self, kernel: str = 'rbf', c_param: int = 100, gamma: float = 0.1) -> float:
        # Create and fit the Support Vector Regressor
        self.regressor = SVR(kernel=kernel, C=c_param, gamma=gamma)
        self.regressor.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.regressor.predict(self.X_test)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    def k_nearest_neighbors(self, n_neighbors: int = 5) -> float:
        # Create and fit the K-Nearest Neighbors Regressor
        self.regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.regressor.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.regressor.predict(self.X_test)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    def deep_learning(self, i, epochs=50, batch_size=16, validation_split=0.1, verbose=1):
        model = Sequential()
        model.add(Input(shape=(self.X_train_scaled.shape[1],)))

        #first hidden
        model.add(Dense(32, kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-4)))
        model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
        model.add(BatchNormalization())
        model.add(Dropout(0.2))  # Dropout for regularization

        # second hidden
        model.add(Dense(32, kernel_initializer=GlorotUniform(),kernel_regularizer=l2(1e-4)))
        model.add(LeakyReLU(negative_slope=0.01))  # Leaky ReLU for the second hidden layer
        model.add(BatchNormalization())
        model.add(Dropout(0.2))  # Dropout for regularization


        model.add(Dense(1, activation='exponential')) #exponential ensures no negative values

        optimizer = Adam(learning_rate=0.0012)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        #set call-backs
        # 1. Implement early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # 2. learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',  # Metric to monitor
            patience=3,  # Number of epochs with no improvement to wait before reducing the learning rate
            verbose=1,  # Print messages when learning rate is reduced
            factor=0.7,  # Factor by which the learning rate will be reduced
            min_lr=1e-4  # Lower bound on the learning rate
        )

        history = model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose, callbacks=[early_stopping, lr_scheduler])

        # Plotting training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Set integer ticks on the x-axis
        epochs = range(1, len(history.history['loss']) + 1)  # Integer epoch numbers
        plt.xticks(ticks=epochs)  # Set the ticks to integer epoch numbers

        plt.legend()
        plt.show()

        # visualize model architecture
        plot_model(model, to_file=f'/Users/kpolonsky/Downloads/TEST/Features/model_architecture_{i}.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)

        # Evaluate the model
        loss = model.evaluate(self.X_test_scaled, self.y_test)
        print(f"Test Loss: {loss}")

        # Make predictions
        self.y_pred = model.predict(self.X_test_scaled)
        self.y_pred = np.ravel(self.y_pred) #flatten multi-dimensional array into one-dimensional
        # Create a DataFrame
        df_res = pd.DataFrame({
            'code1': self.main_codes_test,
            'code': self.file_codes_test,
            'predicted_score': self.y_pred
        })

        # Save the DataFrame to a CSV file
        df_res.to_csv(f'/Users/kpolonsky/Downloads/TEST/Features/prediction_DL_{i}.csv', index=False)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        corr_coefficient, p_value = pearsonr(self.y_test, self.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")
        return mse


    def plot_results(self, model_name: Literal["svr", "rf", "knn-r", "gbr", "dl"], mse: float) -> None:
        # Plot results for many features
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, color='blue', edgecolor='k', alpha=0.7)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', linestyle='--')
        corr_coefficient, _ = pearsonr(self.y_test, self.y_pred)
        plt.text(
            0.05, 0.95,  # Coordinates in relative figure coordinates (0 to 1)
            f'Pearson Correlation: {corr_coefficient:.2f}, MSE: {mse:.6f}',  # Text with the coefficient
            transform=plt.gca().transAxes,  # Use axes coordinate system
            fontsize=12,
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
        plt.show()

    # def get_features_importances(self, model_name: str) -> None:
        if model_name == "rf" or model_name == "gbr":
            importances = self.regressor.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Plot feature importances
            plt.figure(figsize=(12, 11))
            plt.title(f'{title} Feature Importances')
            plt.bar(range(self.X.shape[1]), importances[indices], align='center')
            plt.xticks(range(self.X.shape[1]), self.X.columns[indices], rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.show()