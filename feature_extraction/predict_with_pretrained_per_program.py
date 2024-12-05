import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from typing import Literal
from scipy.stats import pearsonr
import visualkeras
import joblib

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
from tensorflow.keras.models import load_model

from classes.regressor import Regressor

def pick_me_for_all():
    pass

def predict(program):
    # regressor = Regressor("./out/orthomam_treebase_combined_features2.csv", 0.2, mode=2,
    #                       predicted_measure='msa_distance')

    predicted_measure = 'msa_distance'
    mode = 2
    df = pd.read_csv('./out/orthomam_all_w_balify_no_ancestors_67.csv')
    test_codes_source = '/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_all_w_balify_no_ancestors/DL1/prediction_DL_0_mode2_msa_distance.csv'
    df_codes = pd.read_csv(test_codes_source)
    unique_code1 = df_codes['code1'].unique()

    df = df[df['code1'].isin(unique_code1)]

    id_string = ''
    if program == 'MUSCLE':
        id_string = 'MUSCLE'
    elif program == 'BALIPHY':
        id_string = 'bali_phy'
    elif program == 'PRANK':
        id_string = 'PRANK'

    df = df[df['code'].str.contains(id_string, na=False)]

    # df.drop(columns=['sp_score_gap_e_norm.1'], inplace=True)
    df['code1'] = df['code1'].astype(str)
    # Check for missing values
    print("Missing values in each column:\n", df.isnull().sum())
    corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df['dpos_dist_from_true'])
    print(f"Pearson Correlation of SOP and dpos: {corr_coefficient1:.4f}\n",
          f"P-value of non-correlation: {p_value1:.6f}\n")

    # add normalized_rf
    df["normalized_rf"] = df['rf_from_true'] / (df['taxa_num'] - 1)
    df = df.dropna()

    if predicted_measure == 'msa_distance':
        true_score_name = "dpos_dist_from_true"
    elif predicted_measure == 'tree_distance':
        true_score_name = 'normalized_rf'

    y = df[true_score_name]

    # all features
    if mode == 1:
        X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1',
                                  'pypythia_msa_difficulty'])
    # all except 2SoP
    if mode == 2:
        X = df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
                     'sop_score', 'normalised_sop_score'])

    # only 2 features of SoP
    if mode == 3:
        X = df[['sop_score', 'normalised_sop_score']]

    # Get unique 'code1' values
    unique_code1 = df['code1'].unique()
    test_df = df

    # all features
    if mode == 1:
        X_test = test_df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1',
                     'pypythia_msa_difficulty'])

    if mode == 2:
        X_test = test_df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
                     'sop_score', 'normalised_sop_score'])

    # 2 sop features
    if mode == 3:
        X_test = test_df[['sop_score', 'normalised_sop_score']]

    # # load scaler used during the training
    # scaler = joblib.load(
    #     '/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/treebase_orthomam_PRANK_combined/DL_new_sigmoid_transformed2/scaler.pkl')
    # X_test_scaled = scaler.fit_transform(X_test)
    main_codes_test = test_df['code1']
    file_codes_test = test_df['code']

    corr_coefficient1, p_value1 = pearsonr(test_df['normalised_sop_score'], test_df['dpos_dist_from_true'])
    print(f"Pearson Correlation of SOP and dpos in the TEST set: {corr_coefficient1:.4f}\n",
          f"P-value of non-correlation: {p_value1:.4f}\n")

    # Set train and test Labels
    y_test = test_df[true_score_name]

    # Check the size of each set
    print(f"Test set size: {test_df.shape}")

    # load scaler used during the training
    # scaler = joblib.load(
    #     f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_wo_BaliPhy/DL1/scaler_{i}.pkl')
    scaler = joblib.load(
        f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_all_w_balify_no_ancestors/DL1/scaler_0.pkl')
    X_test_scaled = scaler.fit_transform(X_test)

    # Set train and test Labels
    y_test = test_df[true_score_name]

    # Check the size of each set
    print(f"Test set size (final): {X_test_scaled.shape}")

    print(np.any(np.isnan(X_test_scaled)))
    print(np.any(np.isinf(X_test_scaled)))
    print(np.any(np.isnan(y_test)))
    print(np.any(np.isinf(y_test)))

    # Load the saved model
    # model = load_model(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/treebase_orthomam_PRANK_combined/DL2/regressor_model_{i}_mode2_msa_distance.keras')
    # model = load_model(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/treeBase_4000/MSA_dist_RESULTS/DL7_all_except2/regressor_model_{i}_mode2_msa_distance.keras')
    model = load_model(
        f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_all_w_balify_no_ancestors/DL1/regressor_model_0_mode2_msa_distance.keras')
    # with open(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_w_BaliPhy/RF/random_forest_model_{i}.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred = np.ravel(y_pred)  # flatten multi-dimensional array into one-dimensional

    # Create a DataFrame
    df_res = pd.DataFrame({
        'code1': main_codes_test,
        'code': file_codes_test,
        'predicted_score': y_pred
    })

    # Save the DataFrame to a CSV file
    df_res.to_csv(f'./out/prediction_DL_pretrained_{id_string}_model_0_msa_distance.csv', index=False)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    corr_coefficient, p_value = pearsonr(y_test, y_pred)
    print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red',
             linestyle='--')
    corr_coefficient, _ = pearsonr(y_test, y_pred)
    plt.text(
        0.05, 0.95,  # Coordinates in relative figure coordinates (0 to 1)
        f'Pearson Correlation: {corr_coefficient:.2f}, MSE: {mse:.6f}',  # Text with the coefficient
        transform=plt.gca().transAxes,  # Use axes coordinate system
        fontsize=12,
        verticalalignment='top'
    )
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    title = "Deep learning"
    plt.title(f'{title}: Predicted vs. True Values')
    plt.grid(True)
    plt.savefig(fname=f'./out/regression_results_DL_pretrained_{id_string}_model0_msa_distance.png', format='png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    predict(program='MUSCLE')

