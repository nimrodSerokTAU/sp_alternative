import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from typing import Literal
from scipy.stats import pearsonr
import joblib

import pydot
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


def construct_test_set(file, mode=1):
    predicted_measure = 'msa_distance'
    df = pd.read_csv(file)
    # df.drop(columns=['sp_score_gap_e_norm.1'], inplace=True)
    df['code1'] = df['code1'].astype(str)
    # Check for missing values
    print("Missing values in each column:\n", df.isnull().sum())
    corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df['dpos_dist_from_true'])
    print(f"Pearson Correlation of SOP and dpos: {corr_coefficient1:.4f}\n",
          f"P-value of non-correlation: {p_value1:.6f}\n")

    # add normalized_rf
    df["normalized_rf"] = df['rf_from_true'] / (df['taxa_num'] - 1)
    df["class_label"] = np.where(df['dpos_dist_from_true'] <= 0.02, 0, 1)
    df["class_label2"] = np.where(df['dpos_dist_from_true'] <= 0.1, 0, np.where(df['dpos_dist_from_true'] <= 0.5, 1, 2))
    df = df.dropna()
    if 'sp_score_gap_e_norm.1' in df.columns:
        df = df.drop(columns=['sp_score_gap_e_norm.1'])

    if predicted_measure == 'msa_distance':
        true_score_name = "dpos_dist_from_true"
    elif predicted_measure == 'tree_distance':
        true_score_name = 'normalized_rf'

    y = df[true_score_name]

    # all features
    if mode == 1:
        X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2'])
    # all except 2SoP
    # if mode == 2:
    #     X = df.drop(
    #         columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
    #                  'sop_score', 'normalised_sop_score','class_label', 'number_of_gap_segments', 'number_of_mismatches'])
    if mode == 2:
        X = df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
                     'sop_score', 'normalised_sop_score','class_label'])

    # only 2 features of SoP
    if mode == 3:
        X = df[['k_mer_10_norm', 'entropy_mean', 'constant_sites_pct']]

    if mode == 4:  # test removing features
        X = df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'class_label2', 'code', 'code1',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'sp_score_gap_e_norm','single_char_count', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len']
        )

    # Get unique 'code1' values
    unique_code1 = df['code1'].unique()
    test_df = df

    # all features
    if mode == 1:
        X_test = test_df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2'])

    if mode == 2:
        # X_test = test_df.drop(
        #     columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
        #              'sop_score', 'normalised_sop_score', 'class_label', 'number_of_gap_segments', 'number_of_mismatches'])
        X_test = test_df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
                     'sop_score', 'normalised_sop_score', 'class_label'])

    # 2 sop features
    if mode == 3:
        X_test = test_df[['k_mer_10_norm', 'entropy_mean', 'constant_sites_pct']]

    if mode == 4:
        X_test = test_df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'class_label2', 'code', 'code1',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'sp_score_gap_e_norm','single_char_count', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len']
        )

    # # load scaler used during the training
    # scaler = joblib.load(
    #     '/Users/kpolonsky/Documents/sp_alternative/dl_model/out/treebase_orthomam_PRANK_combined/DL_new_sigmoid_transformed2/scaler.pkl')
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

    return X_test, test_df, true_score_name, main_codes_test, file_codes_test


def use_test_from_origin(features_file, predictions_file, mode=1, portion=1):
    predicted_measure = 'msa_distance'
    # mode = 1

    df = pd.read_csv(features_file)
    # to make sure that all dataset codes are read as strings and not integers
    df['code1'] = df['code1'].astype(str)
    # Check for missing values
    print("Missing values in each column:\n", df.isnull().sum())
    corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df['dpos_dist_from_true'])
    print(f"Pearson Correlation of SOP and dpos: {corr_coefficient1:.4f}\n",
          f"P-value of non-correlation: {p_value1:.6f}\n")

    # add normalized_rf
    df["normalized_rf"] = df['rf_from_true'] / (df['taxa_num'] - 1)
    df["class_label"] = np.where(df['dpos_dist_from_true'] <= 0.010, 0, 1)
    df["class_label2"] = np.where(df['dpos_dist_from_true'] <= 0.01, 0, np.where(df['dpos_dist_from_true'] <= 0.05, 1, 2))

    # Handle missing values (if any)
    # Example: Filling missing values with the mean (for numerical columns)
    # df['orig_tree_ll'] = df['orig_tree_ll'].fillna(df['orig_tree_ll'].mean())
    # remove NaNs?
    df = df.dropna()

    if predicted_measure == 'msa_distance':
        true_score_name = "dpos_dist_from_true"
    elif predicted_measure == 'tree_distance':
        # true_score_name = "rf_from_true"
        true_score_name = 'normalized_rf'

    y = df[true_score_name]

    # all features
    if mode == 1:
        X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2'])

    # all features except 2 features of SoP
    if mode == 2:
        X = df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
                     'class_label',
                     'sop_score', 'normalised_sop_score'])

    # only 2 features of SoP
    if mode == 3:
        X = df[['k_mer_10_norm', 'entropy_mean', 'constant_sites_pct']]

    if mode == 4:  # test removing features
        X = df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'class_label2', 'code', 'code1',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'sp_score_gap_e_norm','single_char_count', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len']
        )

    # Split the unique 'code1' into training and test sets
    df_codes = pd.read_csv(predictions_file)
    test_code1 = df_codes['code1'].unique()

    test_code1 = np.random.choice(test_code1, size=int(len(test_code1) * portion), replace=False)

    print(f"the testing set is: {test_code1} \n")

    # Create training and test DataFrames by filtering based on 'code1'
    test_df = df[df['code1'].isin(test_code1)]

    # all features
    if mode == 1:
        X_test = test_df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty', 'class_label', 'class_label2'])

    if mode == 2:
        X_test = test_df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
                     'sop_score', 'class_label', 'normalised_sop_score'])

    # 2 sop features
    if mode == 3:
        X_test = test_df[['k_mer_10_norm', 'entropy_mean', 'constant_sites_pct']]

    if mode == 4:
        X_test = test_df.drop(
            columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'class_label', 'class_label2', 'code', 'code1',
                         'pypythia_msa_difficulty', 'normalised_sop_score', 'entropy_median',
                         'entropy_pct_25', 'entropy_min', 'entropy_max', 'bl_25_pct', 'bl_75_pct', 'var_bl',
                         'skew_bl', 'kurtosis_bl', 'bl_max', 'bl_min','gaps_len_two',
            'gaps_len_three', 'gaps_len_three_plus', 'gaps_1seq_len1',
            'gaps_2seq_len1', 'gaps_all_except_1_len1', 'gaps_1seq_len2', 'gaps_2seq_len2',
            'gaps_all_except_1_len2', 'gaps_1seq_len3', 'gaps_2seq_len3', 'gaps_all_except_1_len3',
            'gaps_1seq_len3plus', 'gaps_2seq_len3plus', 'gaps_all_except_1_len3plus', 'sp_score_gap_e_norm', 'sp_score_gap_e_norm','single_char_count', 'double_char_count','k_mer_10_max',  'k_mer_10_pct_95', 'k_mer_10_pct_90', 'k_mer_10_top_10_norm',
            'k_mer_20_max', 'k_mer_20_mean', 'k_mer_20_var', 'k_mer_20_pct_95', 'k_mer_20_pct_90', 'k_mer_20_top_10_norm', 'median_bl', 'num_cols_2_gaps', 'num_cols_all_gaps_except1', 'seq_min_len']
        )

    main_codes_test = test_df['code1']
    file_codes_test = test_df['code']

    corr_coefficient1, p_value1 = pearsonr(test_df['normalised_sop_score'], test_df['dpos_dist_from_true'])
    print(f"Pearson Correlation of SOP and dpos in the TEST set: {corr_coefficient1:.4f}\n",
          f"P-value of non-correlation: {p_value1:.4f}\n")

    # Check the size of each set
    print(f"Test set size: {test_df.shape}")

    return X_test, test_df, true_score_name, main_codes_test, file_codes_test


if __name__ == '__main__':

    # regressor = Regressor("./out/orthomam_treebase_combined_features2.csv", 0.2, mode=2,
    #                       predicted_measure='msa_distance')

    # predicted_measure = 'msa_distance'
    # mode = 2
    # df = pd.read_csv('./out/balibase_features.csv')
    # # df.drop(columns=['sp_score_gap_e_norm.1'], inplace=True)
    # df['code1'] = df['code1'].astype(str)
    # # Check for missing values
    # print("Missing values in each column:\n", df.isnull().sum())
    # corr_coefficient1, p_value1 = pearsonr(df['normalised_sop_score'], df['dpos_dist_from_true'])
    # print(f"Pearson Correlation of SOP and dpos: {corr_coefficient1:.4f}\n",
    #       f"P-value of non-correlation: {p_value1:.6f}\n")
    #
    # # add normalized_rf
    # df["normalized_rf"] = df['rf_from_true'] / (df['taxa_num'] - 1)
    # df = df.dropna()
    #
    # if predicted_measure == 'msa_distance':
    #     true_score_name = "dpos_dist_from_true"
    # elif predicted_measure == 'tree_distance':
    #     true_score_name = 'normalized_rf'
    #
    # y = df[true_score_name]
    #
    # # all features
    # if mode == 1:
    #     X = df.drop(columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1',
    #                               'pypythia_msa_difficulty'])
    # # all except 2SoP
    # if mode == 2:
    #     X = df.drop(
    #         columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
    #                  'sop_score', 'normalised_sop_score'])
    #
    # # only 2 features of SoP
    # if mode == 3:
    #     X = df[['sop_score', 'normalised_sop_score']]
    #
    # # Get unique 'code1' values
    # unique_code1 = df['code1'].unique()
    # test_df = df
    #
    # # all features
    # if mode == 1:
    #     X_test = test_df.drop(
    #         columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1',
    #                  'pypythia_msa_difficulty'])
    #
    # if mode == 2:
    #     X_test = test_df.drop(
    #         columns=['dpos_dist_from_true', 'rf_from_true', 'normalized_rf', 'code', 'code1', 'pypythia_msa_difficulty',
    #                  'sop_score', 'normalised_sop_score'])
    #
    # # 2 sop features
    # if mode == 3:
    #     X_test = test_df[['sop_score', 'normalised_sop_score']]
    #
    # # # load scaler used during the training
    # # scaler = joblib.load(
    # #     '/Users/kpolonsky/Documents/sp_alternative/dl_model/out/treebase_orthomam_PRANK_combined/DL_new_sigmoid_transformed2/scaler.pkl')
    # # X_test_scaled = scaler.fit_transform(X_test)
    # main_codes_test = test_df['code1']
    # file_codes_test = test_df['code']
    #
    # corr_coefficient1, p_value1 = pearsonr(test_df['normalised_sop_score'], test_df['dpos_dist_from_true'])
    # print(f"Pearson Correlation of SOP and dpos in the TEST set: {corr_coefficient1:.4f}\n",
    #       f"P-value of non-correlation: {p_value1:.4f}\n")
    #
    # # Set train and test Labels
    # y_test = test_df[true_score_name]
    #
    # # Check the size of each set
    # print(f"Test set size: {test_df.shape}")

    # X_test, test_df, true_score_name, main_codes_test, file_codes_test = construct_test_set("")

    n = 1
    for i in range(n):
        # X_test, test_df, true_score_name, main_codes_test, file_codes_test = use_test_from_origin(
        #     features_file='/Users/kpolonsky/Documents/sp_alternative/dl_model/out/orthomam_all_w_balify_no_ancestors_67.csv',
        #     predictions_file=f'/Users/kpolonsky/Documents/sp_alternative/dl_model/out/orthomam_all_w_balify_no_ancestors/DL1/prediction_DL_{i}_mode2_msa_distance.csv')
        # load scaler used during the training
        X_test, test_df, true_score_name, main_codes_test, file_codes_test = construct_test_set(
            "/dl_model/out/orthomam_features_251224.csv", mode=1)

        scaler = joblib.load(
            f'/dl_model/out/Ensemble/DL2_w_new_features/scaler_{i}_mode1_msa_distance.pkl')
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = X_test_scaled.astype('float64')
        X_test_scaled_with_names = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        # Set train and test Labels
        y_test = test_df[true_score_name]
        y_test = y_test.astype('float64')

        # Check the size of each set
        print(f"Test set size (final): {X_test_scaled.shape}")

        print(np.any(np.isnan(X_test_scaled)))
        print(np.any(np.isinf(X_test_scaled)))
        print(np.any(np.isnan(y_test)))
        print(np.any(np.isinf(y_test)))

        # Load the saved model
        model = load_model(
            f'/Users/kpolonsky/Documents/sp_alternative/dl_model/out/Ensemble/DL2_w_new_features/regressor_model_{i}_mode1_msa_distance.keras')
        # with open(f'/Users/kpolonsky/Documents/sp_alternative/dl_model/out/orthomam_w_BaliPhy/RF/random_forest_model_{i}.pkl', 'rb') as f:
        #     model = pickle.load(f)
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred = np.ravel(y_pred)  # flatten multi-dimensional array into one-dimensional

        # # try to explain the features
        # explainer = shap.Explainer(model, X_test_scaled_with_names)
        # shap_values = explainer(X_test_scaled_with_names)
        # joblib.dump(explainer,
        #             '/Users/kpolonsky/Documents/sp_alternative/dl_model/out/explainer_pretrained_model.pkl')
        # joblib.dump(shap_values,
        #             '/Users/kpolonsky/Documents/sp_alternative/dl_model/out/shap_values_pretrained_model.pkl')

        # Create a DataFrame
        df_res = pd.DataFrame({
            'code1': main_codes_test,
            'code': file_codes_test,
            'predicted_score': y_pred
        })

        # Save the DataFrame to a CSV file
        df_res.to_csv(f'./out/prediction_DL_pretrained_model{i}_msa_distance.csv', index=False)

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
        plt.savefig(fname=f'./out/regression_results_DL_pretrained_model{i}_msa_distance.png', format='png')
        plt.show()
        plt.close()
