import pickle

import matplotlib
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
import shap
from predict_with_pretrained import use_test_from_origin


if __name__ == '__main__':
    X_test, test_df, true_score_name, main_codes_test, file_codes_test = use_test_from_origin(
        features_file='/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/balibase_features_73.csv',
        predictions_file='/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/Balilbase_mode4/DL_wo_k_mer_10_norm/prediction_DL_0_mode4_msa_distance.csv')
    scaler = joblib.load(
        f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/Balilbase_mode4/DL_wo_k_mer_10_norm/scaler_0_mode4_msa_distance.pkl')
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = X_test_scaled.astype('float64')
    X_test_scaled_with_names = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Set train and test Labels
    y_test = test_df[true_score_name]
    y_test = y_test.astype('float64')

    # Check the size of each set
    print(f"Test set size (final): {X_test_scaled.shape}")

    # shap.initjs()
    matplotlib.use('Agg')
    shap_values = joblib.load(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/Balilbase_mode4/DL_wo_k_mer_10_norm/shap_values__0_mode4_msa_distance.pkl')

    feature_names = [
        a + ": " + str(b) for a, b in zip(X_test.columns, np.abs(shap_values.values).mean(0).round(3))
    ]

    shap.summary_plot(shap_values, X_test_scaled_with_names, max_display=30, feature_names=feature_names)
    # shap.summary_plot(shap_values, X_test_scaled_with_names, max_display=30)
    plt.savefig('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/summary_plot.png', dpi=300,
                bbox_inches='tight')
    # plt.show()
    plt.close()

    shap.plots.waterfall(shap_values[0], max_display=30)
    plt.savefig('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/waterfall_0_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    shap.force_plot(shap_values[0], X_test_scaled[0], matplotlib=True, show=False)
    plt.savefig('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/force_0_plot.png')
    # plt.show()
    plt.close()

    shap.plots.bar(shap_values, max_display=30)
    plt.savefig('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/bar_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    # shap.plots.heatmap(shap_values)

    # indices = [5,18]
    # for i in indices:
    #     shap.plots.waterfall(shap_values[i], max_display=30)
    #     plt.savefig(f'/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/waterfall_{i}_plot.png', dpi=300, bbox_inches='tight')
    #     # plt.show()
    #     plt.close()
