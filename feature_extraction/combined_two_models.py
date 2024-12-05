import numpy as np
from classes.regressor import Regressor
import pandas as pd
from scipy import stats
import joblib
import shap

if __name__ == '__main__':

    auc_values = []
    mse_values = []
    for i in range(1):
        regressor = Regressor("./out/orthomam_all_w_balify_no_ancestors_67.csv", 0.2, mode=1,
                              predicted_measure='class_label', i=i)
        auc = regressor.dl_classifier(epochs=50, batch_size=128, validation_split=0.2, verbose=1, learning_rate=0.001,
                                      i=i)
        auc_values.append(auc)
        binary_feature_train_df = pd.DataFrame(regressor.binary_feature.values, columns=['binary_feature'])
        X_train_scaled_df = pd.DataFrame(regressor.X_train_scaled, columns=regressor.X_train.columns)
        new_x_train_combined_df = pd.concat([X_train_scaled_df, binary_feature_train_df], axis=1)

        binary_feature_test_df = pd.DataFrame(regressor.y_prob, columns=['binary_feature'])
        X_test_scaled_df = pd.DataFrame(regressor.X_test_scaled, columns=regressor.X_test.columns)
        new_x_test_combined_df = pd.concat([X_test_scaled_df, binary_feature_test_df], axis=1)

        regressor.X_train_scaled = new_x_train_combined_df
        regressor.X_test_scaled = new_x_test_combined_df

        regressor.y_train = regressor.train_df['dpos_dist_from_true']
        regressor.y_test = regressor.test_df['dpos_dist_from_true']
        regressor.predicted_measure = 'msa_distance'

        mse = regressor.deep_learning(epochs=50, batch_size=128, validation_split=0.2, verbose=1, learning_rate=0.001, i=i)
        mse_values.append(mse)
        regressor.plot_results("dl", mse, i)

    print(auc_values)
    print(mse_values)
    mean_mse = np.mean(mse_values)
    print(f"Mean MSE: {mean_mse}")
