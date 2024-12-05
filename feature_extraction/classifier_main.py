import numpy as np
from classes.regressor import Regressor
from scipy import stats

if __name__ == '__main__':

    auc_values = []
    for i in range(1):
        regressor = Regressor("./out/orthomam_all_w_balify_no_ancestors_67.csv", 0.2, mode=4,
                              predicted_measure='class_label', i=i)
        # auc = regressor.random_forest_classification(n_estimators=100, i=i)
        # auc = regressor.xgb_classification(i=i)
        # auc=regressor.catboost_classification(i=i)
        auc = regressor.dl_classifier(epochs=30, batch_size=128, validation_split=0.2, verbose=1, learning_rate=0.001, i=i)
        auc_values.append(auc)
    print(auc_values)
    mean_auc = np.mean(auc_values)
    # std_mse = np.std(mse_values, ddof=1)
    # benchmark_mse = 0.0001

    # Perform one-sample t-test
    # t_statistic, p_value = stats.ttest_1samp(mse_values, benchmark_mse)

    print(f"Mean AUC: {mean_auc}")
    # print(f"Standard Deviation of MSE: {std_mse}")
    # print(f"t-statistic: {t_statistic}")
    # print(f"p-value: {p_value}")

    # Check if p-value is less than significance level (e.g., 0.05)
    # alpha = 0.05
    # if p_value < alpha:
    #     print("The mean MSE is significantly different from the benchmark value.")
    # else:
    #     print("The mean MSE is not significantly different from the benchmark value.")


    # Compare standard deviation to a threshold if needed
    # threshold_std = 0.0001  # Define a threshold for acceptable variation
    # if std_mse < threshold_std:
    #     print("The MSE values are consistent with low variation.")
    # else:
    #     print("The MSE values show considerable variation.")