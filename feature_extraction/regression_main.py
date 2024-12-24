import numpy as np
from classes.regressor import Regressor
from scipy import stats

if __name__ == '__main__':

    mse_values = []
    for i in range(1):
        regressor = Regressor("./out/orthomam_w2new_features.csv", 0.2, mode=1,
                              predicted_measure='msa_distance', i=i)
        # regressor = Regressor("./out/TreeBASE_3M_features.csv", 0.2, mode=2,
        #                       predicted_measure='msa_distance', i=i)
        # regressor = Regressor("./out/OrthoMaM_500K_features.csv", 0.2, mode=2,
        #                       predicted_measure='msa_distance', i=i)
        # regressor = Regressor("./out/orthomam_w_balify_features.csv", 0.2, mode=2,
        #                       predicted_measure='msa_distance', i=i)
        # regressor = Regressor("./out/orthomam_treebase_combined_features_w_balify.csv", 0.2, mode=2,
        #                       predicted_measure='msa_distance', i=i)
        # regressor = Regressor("./out/balibase_features_full_74.csv", 0.2, mode=2,
        #                       predicted_measure='msa_distance', i=i)
        # regressor = Regressor("./out/orthomam_treebase_combined_features2.csv", 0.2, mode=2, predicted_measure='msa_distance', i=i)
        # regressor = Regressor("/Users/kpolonsky/Downloads/BaliBase4/balibase_features.csv", 0.2, mode=2,
        #                       predicted_measure='msa_distance', i=i)
        # mse = regressor.random_forest(i=i)
        mse = regressor.deep_learning(i=i, epochs=50, batch_size=32, learning_rate=1e-4, undersampling = False)
        # mse = regressor.deep_learning_with_attention(i=i, epochs=50, batch_size=32, undersampling = False)
        mse_values.append(mse)
        # regressor.plot_results("rf", mse, i)
        regressor.plot_results("dl", mse, i)
    print(mse_values)
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values, ddof=1)
    benchmark_mse = 0.0001

    # Perform one-sample t-test
    t_statistic, p_value = stats.ttest_1samp(mse_values, benchmark_mse)

    print(f"Mean MSE: {mean_mse}")
    print(f"Standard Deviation of MSE: {std_mse}")
    print(f"t-statistic: {t_statistic}")
    print(f"p-value: {p_value}")

    # Check if p-value is less than significance level (e.g., 0.05)
    alpha = 0.05
    if p_value < alpha:
        print("The mean MSE is significantly different from the benchmark value.")
    else:
        print("The mean MSE is not significantly different from the benchmark value.")


    # Compare standard deviation to a threshold if needed
    threshold_std = 0.0001  # Define a threshold for acceptable variation
    if std_mse < threshold_std:
        print("The MSE values are consistent with low variation.")
    else:
        print("The MSE values show considerable variation.")