import numpy as np
# from classes.regressor import Regressor
from classes.regressor_grouped import Regressor
# from classes.regressor_siamese import Regressor
# from classes.regressor_400_attention import Regressor
from scipy import stats
import logging
import inspect

# Set up logging to write to a file
logging.basicConfig(filename='./out/function_log.txt', level=logging.INFO)

def log_function_run(func, *args, **kwargs):
    function_name = func.__name__
    parameters = ', '.join([f'{key}={value}' for key, value in kwargs.items()])
    parameters += ', '.join([str(arg) for arg in args])

    source_code = inspect.getsource(func)

    logging.info(f'Running function: {function_name} with parameters: {parameters}')
    # logging.info(f'Running function: {function_name} with parameters: {parameters}')
    logging.info(f'Source code of {function_name}:\n{source_code}')
    return func(*args, **kwargs)

if __name__ == '__main__':

    mse_values = []
    for i in range(1):
        # regressor = Regressor("./out/orthomam_features_150225.csv", 0.2, mode=3,
        #                       predicted_measure='msa_distance', i=i, remove_correlated_features=False)
        regressor = log_function_run(Regressor,"./out/orthomam_features_150225.csv", 0.2, mode=1,
                              predicted_measure='msa_distance', i=i, remove_correlated_features=False)
        # mse = regressor.random_forest(i=i, n_estimators=500)
        mse = log_function_run(regressor.deep_learning, i=i, epochs=50, batch_size=64, learning_rate=0.0001, dropout_rate=0.2, l1=0.00001, l2=0.00001)
        # mse = log_function_run(regressor.random_forest, n_estimators=500, i=i)
        # mse = regressor.deep_learning(i=i, epochs=50, batch_size=32, learning_rate=0.01, dropout=0.2, l1=1e-5, l2=1e-5, undersampling=False)
        # mse = regressor.deep_learning(i=i, epochs=50, batch_size=32, learning_rate=1e-3, undersampling=False)
        mse_values.append(mse)
        # regressor.plot_results("rf", mse, i)
        regressor.plot_results("dl", mse, i)
    print(mse_values)
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values, ddof=1)
    # benchmark_mse = 0.0001

    # Perform one-sample t-test
    # t_statistic, p_value = stats.ttest_1samp(mse_values, benchmark_mse)

    print(f"Mean MSE: {mean_mse}")
    print(f"Standard Deviation of MSE: {std_mse}")
    # print(f"t-statistic: {t_statistic}")
    # print(f"p-value: {p_value}")


