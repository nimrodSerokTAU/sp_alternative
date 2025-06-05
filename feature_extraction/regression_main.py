import numpy as np
from classes.regressor import Regressor
# from classes.regressor_grouped import Regressor
from scipy import stats
import logging
import inspect

logging.basicConfig(filename='./out/function_log.txt', level=logging.INFO)

def log_function_run(func, *args, **kwargs):
    function_name = func.__name__
    parameters = ', '.join([f'{key}={value}' for key, value in kwargs.items()])
    parameters += ', '.join([str(arg) for arg in args])

    source_code = inspect.getsource(func)

    logging.info(f'Running function: {function_name} with parameters: {parameters}')
    logging.info(f'Source code of {function_name}:\n{source_code}')
    return func(*args, **kwargs)

if __name__ == '__main__':

    mse_values = []
    for i in range(5):
        # regressor = log_function_run(Regressor, features_file="./out/balibase_features_ALL_new_dpos_with_foldmason_200525.parquet",
        #                              test_size=0.2,
        #                              mode=3,
        #                              predicted_measure='msa_distance', i=i, remove_correlated_features=False,
        #                              empirical=True, scaler_type="rank")
        # regressor = log_function_run(Regressor,
        #                              features_file="./out/old_features_files/balibase_ALL_features_260125_w_foldmason.csv",
        #                              test_size=0.2,
        #                              mode=3,
        #                              predicted_measure='msa_distance', i=i, remove_correlated_features=False,
        #                              empirical=True, scaler_type="standard")

        regressor = log_function_run(Regressor, features_file="./out/orthomam_features_w_xtr_NS_KP_290425.parquet",
                                     test_size=0.2,
                                     mode=3,
                                     predicted_measure='msa_distance', i=i, remove_correlated_features=False,
                                     empirical=False, scaler_type="rank")

        # regressor = log_function_run(Regressor, features_file="./out/oxbench_features_new_dpos_with_foldmason_210525.csv",
        #                              test_size=0.2,
        #                              mode=3,
        #                              predicted_measure='msa_distance', i=i, remove_correlated_features=False,
        #                              empirical=True, scaler_type="rank")

        mse = log_function_run(regressor.deep_learning, i=i, epochs=50, batch_size=32, learning_rate=0.0001, neurons =[128, 32, 16],
                               dropout_rate=0.2, l1=0.001, l2=0.001, repeats=1, mixed_portion=0, per_aligner=None, top_k=4,
                               mse_weight=1, ranking_weight=1, loss_fn="custom_mse", regularizer_name='l1_l2', batch_generation='custom')
        mse_values.append(mse)
        regressor.plot_results("dl", mse, i)
    print(mse_values)
    mean_mse = np.mean(mse_values)
    # std_mse = np.std(mse_values, ddof=1)

    print(f"Mean MSE: {mean_mse}")
    # print(f"Standard Deviation of MSE: {std_mse}")

