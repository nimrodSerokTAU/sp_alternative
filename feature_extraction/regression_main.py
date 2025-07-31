import numpy as np
from classes.regressor import Regressor
# from classes.regressor_grouped import Regressor
from scipy import stats
import logging
import inspect
from classes.pick_me_trio import PickMeGameTrio
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
    for i in range(3):
        #configuration
        mode: int = 1
        # true_score_name: str = 'ssp_from_true'
        true_score_name: str = 'dseq_from_true'
        # true_score_name: str = 'dpos_from_true'
        # features_file: str = './out/ortho12_features_250725.csv'
        features_file: str = "./out/balibase_features_with_foldmason_220625.csv"

        #run regressor
        regressor = log_function_run(Regressor, features_file=features_file,
                                     test_size=0.2,
                                     mode=mode, i=i, remove_correlated_features=False,
                                     empirical=True, scaler_type="standard", true_score_name=true_score_name)
        # regressor = log_function_run(Regressor, features_file=features_file,
        #                              test_size=0.2,
        #                              mode=mode,
        #                              i=i, remove_correlated_features=False,
        #                              empirical=True, scaler_type="rank", true_score_name=true_score_name)

        # mse = log_function_run(regressor.deep_learning, i=i, epochs=50, batch_size=32, learning_rate=0.0003,
        #                        neurons=[0, 128, 32, 16], dropout_rate=0.2, l1=0.0001, l2=0.0001, top_k=4, mse_weight=1,
        #                        ranking_weight=5, loss_fn="custom_mse", regularizer_name='l1_l2', batch_generation='custom')
        mse = log_function_run(regressor.deep_learning, i=i, epochs=50, batch_size=32, learning_rate=0.0003,
                               neurons=[0, 128, 32, 16], dropout_rate=0.2, l1=0.0001, l2=0.0001, top_k=4, mse_weight=1,
                               ranking_weight=5, loss_fn="mse", regularizer_name='l1_l2',
                               batch_generation='standard')
        mse_values.append(mse)
        regressor.plot_results("dl", mse, i)

        # run pick-me
        pickme = log_function_run(PickMeGameTrio, features_file=features_file,
                                prediction_file=f'./out/prediction_DL_{i}_mode{mode}_{true_score_name}.csv',
                                true_score_name=true_score_name)
        pickme.run(i)
        pickme.summarize()
        pickme.save_to_csv(i)
        pickme.plot_results(i)
        pickme.plot_overall_results(i)

    print(mse_values)
    mean_mse = np.mean(mse_values)
    print(f"Mean MSE: {mean_mse}")

