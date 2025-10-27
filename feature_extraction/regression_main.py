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
    for i in range(1):
        #configuration
        mode: int = 1
        true_score_name: str = 'dseq_from_true'

        # features_file: str = './out/ortho12_distant_features_260825.csv'
        # empirical = False

        features_file: str = "./out/balibase_features_with_foldmason_231025.csv"
        empirical = True

        scaler_type_labels = "rank"
        scaler_type_features = "rank"
        # scaler_type_labels = "zscore"
        # scaler_type_features = "zscore"

        # loss_fn = "hybrid_mse_ranknet_dynamic"
        # loss_fn = 'custom_mse'
        # loss_fn = "mse"
        loss_fn = 'approx_ndcg_loss'
        # loss_fn = "kendall_loss"
        # loss_fn = 'mse'
        # loss_fn = 'hybrid_mse_ranknet_loss'

        alpha = 0
        eps = 0
        # eps = 4e-5
        # alpha = 0.98
        # eps = 0.00004

        #run regressor
        regressor = log_function_run(Regressor, features_file=features_file,
                                     test_size=0.2,
                                     mode=mode,
                                     i=i, remove_correlated_features=False,
                                     empirical=empirical, scaler_type_features=scaler_type_features,
                                     scaler_type_labels=scaler_type_labels, true_score_name=true_score_name,
                                     explain_features_importance=True)


        mse, loss, val_loss, corr_coefficient, avg_per_msa_corr, top50_percentage, val_kendall = \
            (log_function_run(
                regressor.deep_learning,
                                    i=i, epochs=50, batch_size=32,
                                   learning_rate=0.0003,
                                   neurons=[0, 128, 32, 16],
                                    dropout_rate=0.2,
                                   l1=0.0001, l2=0.0001, top_k=0,
                                   mse_weight=0,
                                   ranking_weight=0, loss_fn=loss_fn,
                                   regularizer_name='l1_l2',
                                   batch_generation='custom', alpha=alpha, eps=eps))

        print(f"Trial {i}: Loss={loss}, Val_MSE={val_loss}, Corr={corr_coefficient}, Avg_MSA_Corr={avg_per_msa_corr}, Top50%={top50_percentage}, Val_Kendall={val_kendall}")
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

