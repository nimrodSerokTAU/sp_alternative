import numpy as np
from classes.regressor_grouped import Regressor
import logging
import inspect
import pandas as pd

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
    regressor = log_function_run(Regressor, features_file="./out/orthomam_features_260225_with_NS_300Alt.csv",
                                     test_size=0.2,
                                     mode=1,
                                     predicted_measure='class_label', remove_correlated_features=False)

    auc = log_function_run(regressor.dl_classifier, epochs=30, batch_size=32, learning_rate=0.0001, dropout_rate=0.2, l1=0.00001, l2=0.00001, repeats=1, mixed_portion=0.2, threshold=0.55)
    regressor = log_function_run(Regressor, features_file="./out/orthomam_features_260225_with_NS_300Alt.csv",
                                 test_size=0.2,
                                 mode=1,
                                 predicted_measure='msa_distance', remove_correlated_features=False)
    mse = log_function_run(regressor.deep_learning, epochs=30, batch_size=32, learning_rate=0.0001, dropout_rate=0.2, l1=0.00001, l2=0.00001, repeats=2, mixed_portion=0.2, top_k=4, mse_weight=0, ranking_weight=50)
    print(f"MSE of regression: {mse}")
    print(f"AUC of classification: {auc}")


