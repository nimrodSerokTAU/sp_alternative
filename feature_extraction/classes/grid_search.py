import itertools
import random
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Dense, Dropout, LeakyReLU, Activation, BatchNormalization, Input, ELU
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2, l1, l1_l2
from regressor import Regressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import backend as K
from feature_extraction.classes.batch_generator import BatchGenerator

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError as e:
        print(e)

regressor = Regressor(features_file="../out/orthomam_features_w_xtr_NS_KP_290425.parquet",
                                     test_size=0.2,
                                     mode=3,
                                     predicted_measure='msa_distance', i=0, remove_correlated_features=False, empirical=False, scaler_type="rank")



def _get_regularizer_info(reg):
    if isinstance(reg, l1_l2):
        return {
            'type': 'L1L2',
            'l1': float(reg.l1) if reg.l1 else 0.0,
            'l2': float(reg.l2) if reg.l2 else 0.0
        }
    elif isinstance(reg, l1):
        return {
            'type': 'L1',
            'l1': float(reg.l1),
            'l2': 0.0
        }
    elif isinstance(reg, l2):
        return {
            'type': 'L2',
            'l1': 0.0,
            'l2': float(reg.l2)
        }
    else:
        return {
            'type': None,
            'l1': 0.0,
            'l2': 0.0
        }
def rank_loss(y_true: tf.Tensor, y_pred: tf.Tensor, top_k: int) -> tf.Tensor:
    tf.compat.v1.enable_eager_execution()

    @tf.function
    def print_func(y_true, y_pred, paired, sorted_paired):
        tf.print("y_true:", y_true)
        tf.print("y_pred:", y_pred)
        tf.print("y_paired:", paired)
        tf.print("sorted_paired:", sorted_paired)
        return y_true, y_pred, paired, sorted_paired

    paired = tf.stack([y_true, y_pred], axis=1)

    sorted_indices = tf.argsort(paired[:, 0], axis=0, direction='ASCENDING')
    sorted_paired = tf.gather(paired, sorted_indices, axis=0)

    true_top_k = sorted_paired[:, :top_k, 0]
    pred_top_k = sorted_paired[:, :top_k, 1]
    rank_diff = K.mean(K.square(K.cast(true_top_k - pred_top_k, dtype=tf.float32)))

    return rank_diff


# Combine MSE loss with rank-based loss
def mse_with_rank_loss(y_true: tf.Tensor, y_pred: tf.Tensor, top_k: int = 4, mse_weight: float = 1,
                       ranking_weight: float = 0.3) -> tf.Tensor:
    mse_loss = K.mean(K.square(K.cast(y_true - y_pred, dtype=tf.float32)))  # MSE loss
    top_k_rank_loss = rank_loss(y_true, y_pred, top_k)
    mse_weight = tf.cast(mse_weight, dtype=tf.float32)
    ranking_weight = tf.cast(ranking_weight, dtype=tf.float32)
    top_k_rank_loss = tf.cast(top_k_rank_loss, dtype=tf.float32)
    total_loss = mse_weight * mse_loss + ranking_weight * top_k_rank_loss

    return total_loss

# Define a function to create the model
def create_model(optimizer='adam', learning_rate=0.001, dropout_rate=0.2, neurons=[128, 64, 16], kernel_init=GlorotUniform(), kernel_reg=l2(0.001), activations=['leaky_relu', 'leaky_relu', 'leaky_relu'], loss_func = 'mean_squared_error', top_k=4, mse_weight=1,
                                                                     ranking_weight=20):
    model = Sequential()
    model.add(Input(shape=(regressor.X_train_scaled.shape[1],)))

    # first hidden
    model.add(Dense(neurons[0], kernel_initializer=kernel_init, kernel_regularizer=kernel_reg))

    # Apply the first activation function
    if activations[0] == 'leaky_relu':
        model.add(LeakyReLU(negative_slope=0.01))
    elif activations[0] == 'elu':
        model.add(ELU())
    else:
        model.add(Activation(activations[0]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))  # Dropout for regularization

    # second hidden
    model.add(Dense(neurons[1], kernel_initializer=kernel_init, kernel_regularizer=kernel_reg))

    if activations[1] == 'leaky_relu':
        model.add(LeakyReLU(negative_slope=0.01))
    elif activations[1] == 'elu':
        model.add(ELU())
    else:
        model.add(Activation(activations[1]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))  # Dropout for regularization

    # third hidden
    model.add(Dense(neurons[2], kernel_initializer=kernel_init, kernel_regularizer=kernel_reg))

    if activations[2] == 'leaky_relu':
        model.add(LeakyReLU(negative_slope=0.01))
    elif activations[2] == 'elu':
        model.add(ELU())
    else:
        model.add(Activation(activations[2]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))  # Dropout for regularization

    # model.add(Dense(1, activation='softplus'))  # exponential ensures no negative values
    model.add(Dense(1, activation='sigmoid'))  # exponential ensures no negative values

    optimizer = Adam(learning_rate=learning_rate)
    if loss_func == 'mean_squared_error':
        model.compile(optimizer=optimizer, loss='mean_squared_error')
    elif loss_func == 'custom':
        model.compile(optimizer=optimizer,
                      loss=lambda y_true, y_pred: mse_with_rank_loss(y_true, y_pred, top_k=top_k, mse_weight=mse_weight,
                                                                     ranking_weight=ranking_weight))

    return model

# Define possible activation functions
activation_functions = ['leaky_relu', 'relu', 'elu']
activation_combinations = list(itertools.product(activation_functions, repeat=3))


# Define possible activation functions
neurons = [16, 32, 64, 128]
neuron_combinations = list(itertools.product(neurons, repeat=3))

# Define the grid of hyperparameters
param_grid = {
    'activations': activation_combinations,
    'batch_size': [8, 16, 32, 64],
    'epochs': [30, 50],
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.2, 0.4],
    'neurons': neuron_combinations,
    'validation_split': 0.2,
    'learning_rate': [0.001, 0.0001, 0.00001],
    'regularizer': [l2(0.001), l1_l2(l1=0.001, l2=0.001), l2(1e-4), l1(1e-4), l1(0.001), l1_l2(l1=1e-4, l2=1e-4)],
    'top_k': [1, 4, 10, 20],
    'ranking_weight': [0.0, 0.2, 1, 2, 10, 20],
    'loss_func': ['mean_squared_error', 'custom']
    }

# Manual hyperparameter tuning
# best_score = float('inf')
best_score = float(0)
best_params = None


for round in range(150):
    batch_size = random.choice(param_grid['batch_size'])
    activation_combo = random.choice(param_grid['activations'])
    dropout_rate =  random.choice(param_grid['dropout_rate'])
    epochs =  random.choice(param_grid['epochs'])
    learning_rate = random.choice(param_grid['learning_rate'])
    # learning_rate = 0.001
    neurons_combo =  random.choice(param_grid['neurons'])
    regularizer = random.choice(param_grid['regularizer'])
    loss_func = random.choice(param_grid['loss_func'])
    top_k = random.choice(param_grid['top_k'])
    ranking_weight = random.choice(param_grid['ranking_weight'])

    reg_info = _get_regularizer_info(regularizer)

    try:
        model = create_model(optimizer='adam', learning_rate=learning_rate, dropout_rate=dropout_rate, neurons=neurons_combo, kernel_init=GlorotUniform(), kernel_reg=regularizer, activations=activation_combo, loss_func=loss_func, top_k=top_k, mse_weight=1,
                                                                     ranking_weight=ranking_weight)
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',  # to monitor
            patience=3,  # number of epochs with no improvement before reducing the learning rate
            verbose=1,
            factor=0.5,  # factor by which the learning rate will be reduced
            min_lr=1e-6,  # lower bound on the learning rate
            min_delta=1e-5  # the threshold for val loss improvement - to identify the plateau
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, min_delta=1e-5)
        callbacks = [
            early_stopping,
            lr_scheduler
        ]

        unique_train_codes = regressor.main_codes_train.unique()
        train_msa_ids, val_msa_ids = train_test_split(unique_train_codes, test_size=0.2)
        print(f"the training set is: {train_msa_ids} \n")
        print(f"the validation set is: {val_msa_ids} \n")
        batch_generator = BatchGenerator(features=regressor.X_train_scaled, true_labels=regressor.y_train,
                                         true_msa_ids=regressor.main_codes_train, train_msa_ids=train_msa_ids,
                                         val_msa_ids=val_msa_ids, aligners=regressor.aligners_train, batch_size=batch_size,
                                         validation_split=0.2, is_validation=False, repeats=1,
                                         mixed_portion=0, per_aligner=False,
                                         features_w_names=regressor.train_df)

        val_generator = BatchGenerator(features=regressor.X_train_scaled, true_labels=regressor.y_train,
                                       true_msa_ids=regressor.main_codes_train, train_msa_ids=train_msa_ids,
                                       val_msa_ids=val_msa_ids, aligners=regressor.aligners_train,
                                       batch_size=batch_size, validation_split=0.2, is_validation=True,
                                       repeats=1, mixed_portion=0, per_aligner=False,
                                       features_w_names=regressor.train_df)

        history = model.fit(batch_generator, epochs=epochs, validation_data=val_generator, verbose=1,
                                    callbacks=callbacks)
        loss = model.evaluate(regressor.X_test_scaled, regressor.y_test, verbose=0)

        regressor.y_pred = model.predict(regressor.X_test_scaled)
        regressor.y_pred = np.ravel(regressor.y_pred)  # flatten multi-dimensional array into one-dimensional
        regressor.y_pred = regressor.y_pred.astype('float64')
        mse = mean_squared_error(regressor.y_test, regressor.y_pred)
        print(f"Mean Squared Error: {mse:.4f}\n")
        corr_coefficient, p_value = pearsonr(regressor.y_test, regressor.y_pred)
        print(f"Pearson Correlation: {corr_coefficient:.4f}\n", f"P-value of non-correlation: {p_value:.4f}\n")

        print(
            f"Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Optimizer: {param_grid['optimizer'][0]}, Learning rate: {learning_rate} and lr_scheduler, Validation split: {param_grid['validation_split']}, Regularizer: {reg_info}, Loss_func: {loss_func}, Top_k: {top_k}, Ranking_Weight: {ranking_weight}, Loss: {loss}, MSE: {mse}, Pearson: {corr_coefficient}, p-value: {p_value}\n")

        with open('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/grid_search.txt', 'a') as f:
            f.write(f"Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Optimizer: {param_grid['optimizer'][0]}, Learning rate: {learning_rate} and lr_scheduler, Validation split: {param_grid['validation_split']}, Regularizer: {reg_info}, Loss_func: {loss_func}, Top_k: {top_k}, Ranking_Weight: {ranking_weight}, Loss: {loss}, MSE: {mse}, Pearson: {corr_coefficient}, p-value: {p_value}\n")


        if corr_coefficient > best_score:
            best_score = corr_coefficient
            best_params = {
                'activation': activation_combo,
                'batch_size': batch_size,
                'epochs': epochs,
                'dropout_rate': dropout_rate,
                'neurons': neurons_combo,
                'optimizer': param_grid['optimizer'][0],
                'learning rate': learning_rate,
                'validation split': param_grid['validation_split'],
                'regularizer': reg_info,
                'loss_func': loss_func,
                'top_k': top_k,
                'ranking_weight': ranking_weight
            }
            with open(
                    '/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/grid_search.txt',
                    'a') as f:
                f.write(
                    f"NEW BEST\n")
    except Exception as e:
        print(f'Failed with parameters: Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Regularizer: {reg_info}, Loss_func: {loss_func}, top_k: {top_k}, ranking_weight: {ranking_weight}. Error: {e}')
print(f"Best Score: {best_score} using {best_params}")
