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
from tensorflow.keras.regularizers import l2
from regressor import Regressor

regressor = Regressor("/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/orthomam_treebase_combined_features_w_balify2.csv", 0.2, mode=2, predicted_measure='msa_distance')

# Define a function to create the model
def create_model(optimizer='adam', learning_rate=0.001, dropout_rate=0.2, neurons=[64,32,16], kernel_init=GlorotUniform(), kernel_reg=l2(1e-4), activations=['leaky_relu', 'leaky_relu', 'leaky_relu']):
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
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

# Define possible activation functions
activation_functions = ['leaky_relu', 'relu', 'elu']
activation_combinations = list(itertools.product(activation_functions, repeat=3))

# Define possible activation functions
neurons = [32, 64, 128]
neuron_combinations = list(itertools.product(neurons, repeat=3))

# Define the grid of hyperparameters
param_grid = {
    'activations': activation_combinations,
    'batch_size': [8, 16, 32],
    'epochs': [30, 50],
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.2, 0.4],
    'neurons': neuron_combinations,
    'validation_split': 0.2,
    'learning_rate': [0.001, 0.0001, 0.00001]
}

# Manual hyperparameter tuning
best_score = float('inf')
best_params = None


# for batch_size in param_grid['batch_size']:
#     for activation_combo in activation_combinations:
#         for dropout_rate in param_grid['dropout_rate']:
#             for epochs in param_grid['epochs']:
#                 for learning_rate in param_grid['learning_rate']:
#                     for neurons_combo in param_grid['neurons']:
#                         try:
#                             model = create_model(optimizer='adam', learning_rate=learning_rate, dropout_rate=dropout_rate, neurons=neurons_combo, kernel_init=GlorotUniform(), kernel_reg=l2(1e-4), activations=activation_combo)
#                             early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#                             model.fit(regressor.X_train_scaled, regressor.y_train, validation_split=param_grid['validation_split'], batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[early_stopping])
#                             score = model.evaluate(regressor.X_test_scaled, regressor.y_test, verbose=0)
#                             print(
#                                 f"Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Optimizer: {param_grid['optimizer'][0]}, Learning rate: {learning_rate}, Validation split: {param_grid['validation_split']}, Score: {score}")
#
#                             with open('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/grid_search.txt', 'a') as f:
#                                 f.write(f"Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Optimizer: {param_grid['optimizer'][0]}, Learning rate: {learning_rate}, Validation split: {param_grid['validation_split']}, Score: {score}\n")
#
#                             if score < best_score:
#                                 best_score = score
#                                 best_params = {
#                                     'activation': activation_combo,
#                                     'batch_size': batch_size,
#                                     'epochs': epochs,
#                                     'dropout_rate': dropout_rate,
#                                     'neurons': neurons_combo,
#                                     'optimizer': {param_grid['optimizer'][0]},
#                                     'learning rate': {learning_rate},
#                                     'validation split': {param_grid['validation_split']}
#                                 }
#                                 with open(
#                                         '/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/grid_search.txt',
#                                         'a') as f:
#                                     f.write(
#                                         f"NEW BEST\n")
#                         except Exception as e:
#                             print(f'Failed with parameters: Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}. Error: {e}')
# print(f"Best Score: {best_score} using {best_params}")
for round in range(100):
    batch_size = random.choice(param_grid['batch_size'])
    activation_combo = random.choice(param_grid['activations'])
    dropout_rate =  random.choice(param_grid['dropout_rate'])
    epochs =  random.choice(param_grid['epochs'])
    # learning_rate = random.choice(param_grid['learning_rate'])
    learning_rate = 0.001
    neurons_combo =  random.choice(param_grid['neurons'])
    try:
        model = create_model(optimizer='adam', learning_rate=learning_rate, dropout_rate=dropout_rate, neurons=neurons_combo, kernel_init=GlorotUniform(), kernel_reg=l2(1e-4), activations=activation_combo)
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',  # Metric to monitor
            patience=3,  # Number of epochs with no improvement to wait before reducing the learning rate
            verbose=1,  # Print messages when learning rate is reduced
            factor=0.7,  # Factor by which the learning rate will be reduced
            min_lr=1e-5  # Lower bound on the learning rate
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(regressor.X_train_scaled, regressor.y_train, validation_split=param_grid['validation_split'], batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[early_stopping, lr_scheduler])
        score = model.evaluate(regressor.X_test_scaled, regressor.y_test, verbose=0)
        print(
            f"Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Optimizer: {param_grid['optimizer'][0]}, Learning rate: {learning_rate} and lr_scheduler, Validation split: {param_grid['validation_split']}, Score: {score}")

        with open('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/grid_search.txt', 'a') as f:
            f.write(f"Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Optimizer: {param_grid['optimizer'][0]}, Learning rate: {learning_rate} and lr_scheduler, Validation split: {param_grid['validation_split']}, Score: {score}\n")

        if score < best_score:
            best_score = score
            best_params = {
                'activation': activation_combo,
                'batch_size': batch_size,
                'epochs': epochs,
                'dropout_rate': dropout_rate,
                'neurons': neurons_combo,
                'optimizer': {param_grid['optimizer'][0]},
                'learning rate': {learning_rate},
                'validation split': {param_grid['validation_split']}
            }
            with open(
                    '/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/grid_search_new.txt',
                    'a') as f:
                f.write(
                    f"NEW BEST\n")
    except Exception as e:
        print(f'Failed with parameters: Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}. Error: {e}')
print(f"Best Score: {best_score} using {best_params}")