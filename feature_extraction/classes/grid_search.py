import itertools
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

regressor = Regressor("/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/500K_features.csv", 0.2, mode=1, predicted_measure='msa_distance')

# Define a function to create the model
def create_model(optimizer='adam', dropout_rate=0.2, neurons=[64,32,16], kernel_init=GlorotUniform(), kernel_reg=l2(1e-4),activations=['leaky_relu', 'leaky_relu', 'leaky_relu']):
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

    model.add(Dense(1, activation='softplus'))  # exponential ensures no negative values

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
    'neurons': neuron_combinations
}

# Manual hyperparameter tuning
best_score = float('inf')
best_params = None

for activation_combo in activation_combinations:
    for batch_size in param_grid['batch_size']:
        for epochs in param_grid['epochs']:
            for dropout_rate in param_grid['dropout_rate']:
                for neurons_combo in param_grid['neurons']:
                    try:

                        model = create_model(optimizer='adam', dropout_rate=dropout_rate, neurons=neurons_combo, kernel_init=GlorotUniform(), kernel_reg=l2(1e-4),activations=activation_combo)
                        model.fit(regressor.X_train_scaled, regressor.y_train, batch_size=batch_size, epochs=epochs, verbose=1)
                        score = model.evaluate(regressor.X_test_scaled, regressor.y_test, verbose=0)
                        print(
                            f"Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Score: {score}")

                        with open('/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/grid_search.txt', 'a') as f:
                            f.write(f"Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Score: {score}")

                        if score < best_score:
                            best_score = score
                            best_params = {
                                'activation': activation_combo,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'dropout_rate': dropout_rate,
                                'neurons': neurons_combo
                            }
                    except:
                        print(f'failed with parameters: Activation: {activation_combo}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Neurons: {neurons_combo}, Score: {score}')

print(f"Best Score: {best_score} using {best_params}")