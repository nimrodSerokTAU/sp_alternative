import keras_tuner as kt
from keras_tuner import HyperModel, RandomSearch
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, ELU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from regressor import Regressor

class MyHyperModel(HyperModel):
    def __init__(self, regressor):
        self.regressor = regressor

    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(self.regressor.X_train_scaled.shape[1],)))

        # First hidden layer
        model.add(Dense(hp.Choice('units_1', values=[16, 32, 64, 128]),
                        kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)))
        activation_1 = hp.Choice('activation_1', values=['relu', 'elu', 'leaky_relu'])
        if activation_1 == 'leaky_relu':
            model.add(LeakyReLU(negative_slope=0.01))
        elif activation_1 == 'elu':
            model.add(ELU())
        else:
            model.add(layers.Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        # Second hidden layer
        model.add(Dense(hp.Choice('units_2', values=[16, 32, 64, 128]),
                        kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)))
        activation_2 = hp.Choice('activation_2', values=['relu', 'elu', 'leaky_relu'])
        if activation_2 == 'leaky_relu':
            model.add(LeakyReLU(negative_slope=0.01))
        elif activation_2 == 'elu':
            model.add(ELU())
        else:
            model.add(layers.Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        # Third hidden layer
        model.add(Dense(hp.Choice('units_3', values=[16, 32, 64, 128]),
                        kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)))
        activation_3 = hp.Choice('activation_3', values=['relu', 'elu', 'leaky_relu'])
        if activation_3 == 'leaky_relu':
            model.add(LeakyReLU(negative_slope=0.01))
        elif activation_3 == 'elu':
            model.add(ELU())
        else:
            model.add(layers.Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='softplus'))

        learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model


# Initialize the regressor
regressor = Regressor("/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/500K_features.csv", 0.2, mode=2, predicted_measure='tree_distance')

# Define epochs and batch sizes
epochs = [10, 30, 50]
batch_sizes = [8, 16, 32]

results = []
# Search for the best hyperparameters
for batch_size in batch_sizes:
    for epoch in epochs:
        # Create a new instance of the tuner each time
        tuner = RandomSearch(
            MyHyperModel(regressor),
            objective='val_mean_absolute_error',
            max_trials=10,
            executions_per_trial=1,
            directory='/Users/kpolonsky/Documents/sp_alternative/feature_extraction/out/',
            project_name=f'Tree_dist_mode2_bs.{batch_size}_epochs.{epoch}'
        )

        tuner.search(regressor.X_train_scaled, regressor.y_train, epochs=epoch, batch_size=batch_size,
                     validation_split=0.2)

        best_model = tuner.get_best_models(num_models=1)[0]

        # Evaluate the model on the validation set
        val_loss, val_mae = best_model.evaluate(regressor.X_train_scaled, regressor.y_train, verbose=0)

        # Store the results with epochs and batch size
        results.append((epoch, batch_size, val_loss, val_mae))

# Retrieve the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(best_hyperparameters.values)

# Find the best result from stored results
best_result = min(results, key=lambda x: x[2])  # Minimize validation loss
best_epochs, best_batch_size, best_val_loss, best_val_mae = best_result

print(f"Best Batch Size: {best_batch_size}, Best Epochs: {best_epochs}, Validation Loss: {best_val_loss}, Validation MAE: {best_val_mae}")


# Train the best model with the optimal epochs and batch size
final_best_model = tuner.get_best_models(num_models=1)[0]
history = final_best_model.fit(regressor.X_train_scaled, regressor.y_train,
                               epochs=best_epochs,  # default to 30 if not found
                               batch_size=best_batch_size,  # default to 16 if not found
                               validation_split=0.2,
                               verbose=1)


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot training & validation MAE values
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()