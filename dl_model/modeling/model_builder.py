from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, PReLU
from tensorflow.keras import regularizers
from dl_model.config.config import TrainConfig


def build_model(input_dim: int, configuration: TrainConfig) -> tf.keras.Model:
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    if configuration.regularizer_name == "l1":
        ker_reg = regularizers.l1(configuration.l1)
    elif configuration.regularizer_name == "l2":
        ker_reg = regularizers.l2(configuration.l2)
    elif configuration.regularizer_name == "l1_l2":
        ker_reg = regularizers.l1_l2(l1=configuration.l1, l2=configuration.l2)
    else:
        raise ValueError(f"Invalid regularizer_name: {configuration.regularizer_name}")

    for units in configuration.neurons:
        if units and units > 0:
            model.add(Dense(units, kernel_initializer="glorot_uniform", kernel_regularizer=ker_reg))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Dropout(configuration.dropout_rate))

    model.add(Dense(1, activation="linear"))
    return model
