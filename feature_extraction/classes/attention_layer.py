import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.models import Model


class AttentionLayer(Layer):
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)
        self.attn_weights = self.add_weight(shape=(self.units, 1), initializer='random_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Apply attention mechanism (dot-product attention)
        attn_logits = tf.matmul(inputs, self.W) + self.b
        attn_logits = tf.nn.tanh(attn_logits)
        attn_weights = tf.nn.softmax(tf.matmul(attn_logits, self.attn_weights), axis=1)

        # Weight the input features by the attention weights
        output = tf.reduce_sum(inputs * attn_weights, axis=1)
        return output, attn_weights
