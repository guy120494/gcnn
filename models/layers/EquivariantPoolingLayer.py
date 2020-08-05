import tensorflow as tf
from tensorflow.keras import layers


class EquivariantPoolingLayer(layers.Layer):

    def __init__(self):
        super(EquivariantPoolingLayer, self).__init__()

    def call(self, inputs, **kwargs):
        group = kwargs.get("group")
        # coset max-pool
        x_shape = tf.shape(inputs).numpy()

        if group == 'Z2':
            inputs = tf.reshape(inputs, [x_shape[0], x_shape[1], x_shape[2], -1, 1])
        elif group == 'C4':
            inputs = tf.reshape(inputs, [x_shape[0], x_shape[1], x_shape[2], -1, 4])
        else:  # The group is D4
            inputs = tf.reshape(inputs, [x_shape[0], x_shape[1], x_shape[2], -1, 8])
        return tf.reduce_max(inputs, axis=[4])
