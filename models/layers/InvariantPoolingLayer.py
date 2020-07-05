import tensorflow as tf
from tensorflow.keras import layers


class InvariantPoolingLayer(layers.Layer):

    def __init__(self):
        super(InvariantPoolingLayer, self).__init__()

    def call(self, inputs, **kwargs):
        group = kwargs.get("group")
        # coset max-pool
        x_shape = tf.shape(inputs).numpy()

        if group == 'Z2':
            x = tf.reshape(inputs, [x_shape[0], x_shape[1], x_shape[2], -1, 1])
        elif group == 'C4':
            x = tf.reshape(inputs, [x_shape[0], x_shape[1], x_shape[2], -1, 4])
            x = tf.unstack(x, axis=-1)
            x[1] = tf.image.rot90(x[0], 3)
            x[2] = tf.image.rot90(x[1], 2)
            x[3] = tf.image.rot90(x[2], 1)
            x = tf.stack(x, axis=-1)
        else:  # The group is D4
            x = tf.reshape(inputs, [x_shape[0], x_shape[1], x_shape[2], -1, 8])
        return tf.reduce_max(x, axis=[4])
