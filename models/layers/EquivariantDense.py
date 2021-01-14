import tensorflow as tf
from tensorflow.keras import layers


class DenseMaxPooling(layers.Layer):

    def call(self, inputs, **kwargs):
        length_of_input = inputs.shape[1]
        shifts = [i for i in range(0, length_of_input, length_of_input // 4)]
        result = [tf.roll(inputs, shift, axis=-1) for shift in shifts]
        result = tf.stack(result, axis=-1)

        return tf.reduce_max(result, axis=-1)


class EquivariantDense(layers.Layer):

    def __init__(self, output_number):
        super(EquivariantDense, self).__init__()
        self.output_number = output_number // 4
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None

    def build(self, input_shape):
        shape = (input_shape[0], self.output_number, 8 * 8 * 256 // 4)
        self.w1 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
                              dtype="float32", trainable=True)
        self.w2 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
                              dtype="float32", trainable=True)
        self.w3 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
                              dtype="float32", trainable=True)
        self.w4 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
                              dtype="float32", trainable=True)

    def call(self, inputs, **kwargs):
        first_row = tf.concat([self.w1, self.w2, self.w3, self.w4], -1)
        W = [tf.roll(first_row, shift=i * inputs.shape[-1] // 4, axis=-1) for i in range(4)]
        W = tf.concat(W, 1)
        inputs = tf.expand_dims(inputs, axis=-1)

        return tf.squeeze(tf.matmul(W, inputs))
