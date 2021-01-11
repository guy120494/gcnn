import tensorflow as tf
from tensorflow.keras import layers


class EquivariantDense(layers.Layer):

    def __init__(self, output_number):
        super(EquivariantDense, self).__init__()
        self.output_number = output_number
        self.w = None

    def build(self, input_shape):
        shape = (input_shape[0], 8, 8, 256 // 4, self.output_number)
        self.w = tf.Variable(tf.Variable(tf.initializers.GlorotUniform()(shape=shape)),
                             dtype="float32", trainable=True)

    def call(self, inputs, **kwargs):
        Ws = [self.w[:, :, :, :, i] for i in range(self.output_number)]
        result = []
        for W in Ws:
            W = [tf.image.rot90(tf.squeeze(W), i) for i in range(4)]
            W = tf.stack(W, axis=-1)
            result.append(W)

        result = tf.stack(result, axis=-1)
        result = tf.transpose(tf.reshape(tf.reshape(result, shape=[64, 8, 8, 64 * 4, 15]), shape=[64, 8 * 8 * 256, 15]),
                              perm=[0, 2, 1])

        inputs = tf.expand_dims(inputs, axis=-1)
        return tf.matmul(result, inputs)
