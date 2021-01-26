import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util
from groupy.gconv.tensorflow_gconv.transform_filter import transform_filter_2d_nhwc
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
        self.p4func = None
        super(EquivariantDense, self).__init__()
        self.output_number = output_number // 4
        self.w = None
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None

    def build(self, input_shape):
        shape = (input_shape[0], self.output_number, input_shape[1] // 4)
        self.w1 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
                              dtype="float32", trainable=True)
        self.w2 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
                              dtype="float32", trainable=True)
        self.w3 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
                              dtype="float32", trainable=True)
        self.w4 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
                              dtype="float32", trainable=True)

        gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
            h_input='C4', h_output='C4', in_channels=input_shape[-1] // 4,
            out_channels=self.output_number, ksize=input_shape[1])

        self.w = self.add_weight(shape=w_shape, initializer='random_normal', trainable=True)

        self.w = transform_filter_2d_nhwc(w=self.w, flat_indices=gconv_indices, shape_info=gconv_shape_info)
        self.w = tf.reshape(self.w, [self.w.shape[0], -1, self.w.shape[-1]])
        self.w = tf.transpose(self.w, perm=[0, 2, 1])

    def call(self, inputs, **kwargs):
        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        inputs = tf.reshape(inputs, (inputs.shape[0], -1, inputs.shape[-1]))
        result = tf.matmul(self.w, inputs)

        return tf.transpose(result)
        # first_row = tf.concat([self.w1, self.w2, self.w3, self.w4], -1)
        # W = [tf.roll(first_row, shift=i * inputs.shape[-1] // 4, axis=-1) for i in range(4)]
        # W = tf.concat(W, 1)
        # inputs = tf.expand_dims(inputs, axis=-1)
        #
        # return tf.matmul(W, inputs)
