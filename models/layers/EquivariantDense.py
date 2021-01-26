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
        super(EquivariantDense, self).__init__()
        self.output_number = output_number // 4
        self.gconv_indices = None
        self.gconv_shape_info = None
        self.w_shape = None
        self.w = None

    def build(self, input_shape):
        # shape = (input_shape[0], self.output_number, input_shape[1] // 4)
        # self.w1 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
        #                       dtype="float32", trainable=True)
        # self.w2 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
        #                       dtype="float32", trainable=True)
        # self.w3 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
        #                       dtype="float32", trainable=True)
        # self.w4 = tf.Variable(tf.initializers.GlorotUniform()(shape=shape),
        #                       dtype="float32", trainable=True)

        self.gconv_indices, self.gconv_shape_info, self.w_shape = gconv2d_util(
            h_input='C4', h_output='C4', in_channels=input_shape[-1] // 4,
            out_channels=self.output_number, ksize=input_shape[1])

        self.w = self.add_weight(shape=self.w_shape, initializer='random_normal', trainable=True,
                                 name='Equivariant Dense')

    def call(self, inputs, **kwargs):
        w = transform_filter_2d_nhwc(w=self.w, flat_indices=self.gconv_indices, shape_info=self.gconv_shape_info)
        w = tf.reshape(w, [w.shape[0], -1, w.shape[-1]])
        w = tf.transpose(w, perm=[0, 2, 1])

        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        inputs = tf.reshape(inputs, (inputs.shape[0], -1, inputs.shape[-1]))
        result = tf.matmul(w, inputs)

        return tf.transpose(result)
        # first_row = tf.concat([self.w1, self.w2, self.w3, self.w4], -1)
        # W = [tf.roll(first_row, shift=i * inputs.shape[-1] // 4, axis=-1) for i in range(4)]
        # W = tf.concat(W, 1)
        # inputs = tf.expand_dims(inputs, axis=-1)
        #
        # return tf.matmul(W, inputs)
