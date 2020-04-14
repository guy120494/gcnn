import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util

from models.ConvBatchLayer import ConvBatchLayer


class P4Model(tf.keras.Model):

    def __init__(self):
        super(P4Model, self).__init__()
        # self.gcnn1 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn2 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn3 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn4 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn5 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn6 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn7 = tf.keras.layers.Conv2D(filters=20, kernel_size=(4, 4), activation='relu')

        x = tf.compat.v1.placeholder(tf.float32, None)

        gconv_indices, gconv_shape_info, w_shape = gconv2d_util(h_input='Z2', h_output='C4', in_channels=1,
                                                                out_channels=20, ksize=3)
        self.variable1 = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
        self.gcnn1 = ConvBatchLayer(
            conv=gconv2d(input=x, filter=self.variable1, strides=[1, 1, 1, 1], padding='SAME',
                         gconv_indices=gconv_indices,
                         gconv_shape_info=gconv_shape_info), activation='relu')

        gconv_indices, gconv_shape_info, w_shape = gconv2d_util(h_input='C4', h_output='C4', in_channels=20,
                                                                out_channels=20, ksize=3)
        self.variable2 = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
        self.gcnn2 = ConvBatchLayer(
            conv=gconv2d(input=x, filter=self.variable2, strides=[1, 1, 1, 1], padding='SAME',
                         gconv_indices=gconv_indices,
                         gconv_shape_info=gconv_shape_info), activation='relu')
        self.variable3 = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
        self.gcnn3 = ConvBatchLayer(
            conv=gconv2d(input=x, filter=self.variable2, strides=[1, 1, 1, 1], padding='SAME',
                         gconv_indices=gconv_indices,
                         gconv_shape_info=gconv_shape_info), activation='relu')
        self.variable4 = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
        self.gcnn4 = ConvBatchLayer(
            conv=gconv2d(input=x, filter=self.variable2, strides=[1, 1, 1, 1], padding='SAME',
                         gconv_indices=gconv_indices,
                         gconv_shape_info=gconv_shape_info), activation='relu')
        self.variable5 = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
        self.gcnn5 = ConvBatchLayer(
            conv=gconv2d(input=x, filter=self.variable2, strides=[1, 1, 1, 1], padding='SAME',
                         gconv_indices=gconv_indices,
                         gconv_shape_info=gconv_shape_info), activation='relu')
        self.variable6 = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
        self.gcnn6 = ConvBatchLayer(
            conv=gconv2d(input=x, filter=self.variable2, strides=[1, 1, 1, 1], padding='SAME',
                         gconv_indices=gconv_indices,
                         gconv_shape_info=gconv_shape_info), activation='relu')

        gconv_indices, gconv_shape_info, w_shape = gconv2d_util(h_input='C4', h_output='C4', in_channels=20,
                                                                out_channels=10, ksize=4)
        self.variable7 = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
        self.gcnn7 = ConvBatchLayer(
            conv=gconv2d(input=x, filter=self.variable2, strides=[1, 1, 1, 1], padding='SAME',
                         gconv_indices=gconv_indices,
                         gconv_shape_info=gconv_shape_info), activation='relu')

    def call(self, inputs, training=None, mask=None):
        x = self.gcnn1(inputs)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn2(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="SAME")
        x = self.gcnn3(x)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn4(x)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn5(x)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn6(x)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn7(x)

        x = tf.nn.softmax(x)

        return tf.squeeze(x)
