import tensorflow as tf

from models.layers.ConvBatchLayer import ConvBatchLayer
from models.layers.GroupConv import GroupConv
from models.layers.InvariantPoolingLayer import InvariantPoolingLayer


class P4AllCNNC(tf.keras.Model):
    def __init__(self):
        ksize = 3

        super(P4AllCNNC, self).__init__()

        self.l1 = ConvBatchLayer(
            conv=GroupConv(input_gruop='Z2', output_group='C4', input_channels=3, output_channels=48, ksize=ksize)
        )

        self.l2 = ConvBatchLayer(
            conv=GroupConv(input_gruop='C4', output_group='C4', input_channels=48, output_channels=48, ksize=ksize)
        )

        self.l3 = ConvBatchLayer(
            conv=GroupConv(input_gruop='C4', output_group='C4', input_channels=48, output_channels=48, ksize=ksize,
                           strides=2)
        )

        self.l4 = ConvBatchLayer(
            conv=GroupConv(input_gruop='C4', output_group='C4', input_channels=48, output_channels=96, ksize=ksize)
        )

        self.l5 = ConvBatchLayer(
            conv=GroupConv(input_gruop='C4', output_group='C4', input_channels=96, output_channels=96, ksize=ksize)
        )

        self.l6 = ConvBatchLayer(
            conv=GroupConv(input_gruop='C4', output_group='C4', input_channels=96, output_channels=96, ksize=ksize,
                           strides=2)
        )

        self.l7 = ConvBatchLayer(
            conv=GroupConv(input_gruop='C4', output_group='C4', input_channels=96, output_channels=96, ksize=ksize)
        )

        self.l8 = ConvBatchLayer(
            conv=GroupConv(input_gruop='C4', output_group='C4', input_channels=96, output_channels=96, ksize=1)
        )

        # Note: it's unusual to have a bn + relu before softmax, but this is what's described by springenberg et al.
        self.l9 = ConvBatchLayer(
            conv=GroupConv(input_gruop='C4', output_group='C4', input_channels=96, output_channels=10, ksize=1)
        )

        self.pooling = InvariantPoolingLayer()

    def __call__(self, inputs, training=None, mask=None):
        h = inputs
        h = tf.nn.dropout(h, rate=0.2)
        h = self.l1(h)
        h = self.l2(h)
        h = self.l3(h)
        h = tf.nn.dropout(h, rate=0.5)
        h = self.l4(h)
        h = self.l5(h)
        h = self.l6(h)
        h = tf.nn.dropout(h, rate=0.5)
        h = self.l7(h)
        h = self.l8(h)
        h = self.l9(h)

        h = tf.math.reduce_sum(h, axis=-1)
        h = tf.math.reduce_sum(h, axis=-1)
        h = tf.math.reduce_sum(h, axis=-1)
        h /= 8 * 8 * 4

        return tf.nn.softmax(h)
