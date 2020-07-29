import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_core.python.layers.base import Layer


class ConvBatchLayer(layers.Layer):
    def __init__(self, conv: Layer, **kwargs):
        super(ConvBatchLayer, self).__init__(**kwargs)
        self.batch = layers.BatchNormalization()
        self.conv = conv

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        if not hasattr(x, 'activation'):
            x = tf.nn.relu(x)
        return self.batch(x, training)
