from tensorflow.keras import layers


class ConvBatchLayer(layers.Layer):
    def __init__(self, conv, **kwargs):
        super(ConvBatchLayer, self).__init__(**kwargs)
        self.batch = layers.BatchNormalization()
        self.conv = conv

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        return self.batch(x, training)
