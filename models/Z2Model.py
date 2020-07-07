import tensorflow as tf
from tensorflow_core.python.keras.layers import Conv2D

from models.layers.ConvBatchLayer import ConvBatchLayer


class Z2Model(tf.keras.Model):

    def __init__(self):
        super(Z2Model, self).__init__()
        # self.gcnn1 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn2 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn3 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn4 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn5 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn6 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu')
        # self.gcnn7 = tf.keras.layers.Conv2D(filters=20, kernel_size=(4, 4), activation='relu')

        self.gcnn1 = ConvBatchLayer(conv=Conv2D(filters=20, kernel_size=(3, 3), activation='relu'))
        self.gcnn2 = ConvBatchLayer(conv=Conv2D(filters=20, kernel_size=(3, 3), activation='relu'))
        self.gcnn3 = ConvBatchLayer(conv=Conv2D(filters=20, kernel_size=(3, 3), activation='relu'))
        self.gcnn4 = ConvBatchLayer(conv=Conv2D(filters=20, kernel_size=(3, 3), activation='relu'))
        self.gcnn5 = ConvBatchLayer(conv=Conv2D(filters=20, kernel_size=(3, 3), activation='relu'))
        self.gcnn6 = ConvBatchLayer(conv=Conv2D(filters=20, kernel_size=(3, 3), activation='relu'))
        self.gcnn7 = ConvBatchLayer(conv=Conv2D(filters=10, kernel_size=(3, 3)))

    def call(self, inputs, training=None, mask=None):
        x = self.gcnn1(inputs, training=training)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn2(x, training=training)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="SAME")
        x = self.gcnn3(x, training=training)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn4(x, training=training)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn5(x, training=training)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn6(x, training=training)
        x = tf.nn.dropout(x, rate=0.3)
        x = self.gcnn7(x, training=training)

        x = tf.nn.softmax(x)

        return tf.squeeze(x)
