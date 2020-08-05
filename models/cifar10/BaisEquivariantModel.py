import tensorflow as tf

from models.layers.EquivariantPoolingLayer import EquivariantPoolingLayer
from models.layers.GroupConv import GroupConv


class BasicEquivariantModel(tf.keras.Model):
    def __init__(self):
        super(BasicEquivariantModel, self).__init__()
        self.conv1 = GroupConv(input_gruop='Z2', output_group='C4', input_channels=3, output_channels=16, ksize=3)
        self.conv2 = GroupConv(input_gruop='C4', output_group='C4', input_channels=16, output_channels=32, ksize=3)
        self.drop1 = tf.keras.layers.Dropout(rate=0.25)
        self.conv3 = GroupConv(input_gruop='C4', output_group='C4', input_channels=32, output_channels=64, ksize=3)
        self.conv4 = GroupConv(input_gruop='C4', output_group='C4', input_channels=64, output_channels=64, ksize=3)
        self.max_pooling = EquivariantPoolingLayer()
        self.drop2 = tf.keras.layers.Dropout(rate=0.25)
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1500, activation='relu')
        self.drop3 = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=[2, 2], strides=2, padding='SAME')
        x = self.drop1(x)
        x = self.conv3(x)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=[2, 2], strides=2, padding='SAME')
        x = self.conv4(x)
        x = tf.nn.relu(x)
        x = self.max_pooling(x, group='C4')
        x = self.drop2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.drop3(x)

        return self.dense2(x)
