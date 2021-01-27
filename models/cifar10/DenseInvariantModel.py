import tensorflow as tf

from models.layers.EquivariantDense import EquivariantDense, DenseMaxPooling
from models.layers.GroupConv import GroupConv


class DenseInvariantModel(tf.keras.Model):
    def __init__(self, number_of_labels=10):
        super(DenseInvariantModel, self).__init__()
        self.conv1 = GroupConv(input_gruop='Z2', output_group='C4', input_channels=3, output_channels=16, ksize=3)
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = GroupConv(input_gruop='C4', output_group='C4', input_channels=16, output_channels=32, ksize=3)
        self.relu2 = tf.keras.layers.ReLU()
        self.max1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='SAME')
        # self.drop1 = tf.keras.layers.Dropout(rate=0.25)
        self.conv3 = GroupConv(input_gruop='C4', output_group='C4', input_channels=32, output_channels=64, ksize=3)
        self.relu3 = tf.keras.layers.ReLU()
        self.max2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='SAME')
        self.conv4 = GroupConv(input_gruop='C4', output_group='C4', input_channels=64, output_channels=64, ksize=3)
        self.relu4 = tf.keras.layers.ReLU()
        # self.drop2 = tf.keras.layers.Dropout(rate=0.25)
        self.dense1 = EquivariantDense(1500)
        self.max_pooling = DenseMaxPooling()
        self.flat = tf.keras.layers.Flatten()
        # self.drop3 = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(units=number_of_labels, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max1(x)
        # x = self.drop1(x, training=training)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.max2(x)
        x = self.conv4(x)
        x = self.relu4(x)
        # x = self.drop2(x, training=training)
        # x = self.flat(x)
        x = self.dense1(x)
        x = self.relu4(x)
        x = self.flat(x)
        x = self.max_pooling(x, group='C4')
        # x = self.drop3(x, training=training)

        return self.dense2(x)

    def get_config(self):
        pass
