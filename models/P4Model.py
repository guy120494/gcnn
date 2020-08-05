import tensorflow as tf

from models.layers.EquivariantPoolingLayer import EquivariantPoolingLayer
from models.layers.GroupConv import GroupConv


class P4Model(tf.keras.Model):

    def __init__(self):
        super(P4Model, self).__init__()
        # self.gcnn1 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu')
        # self.gcnn2 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu')
        # self.gcnn3 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu')
        # self.gcnn4 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu')
        # self.gcnn5 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu')
        # self.gcnn6 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu')
        # self.gcnn7 = tf.keras.layers.Conv2D(filters=10, kernel_size=(4, 4), activation='relu')

        self.gcnn1 = GroupConv(input_gruop='Z2', output_group='C4', input_channels=1, output_channels=10, ksize=3)

        self.gcnn2 = GroupConv(input_gruop='C4', output_group='C4', input_channels=10, output_channels=10, ksize=3)
        self.gcnn3 = GroupConv(input_gruop='C4', output_group='C4', input_channels=10, output_channels=10, ksize=3)
        self.gcnn4 = GroupConv(input_gruop='C4', output_group='C4', input_channels=10, output_channels=10, ksize=3)
        self.gcnn5 = GroupConv(input_gruop='C4', output_group='C4', input_channels=10, output_channels=10, ksize=3)
        self.gcnn6 = GroupConv(input_gruop='C4', output_group='C4', input_channels=10, output_channels=10, ksize=3)

        self.gcnn7 = GroupConv(input_gruop='C4', output_group='C4', input_channels=10, output_channels=10, ksize=3)
        self.max_pooling = EquivariantPoolingLayer()
        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(units=9)

    def call(self, inputs, training=None, mask=None):
        x = tf.nn.relu(self.gcnn1(inputs))
        # x = tf.nn.dropout(x, rate=0.3)
        x = tf.nn.relu(self.gcnn2(x))
        # x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="SAME")
        x = tf.nn.relu(self.gcnn3(x))
        # x = tf.nn.dropout(x, rate=0.3)
        x = tf.nn.relu(self.gcnn4(x))
        # x = tf.nn.dropout(x, rate=0.3)
        x = tf.nn.relu(self.gcnn5(x))
        # x = tf.nn.dropout(x, rate=0.3)
        x = tf.nn.relu(self.gcnn6(x))
        # x = tf.nn.dropout(x, rate=0.3)
        x = tf.nn.relu(self.gcnn7(x))
        x = self.max_pooling(x, 'C4')
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.nn.softmax(x)

        return tf.squeeze(x)
