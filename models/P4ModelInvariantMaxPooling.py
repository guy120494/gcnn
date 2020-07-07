import tensorflow as tf
from tensorflow_core.python.keras import Sequential

from models.layers.GroupConv import GroupConv
from models.layers.InvariantPoolingLayer import InvariantPoolingLayer


class P4ModelInvariantMaxPooling(tf.keras.Model):

    def __init__(self):
        super(P4ModelInvariantMaxPooling, self).__init__()
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
        self.invariant_pooling = InvariantPoolingLayer('C4')
        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(units=10, activation="softmax")

        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        x = self.relu(self.gcnn1(inputs))
        # x = tf.nn.dropout(x, rate=0.3)
        x = self.relu(self.gcnn2(x))
        # x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="SAME")
        x = self.relu(self.gcnn3(x))
        # x = tf.nn.dropout(x, rate=0.3)
        x = self.relu(self.gcnn4(x))
        # x = tf.nn.dropout(x, rate=0.3)
        x = self.relu(self.gcnn5(x))
        # x = tf.nn.dropout(x, rate=0.3)
        x = self.relu(self.gcnn6(x))
        # x = tf.nn.dropout(x, rate=0.3)
        x = self.relu(self.gcnn7(x))
        x = self.invariant_pooling(x)
        x = self.flatten(x)
        x = self.dense(x)

        return tf.squeeze(x)


if __name__ == '__main__':
    for i in range(5, 51):
        p4_model_invariant_max_pooling = P4ModelInvariantMaxPooling()
        input_tensor = tf.random.uniform(shape=(1, i, i, 1))

        invariant_layers = p4_model_invariant_max_pooling.layers[:8]

        check_invariance_model = Sequential(invariant_layers)

        result = check_invariance_model(input_tensor, training=False)

        rotated_input = tf.image.rot90(input_tensor, 1)
        rotated_result = check_invariance_model(rotated_input, training=False)
        print(f'Image of size {i}: {tf.math.reduce_max(tf.abs(tf.subtract(result, rotated_result))) < 10 ** -6}')
