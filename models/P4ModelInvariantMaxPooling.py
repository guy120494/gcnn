import tensorflow as tf

from models.layers.GroupConv import GroupConv


def invariant_max_pooling(x, group):
    # coset max-pool
    x_shape = tf.shape(x).numpy()
    if group == 'Z2':
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], -1, 1])
    elif group == 'C4':
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], -1, 4])
        x = tf.unstack(x, axis=-1)
        x[0] = tf.image.rot90(x[0], 3)
        x[1] = tf.image.rot90(x[1], 2)
        x[2] = tf.image.rot90(x[2], 1)
        x = tf.stack(x, axis=-1)
    else:  # The group is D4
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], -1, 8])
    return tf.reduce_max(x, axis=[4])


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

        self.gcnn7 = GroupConv(input_gruop='C4', output_group='C4', input_channels=10, output_channels=10, ksize=4)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(units=10)

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
        x = invariant_max_pooling(x, 'C4')
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.nn.softmax(x)

        return tf.squeeze(x)