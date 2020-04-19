import numpy as np
import tensorflow as tf
from numpy import newaxis
from tensorflow import keras

from models.P4Model import P4Model
from models.Z2Model import Z2Model


def grad(model, loss_fn, inputs, targets, training=True):
    with tf.GradientTape() as tape:
        y_pred = model(inputs, training=training)
        loss_value = loss_fn(y_true=targets, y_pred=y_pred)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), y_pred


def train_model(model):
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train[..., newaxis]
    x_test = x_test[..., newaxis]

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    x_train = x_train[:640]
    y_train = y_train[:640]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    # Iterate over epochs.
    for epoch in range(10):
        print(f'Start of epoch {epoch + 1}')

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Iterate over the batches of the dataset.
        for x_batch_train, y_true in train_dataset:
            loss_value, grads, y_pred = grad(model, loss_fn, x_batch_train, y_true)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y_true, y_pred)

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch + 1,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

    test_accuracy = tf.keras.metrics.Accuracy()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)
    for (x, y) in test_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


if __name__ == '__main__':
    print("----- P4 MODEL -----")
    p4_model = P4Model()
    train_model(p4_model)
    print("----- Z2 MODEL -----")
    z2_model = Z2Model()
    train_model(z2_model)
