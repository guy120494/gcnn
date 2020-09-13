import random
from copy import deepcopy
from typing import Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras import Model

from models.cifar10.BaisEquivariantModel import BasicEquivariantModel

EPOCHS = 60


def get_cifar_data() -> Tuple[Any, Any, Any, Any]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # x_train = x_train[:640]
    # y_train = y_train[:640]

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # x_test = x_test[:640]
    # y_test = y_test[:640]

    return x_train / 255.0, y_train, x_test / 255.0, y_test


def get_datasets():
    x_train, y_train, x_test, y_test = get_cifar_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
    return train_dataset, test_dataset


def grad(model, loss_fn, inputs, targets, training=True):
    with tf.GradientTape() as tape:
        y_pred = model(inputs, training=training)
        loss_value = loss_fn(y_true=targets, y_pred=y_pred)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), y_pred


def randomly_rotate(x):
    number_of_rotations = random.choice([0, 1, 2, 3])
    return tf.image.rot90(x, k=number_of_rotations)


def get_learning_rate(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate


def train_model(model, train_dataset, rotate_train=False, epochs=EPOCHS):
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    # Iterate over epochs.
    for epoch in range(epochs):
        lr = get_learning_rate(epoch)
        tf.keras.backend.set_value(optimizer.lr, lr)

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Iterate over the batches of the dataset.
        for x_batch_train, y_true in train_dataset:
            if rotate_train:
                x_batch_train = randomly_rotate(x_batch_train)
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
        # if (epoch + 1) % 10 == 0:
        #     print("Epoch {}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch + 1,
        #                                                             epoch_loss_avg.result(),
        #                                                             epoch_accuracy.result()))


def test_model(model, test_dataset, rotate_test=False):
    test_accuracy = tf.keras.metrics.Accuracy()
    for (x, y) in test_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        if rotate_test:
            x = randomly_rotate(x)
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    return test_accuracy.result().numpy()


def eval_number_of_neurons_in_dense(model: Model, train_set, test_set, rotate_train=False, rotate_test=False,
                                    neurons=None):
    result = {"model": [], "neurons_in_dense": [], "accuracy": []}
    if neurons is None:
        neurons = [i for i in range(1000, 1501, 50)]

    for i in neurons:
        layers = deepcopy(model.layers)
        layers.pop(-3)
        layers.insert(-2, tf.keras.layers.Dense(units=i, activation='relu'))
        copy_model = keras.Sequential(layers)
        train_model(copy_model, train_set, rotate_train, epochs=40)
        accuracy = test_model(copy_model, test_set, rotate_test)
        result["model"].append(model.name)
        result["neurons_in_dense"].append(i)
        result["accuracy"].append(accuracy)

    result = pd.DataFrame(data=result)
    result.to_csv(path_or_buf=f'./{model.name}.csv')


if __name__ == '__main__':
    train_dataset, test_dataset = get_datasets()

    # print("\n----- P4 MODEL INVARIANT POOLING CIFAR ROTATED TRAIN-----\n")
    # p4_model_invariant_max_pooling = BasicInvariantModel()
    # train_model(p4_model_invariant_max_pooling, train_dataset, rotate_train=True)
    # test_model(p4_model_invariant_max_pooling, rotate_test=True, test_dataset=test_dataset)
    #
    # print("\n----- P4 MODEL INVARIANT POOLING CIFAR NOT ROTATED TRAIN-----\n")
    # p4_model_invariant_max_pooling = BasicInvariantModel()
    # train_model(p4_model_invariant_max_pooling, train_dataset, rotate_train=False)
    # test_model(p4_model_invariant_max_pooling, rotate_test=True, test_dataset=test_dataset)

    # print("\n----- P4 MODEL EQUIVARIANT POOLING CIFAR ROTATED TRAIN-----\n")
    # p4_model_equivariant_max_pooling = BasicEquivariantModel()
    # train_model(p4_model_equivariant_max_pooling, train_dataset, rotate_train=True)
    # test_model(p4_model_equivariant_max_pooling, rotate_test=True, test_dataset=test_dataset)
    #
    # print("\n----- P4 MODEL EQUIVARIANT POOLING CIFAR NOT ROTATED TRAIN-----\n")
    # p4_model_equivariant_max_pooling = BasicEquivariantModel()
    # train_model(p4_model_equivariant_max_pooling, train_dataset, rotate_train=False)
    # test_model(p4_model_equivariant_max_pooling, rotate_test=True, test_dataset=test_dataset)

    p4_model_equivariant_max_pooling = BasicEquivariantModel()
    eval_number_of_neurons_in_dense(p4_model_equivariant_max_pooling, train_dataset, test_dataset, rotate_train=True,
                                    rotate_test=True, neurons=[i for i in range(1300, 1501, 100)])

    # p4_model_invariant_max_pooling = BasicInvariantModel()
    # eval_number_of_neurons_in_dense(p4_model_invariant_max_pooling, train_dataset, test_dataset, rotate_train=False,
    #                                 rotate_test=True)
