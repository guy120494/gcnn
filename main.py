from copy import deepcopy
from enum import Enum
from typing import Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras import Model

from model_training import train_model, test_model


class DataSetName(Enum):
    MNIST = 1
    CIFAR = 2


def get_mnist_data() -> Tuple[Any, Any, Any, Any]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train[y_train != 9]
    y_train = y_train[y_train != 9]

    x_test = x_test[y_test != 9]
    y_test = y_test[y_test != 9]

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # x_train = x_train[:640]
    # y_train = y_train[:640]

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    return x_train / 255.0, y_train, x_test / 255.0, y_test


def get_cifar_data() -> Tuple[Any, Any, Any, Any]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # x_train = x_train[:640]
    # y_train = y_train[:640]

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # x_test = x_test[:640]
    # y_test = y_test[:640]

    return x_train / 255.0, y_train, x_test / 255.0, y_test


def get_datasets(data_set_name: DataSetName):
    if data_set_name == DataSetName.CIFAR:
        x_train, y_train, x_test, y_test = get_cifar_data()
    else:
        x_train, y_train, x_test, y_test = get_mnist_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
    return train_dataset, test_dataset


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
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    data = pd.DataFrame(data)
    data.to_csv("./test.csv")
