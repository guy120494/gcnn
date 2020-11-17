from copy import deepcopy
from enum import Enum
from typing import Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras import Model

from model_training import train_model, test_model
from models.P4Model import P4Model
from models.P4ModelInvariantMaxPooling import P4ModelInvariantMaxPooling
from models.Z2Model import Z2Model


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
    train_dataset, test_dataset = get_datasets(DataSetName.MNIST)
    test_accuracies = []
    results = {"model": [], "avg_accuracy": [], "rotated_train": [], "dataset": []}

    repeats = 5
    augmentations = [True, False]

    print("Z2")
    for aug in augmentations:
        for i in range(repeats):
            z2_model = Z2Model()
            train_model(z2_model, train_dataset, rotate_train=aug, epochs=100)
            test_accuracy, _ = test_model(z2_model, test_dataset, rotate_test=True)
            test_accuracies.append(test_accuracy)
        results["model"].append("z2")
        results["rotated_train"].append(aug)
        results["avg_accuracy"].append(np.mean(test_accuracies))
        results["dataset"].append("mnist")
        test_accuracies = []

    print("P4")
    for aug in augmentations:
        for i in range(repeats):
            p4_model = P4Model()
            train_model(p4_model, train_dataset, rotate_train=aug, epochs=100)
            test_accuracy, _ = test_model(p4_model, test_dataset, rotate_test=True)
            test_accuracies.append(test_accuracy)
        results["model"].append("p4")
        results["rotated_train"].append(aug)
        results["avg_accuracy"].append(np.mean(test_accuracies))
        results["dataset"].append("mnist")
        test_accuracies = []

    print("P4_Invariant")
    for aug in augmentations:
        for i in range(repeats):
            p4_model_invariant = P4ModelInvariantMaxPooling()
            train_model(p4_model_invariant, train_dataset, rotate_train=aug, epochs=100)
            test_accuracy, _ = test_model(p4_model_invariant, test_dataset, rotate_test=True)
            test_accuracies.append(test_accuracy)
        results["model"].append("p4_invariant")
        results["rotated_train"].append(aug)
        results["avg_accuracy"].append(np.mean(test_accuracies))
        results["dataset"].append("mnist")
        test_accuracies = []

    result = pd.DataFrame(data=results)
    print("DONE!")
    print(results)
    result.to_csv(path_or_buf='./mnist_test_accuracy_z2.csv')

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

    # print("\n----- P4 MODEL EQUIVARIANT POOLING CIFAR NOT ROTATED TRAIN-----\n")
    # p4_model_equivariant_max_pooling = BasicEquivariantModel()
    # eval_number_of_neurons_in_dense(p4_model_equivariant_max_pooling, train_dataset, test_dataset, rotate_train=True,
    #                                 rotate_test=True, neurons=[i for i in range(500, 1001, 50)])
    #
    # print("\n----- P4 MODEL INVARIANT POOLING CIFAR NOT ROTATED TRAIN-----\n")
    # p4_model_invariant_max_pooling = BasicInvariantModel()
    # eval_number_of_neurons_in_dense(p4_model_invariant_max_pooling, train_dataset, test_dataset, rotate_train=False,
    #                                 rotate_test=True, neurons=[i for i in range(500, 1001, 50)])
