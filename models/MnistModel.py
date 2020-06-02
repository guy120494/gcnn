import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D


class MnistModel(tf.keras.Model):

    def __init__(self):
        super(MnistModel, self).__init__()

        self.l1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))
        self.l2 = MaxPooling2D((2, 2))
        self.l3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')
        self.l4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')
        self.l5 = MaxPooling2D((2, 2))
        self.l6 = Flatten()
        self.l7 = Dense(100, activation='relu', kernel_initializer='he_uniform')
        self.l8 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        return x

# def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(10, activation='softmax'))
#     # compile model
#     opt = SGD(lr=0.01, momentum=0.9)
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
