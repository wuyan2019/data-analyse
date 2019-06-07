# !/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import layers


class NNModel:
    # init部分用来调整参数
    def __init__(self, learning_rate=0.001, momentum=0.9, batch_size=64, epoch=50):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.epoch = epoch

    def neural_network_model(self, x_train, y_train, x_test, y_test):
        model = tf.keras.Sequential()
        # Adds a densely-connected layer
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        # Add a softmax layer with 3 output units:
        model.add(layers.Dense(3, activation='softmax'))
        model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                           momentum=self.momentum),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epoch,
                  validation_data=(x_test, y_test))
        return model
