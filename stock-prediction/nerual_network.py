# !/usr/bin/env python
from keras import Sequential
from keras.optimizers import RMSprop, Adam, SGD
from keras import layers
from sklearn.model_selection import GridSearchCV
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier


def neural_network_model(lr=0.001, momentum=0.9):
    model = Sequential()
    # Adds a densely-connected layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    # Add a softmax layer with 3 output units:
    model.add(layers.Dense(3, activation='softmax'))
    optimizer = SGD(lr=lr,
                    momentum=momentum)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def parameter_search(x_train, y_train, x_test, y_test):
    model = KerasClassifier(build_fn=neural_network_model,
                            batch_size=64,
                            epoch=50, verbose=0)
    learning_rate = np.arange(0.001, 0.01, 0.001)
    momentum = np.arange(0.4, 0.9, 0.05)
    param_test1 = dict(lr=learning_rate, momentum=momentum)
    grid_search1 = GridSearchCV(
        estimator=model,
        param_grid=param_test1, n_jobs=-1, cv=5)  # scoring='accuracy',
    grid_result1 = grid_search1.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result1.best_score_, grid_result1.best_params_))
    means = grid_result1.cv_results_['mean_test_score']
    stds = grid_result1.cv_results_['std_test_score']
    params1 = grid_result1.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params1):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # model = KerasClassifier(build_fn=neural_network_model,
    #                         batch_size=10,
    #                         epoch=120, verbose=0)
    batch_size = np.arange(10, 100, 10)
    epochs = np.arange(60, 130, 20)
    param_test2 = dict(batch_size=batch_size, epochs=epochs)
    grid_search2 = GridSearchCV(estimator=model, param_grid=param_test2, n_jobs=-1)
    grid_result2 = grid_search2.fit(x_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))

    nn_model = neural_network_model(lr=grid_result1.best_params_['lr'],
                                    momentum=grid_result1.best_params_['momentum'])
    nn_model.fit(x_train, y_train, batch_size=grid_result2.best_params_['batch_size'],
                 epochs=grid_result2.best_params_['epochs'],
                 validation_data=(x_test, y_test))
    train_result = model.evaluate(x_train, y_train, batch_size=10000)
    print(train_result)
    return nn_model
