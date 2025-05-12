import numpy
import time

import pandas as pd
import numpy as np
import random as python_random
import tensorflow
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow
import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from scikeras.wrappers import KerasClassifier

LABELING = INPUT_FORMAT = 'ovs'


def build_model(filters_1=8, filters_2=32, filters_3=64, filters_4=128, filters_5=128, kernel_size=3, pool_size=1,
                units=64, rate=0.25):
    model = Sequential()

    model.add(Conv1D(filters=filters_1, kernel_size=kernel_size, activation='relu', input_shape=(14, 1)))
    model.add(Conv1D(filters=filters_2, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size))
    model.add(Conv1D(filters=filters_3, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size))
    model.add(Conv1D(filters=filters_4, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size))
    model.add(Conv1D(filters=filters_5, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size))
    model.add(Flatten())
    model.add(Dense(units=units, activation='relu'))
    model.add(Dropout(rate=rate))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(data_path):
    np.random.seed(1234)
    tensorflow.random.set_seed(1234)
    python_random.seed(1234)

    path = '{0}_{1}.csv'.format(data_path, INPUT_FORMAT)
    data = pd.read_csv(path, index_col=0)

    x = data[['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position', 'engine_temperature',
              'engine_load_value', 'heart_rate', 'current_weather',	'visibility', 'precipitation',
              'accidents_onsite', 'design_speed', 'accidents_time']]
    y = data['risk_level']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # cnn = KerasClassifier(build_fn=build_model, epochs=5, batch_size=32, verbose=0)
    # param_grid = {'model__filters_1': [8, 16, 32], 'model__filters_2': [32, 64, 128],
    #              'model__filters_3': [64, 128, 256], 'model__filters_4': [128, 256, 512],
    #              'model__filters_5': [128, 256, 512], 'model__units': [64, 128, 256, 512], 'model__rate': [0.25, 0.5]}

    # param_grid = {'model__filters_1': [16, 32], 'model__filters_2': [64, 128],
    #               'model__filters_3': [64, 128, 256], 'model__filters_4': [128, 256],
    #               'model__filters_5': [256, 512], 'model__units': [256, 512], 'model__rate': [0.5]}
    #
    # grid = GridSearchCV(estimator=cnn, param_grid=param_grid, cv=3, refit=True, verbose=10, n_jobs=-1)
    # grid.fit(x_train, y_train)
    # print(grid.best_params_)
    # print(grid.best_score_)
    # y_pred = grid.predict(x_test)
    # print(classification_report(y_test, y_pred))

    cnn = Sequential()
    cnn.add(Conv1D(16, kernel_size=3, activation='relu', input_shape=(x.shape[1], 1)))
    cnn.add(Conv1D(128, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling1D(1))
    cnn.add(Conv1D(128, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling1D(1))
    cnn.add(Conv1D(256, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling1D(1))
    cnn.add(Conv1D(512, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling1D(1))
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(4, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False,
                                ema_momentum=0.99, ema_overwrite_frequency=None, name="adam")
    cnn.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    keras_cnn = KerasClassifier(model=cnn, epochs=10, batch_size=32, random_state=42)

    scores = cross_val_score(keras_cnn, x, y, scoring='accuracy', cv=cv, verbose=10)
    print(scores)
    print('cnn:{}'.format(np.average(scores)))
    y_pred = cross_val_predict(keras_cnn, x, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)

    # Calculating training and inferring times
    model = cnn
    print("Training model...")
    start_time = time.time()
    #model.fit(x_train, y_train)
    model.fit(x_train, y_train-1, epochs=5)
    end_time = time.time()
    training_time = end_time - start_time
    print("Training time: {:.3f} s".format(training_time))

    print("Inferring...")
    start_time = time.time()
    y_pred = model.predict(x_test)
    end_time = time.time()
    inference_time = end_time - start_time
    print("Inferring time: {:.3f} s".format(inference_time))
    print("Inferring time / sample: {:.6f} s".format(inference_time / len(y_pred)))

    # Calculating accuracy and confusion matrix
    y_pred = pd.DataFrame(y_pred, columns=['1', '2', '3', '4'])
    y_pred = y_pred.idxmax(axis='columns')
    y_pred = y_pred.astype(float)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.3f}".format(accuracy))

    print(classification_report(y_test, y_pred, digits=3, target_names=['low', 'medium', 'high', 'very high']))
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)

    matplotlib.rcParams['font.size'] = 18
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                                   display_labels=['low', 'medium', 'high', 'very high'],
                                                   colorbar=False, cmap='Blues')
    plt.show()
