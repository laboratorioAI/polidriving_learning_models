import keras.src.activations
import tensorflow
import numpy as np
import random as pr
from keras.api.models import Sequential
from keras.api.layers import Dense, Activation
from keras.api.optimizers import RMSprop
import numpy as np
import pandas as pd
import time
import matplotlib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from rbflayer import RBFLayer, InitCentersRandom, InitCentersKMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import OneHotEncoder


LABELING = INPUT_FORMAT = 'ovs'

np.random.seed(1234)
tensorflow.random.set_seed(1234)
pr.seed(1234)

path = '{0}_{1}.csv'.format('20240208_120000', INPUT_FORMAT)
data = pd.read_csv(path, index_col=0)

x = data[['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position', 'engine_temperature',
          'engine_load_value', 'heart_rate', 'current_weather', 'visibility', 'precipitation',
          'accidents_onsite', 'design_speed', 'accidents_time']]
y = data['risk_level']

y_ = data['risk_level'].values
y_ = y_.reshape(len(y_), 1)
ohe = OneHotEncoder()
Y = ohe.fit_transform(y_).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.30, random_state=42)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Radial basis function neural network
model = Sequential()
rbflayer = RBFLayer(100, initializer=InitCentersKMeans(x_train), betas=3.0, input_shape=(14, 1))
model.add(rbflayer)
model.add(Dense(4))
model.add(Activation('linear'))
print(model.summary())
model.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['accuracy'])

keras_rbfnn = KerasClassifier(model=model, epochs=1, batch_size=64, random_state=42)
scores = cross_val_score(keras_rbfnn, x, Y, scoring='accuracy', cv=cv, verbose=10, n_jobs=-1)
print(scores)
print('rbfnn:{}'.format(np.average(scores)))

# Calculating training and inferring times
print("Training model...")
start_time = time.time()
# model.fit(x_train, y_train)
model.fit(x_train, y_train, epochs=300, batch_size=64)
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
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(accuracy))

print(classification_report(y_test, y_pred, digits=3, target_names=['low', 'medium', 'high', 'very high']))
conf_mat = confusion_matrix(y_test, y_pred)

# Visualizing confusion matrix
matplotlib.rcParams['font.size'] = 18
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                               display_labels=['low', 'medium', 'high', 'very high'],
                                               colorbar=False, cmap='Blues')
plt.show()
