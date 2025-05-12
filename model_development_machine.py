import joblib
import matplotlib
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import subprocess


LABELING = INPUT_FORMAT = 'ovs'


def main(data_path):
    path = '{0}_{1}.csv'.format(data_path, INPUT_FORMAT)
    data = pd.read_csv(path, index_col=0)

    x = data[['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position', 'engine_temperature',
              'engine_load_value', 'heart_rate', 'current_weather',	'visibility', 'precipitation',
              'accidents_onsite', 'design_speed', 'accidents_time']]
    y = data['risk_level']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Hyperparameter tuning for Multilayer Perceptron using GridSearchCV
    # paramgrid = {'hidden_layer_sizes': [(10, 10, 10), (50, 50, 50), (100, 100, 100), (10, 50, 10), (50, 10, 50),
    #                                    (50, 100, 50), (10, 10), (50, 50), (100, 100), (10, ), (50, ), (100,)],
    #             'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
    #             'learning_rate': ['constant', 'invscaling', 'adaptive'], 'max_iter': [1000]}
    # grid = GridSearchCV(MLPClassifier(), param_grid=paramgrid, refit=True, verbose=3, n_jobs=-1)
    # grid.fit(x_train, y_train)
    # print(grid.best_params_)
    # print(grid.best_score_)
    # y_pred = grid.predict(x_test)
    # print(classification_report(y_test, y_pred))

    # Multilayer perceptron
    mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive',
                        max_iter=200, solver='adam', random_state=42, verbose=10)

    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('mlp:{}'.format(accuracy))
    print('MLP train:{}'.format(accuracy_score(y_train, mlp.predict(x_train))))
    print('MLP test:{}'.format(accuracy_score(y_test, mlp.predict(x_test))))

    # Hyperparameter tuning for Gradient Boosting Machine using GridSearchCV
    # paramgrid = {'n_estimators': [1, 2, 5, 8, 10, 30, 50, 80, 100],
    #             'loss': ['log_loss', 'exponential'], 'learning_rate': [0.1, 0.5, 0.8, 1.0],
    #             'max_depth': [1, 2, 5, 8, 10, 30], 'min_samples_split': [0.1, 0.3, 0.5, 0.8, 1.0],
    #             'max_features': ['sqrt', 'log2']}
    # grid = GridSearchCV(GradientBoostingClassifier(), param_grid=paramgrid, refit=True, verbose=10, n_jobs=-1)
    # grid.fit(x_train, y_train)
    # print(grid.best_params_)
    # print(grid.best_score_)
    # y_pred = grid.predict(x_test)
    # print(classification_report(y_test, y_pred))

    # Gradient boosting machine
    # gbc = GradientBoostingClassifier(learning_rate=0.8, loss='log_loss', max_depth=30, max_features='sqrt',
    #                                 min_samples_split=0.5, n_estimators=100, random_state=42)
    # print(gbc.get_params())
    # gbc.fit(x_train, y_train)
    # y_pred = gbc.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print('gbc.{}'.format(accuracy))
    # print('GBC train:{}'.format(accuracy_score(y_train, gbc.predict(x_train))))
    # print('GBC test:{}'.format(accuracy_score(y_test, gbc.predict(x_test))))

    # Hyperparameter tuning for Support Vector Machine using GridSearchCV
    # svm = SVC(kernel='rbf')
    # param_grid = {'C': [1000, 3000, 5000, 7000, 9000, 11000, 13000], 'gamma': [0.001, 0.09, 0.05, 0.03, 0.01, 0.1]}
    # grid = GridSearchCV(svm, param_grid=param_grid, refit=True, verbose=10, n_jobs=-1)
    # grid.fit(x_train, y_train)
    # print(grid.best_params_)
    # print(grid.best_score_)

    svm = SVC(kernel='rbf', C=13000, gamma=0.1, random_state=42)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('mlp:{}'.format(accuracy))
    print('MLP train:{}'.format(accuracy_score(y_train, svm.predict(x_train))))
    print('MLP test:{}'.format(accuracy_score(y_test, svm.predict(x_test))))

    # Calculating training and inferring times
    model = mlp
    # model = gbc
    print("Training model...")
    start_time = time.time()
    model.fit(x_train, y_train)
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

    # Saving the model
    with open('model.joblib', 'wb') as f:
        joblib.dump(mlp, f)

    # Opening the model
    predictor = np.nan
    with open('model.joblib', 'rb') as f:
        predictor = joblib.load(f)

    # Testing the model
    # Sample 1
    # sample = {'steering_angle_': -10.2, 'speed': 91, 'rpm_': 4500, 'acceleration': 1.785648,
    #               'throttle_position': 15.89768, 'engine_temperature': 94, 'system_voltage': 13.2, 'heart_rate': 60,
    #               'distance_travelled': 15.38988989, 'latitude': -0.324568329, 'longitude': -78.37786345,
    #               'current_weather': 2, 'accidents_onsite': 27}
    # sample = [[16, 99, 4500, 0.29, 65.1, 94, 86.7, 94, 18, 3.2, 24, 105, 90, 15]]

    # x = data[['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position', 'engine_temperature',
    #          'engine_load_value', 'heart_rate', 'current_weather',	'visibility', 'precipitation',
    #          'accidents_onsite', 'design_speed', 'accidents_time']]

    # observation = pd.DataFrame(sample, index=[0])
    # output = predictor.predict(sample)

    # Sample 2
    # sample = [[12, 30, 1500, 0.29, 65.1, 94, 86.7, 74, 2, 16.1, 2, 10, 90, 2]]

    # observation = pd.DataFrame(sample, index=[0])
    # output = predictor.predict(sample)

    # Sample 3
    # sample = [[0.833333333, 0.217054264, 0.489009744, 0.778287462, 0.874499332, 0.548387097, 0.928714859, 0.387096774,
    #           0.066666667,	0.248062016, 0,	0.413385827, 1,	0.176470588]]

    # observation = pd.DataFrame(sample, index=[0])
    # output = predictor.predict(sample)
    # print(output[0])

    # Building tar file with model data + inference code
    bashCommand = "tar -cvpzf model.tar.gz model.joblib inference.py"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()