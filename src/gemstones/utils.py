import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.gemstones.exception import CustomException
from src.gemstones.logger import logging

def save_obj(path, obj):
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok = True)

        with open(path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        logging.exception('Exception Occurred at Save Object')
        raise CustomException(e, sys)
    
def evaluate_model(models: dict, X_train, X_test, y_train, y_test):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_score
        
        return report
    except Exception as e:
        logging.exception('Exception Occurred at Model Training')
        raise CustomException(e, sys)


def model_metrics(true, pred):
    try:
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)

        return mae, rmse, r2
    except Exception as e:
        logging.exception('Excption Occurred at Model Metrics')
        raise CustomException(e, sys)
    
def print_result(model, X_train, X_test, y_train, y_test):
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mae_train, rmse_train, r2_train = model_metrics(y_train, y_train_pred)
        mae_test, rmse_test, r2_test = model_metrics(y_test, y_test_pred)

        print('Train Metrics: ')
        print(' - Root mean squared error: ', rmse_train)
        print(' - Mean absloute error: ', mae_train)
        print(' - r2 score: ', r2_train)
        print(f'\n {'-'*50} \n')
        print('Test Metrics: ')
        print(' - Root mean squared error: ', rmse_test)
        print(' - Mean absloute error: ', mae_test)
        print(' - r2 score: ', r2_test)
    except Exception as e:
        logging.exception('Exception Occurred at Print result')
        raise CustomException(e, sys)

def load_objects(file_path):
    with open(file_path, 'rb') as f:
        try:
            with open(file_path, 'rb') as f:
                return dill.load(f)
        except Exception as e:
            logging.exception('Exception occurred at load_object')
            raise CustomException(e, sys)
    