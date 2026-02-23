import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.gemstones.exception import CustomException
from src.gemstones.logger import logging
from src.gemstones.utils import save_obj, evaluate_model, print_result, model_metrics

import os
import sys
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_training(self, train, test):
        try:
            logging.info('Splliting Data')
            X_train, y_train, X_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'RandomForestRegressor': RandomForestRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose = False),
                'XGBRegressor': XGBRegressor()
            }

            model_report:dict = evaluate_model(models, X_train, X_test, y_train, y_test)
            print('Model Report')
            print('\n==========================================================================\n')
            logging.info(f'Model Report: {model_report}')
            best_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]

            if best_score < 0.6:
                logging.info('Model score is < 60%')
                raise CustomException('No best model found')
            
            print(f'Best model: {best_model_name} with score of {best_score}')
            logging.info(f'Best model: {best_model_name} with score of {best_score}')

            save_obj(
                path = self.model_trainer_config.model_trainer_path,
                obj = models[best_model_name]
            )
            logging.info('Model Saved')
        except Exception as e:
            logging.exception('Exception at initiate training')
            raise CustomException(e, sys)