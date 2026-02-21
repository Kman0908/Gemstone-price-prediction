import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from src.gemstones.exception import CustomException
from src.gemstones.logger import logging

from src.gemstones.utils import save_obj

#initilaize data transformation config
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        # this fucntion is used for data transformation

        try:
            #splitting columns
            cat_cols = ['cut', 'color', 'clarity']
            num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            #ranking for gems form real world data
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            #numerical pipeline
            num_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'median')),
                ('scalar', StandardScaler())
            ])

            #categorical pipeline
            cat_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'most_frequent')),
                ('encoder', OrdinalEncoder(categories = [cut_categories, color_categories, clarity_categories])),
                ('scalar', StandardScaler())
            ])

            logging.info(f'Numerical columns: {num_cols}')
            logging.info(f'Categorical columns: {cat_cols}')

            preprocessor = ColumnTransformer(transformers = [
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline,cat_cols)
            ])

            return preprocessor
        except Exception as e:
            logging.exception('Exception occurred at data transformation')
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info('Data read as pandas dataframe')
            logging.info(f'Sample of train: {train_data.head()}')
            logging.info(f'Sample of test: {test_data.head()}')

            #preprocessor obj
            logging.info('Got preprocessor object')
            preprocessor_obj = self.get_data_transformation_object()

            target_col = 'price'
            drop_col = ['id', 'price']

            input_features_train = train_data.drop(columns = drop_col)
            target_features_train = train_data[target_col]

            input_features_test = test_data.drop(columns = drop_col)
            target_features_test = test_data[target_col]

            logging.info('Preprocessing started')
            input_features_train = preprocessor_obj.fit_transform(input_features_train)
            input_features_test = preprocessor_obj.transform(input_features_test)

            train_arr = np.c_[input_features_train, np.array(target_features_train)]
            test_arr = np.c_[input_features_test, np.array(target_features_test)]

            save_obj(
                path = self.data_transformation_config.preprocessor_obj_file,
                obj = preprocessor_obj
            )

            logging.info('Preprocessor file saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )

        except Exception as e:
            logging.exception('Exception occurred at initiate data dransformation')
            raise CustomException(e, sys)