import os
import sys

import pandas as pd

from src.gemstones.exception import CustomException
from src.gemstones.logger import logging
from src.gemstones.utils import load_objects

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_objects(preprocessor_path)
            model = load_objects(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        except Exception as e:
            logging.exception('Error occurred at predict')
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, carat: float, cut: str, color: str, clarity: str, depth: float, table: float, x: float, y: float, z: float):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

    def get_dataframe(self):
        try:
            custom_data = {
                'carat': [self.carat],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z]
            }

            df = pd.DataFrame(data = custom_data)
            logging.info('DataFrame added')
            return df
        except Exception as e:
            logging.exception('Exception occurred at custom data')
