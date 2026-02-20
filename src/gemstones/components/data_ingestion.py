import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from gemstones.exception import CustomException
from gemstones.logger import logging
from dataclasses import dataclass

#initialize data ingestion config
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

#data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion started')
        try:
            data_path = os.path.join('data', 'train.csv')
            data = pd.read_csv(data_path)
            logging.info('Data read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)
            data.to_csv(self.ingestion_config.raw_data_path, index = False)

            train_set, test_set = train_test_split(data, random_state = 42, test_size = 0.2)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.exception('Exception occurred at data ingestion')
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    data_ingestion_obj = DataIngestion()
    train_data, test_data = data_ingestion_obj.initiate_data_ingestion()