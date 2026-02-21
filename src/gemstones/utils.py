import os
import sys

import numpy as np
import pandas as pd
import dill

from src.gemstones.exception import CustomException

def save_obj(path, obj):
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok = True)

        with open(path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)