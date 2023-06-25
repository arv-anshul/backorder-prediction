""" Default configuration for the project. """


from pathlib import Path
from typing import Literal

STORED_MODEL_PATH = Path('stored_models')
PREDICTION_TYPE: Literal['regression', 'classification'] = 'classification'
BASE_DATA_NAME = 'raw_data.csv'
TARGET_COLUMN = 'went_on_backorder'
