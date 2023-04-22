""" Default configuration for the project. """


from pathlib import Path
from typing import Literal

STORED_MODEL_PATH = Path('stored_models')
PREDICTION_TYPE: Literal['regression', 'classification'] = 'classification'
BASE_DATA_NAME = 'cleaned_back_order_data_5000.parquet'
TARGET_COLUMN = 'went_on_backorder'
