""" Extra functions for the project. """

from pathlib import Path
from sys import exc_info
from warnings import warn

import dill
import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame

from backorder.exception import CustomException
from backorder.logger import logging


def read_dataset(fp: Path) -> DataFrame:
    """ Mostly supports `csv` and `parquet`. """
    # Extract pandas attribute from file extension
    suffix = fp.suffix[1:]

    # Display and log the warning
    if suffix not in ['csv', 'parquet']:
        warn_msg = 'utils.read_dataset: Supports CSV and parquet files easily.'
        warn(warn_msg)
        logging.warn(warn_msg)

    pd_attr = 'read_' + suffix
    df: DataFrame = getattr(pd, pd_attr)(fp)
    return df


def to_yaml(fp: Path, data: dict):
    """ Function for Data Validation process. """
    fp.parent.mkdir(exist_ok=True)

    with open(fp, 'w') as f:
        yaml.dump(data, f)


def dump_object(fp: Path, obj: object) -> None:
    """ Function for Data Transformation process. """
    try:
        logging.info('Enter in the save_object function of utils.')
        with open(fp, 'wb') as f:
            dill.dump(obj, f)
        logging.info('Exit from save_object function of utils.')
    except Exception as e:
        raise CustomException(e, exc_info()) from e


def load_object(fp: Path) -> object:
    """ Function for Data Transformation process. """
    try:
        if not fp.exists():
            raise FileNotFoundError(fp)
        with open(fp, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e, exc_info()) from e


def dump_array(fp: Path, array):
    with open(fp, "wb") as f:
        np.save(f, array)


def load_array(fp: Path):
    with open(fp, "rb") as f:
        return np.load(f)
