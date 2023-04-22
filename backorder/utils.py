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


def exception_wrapper(func):
    """
    Wraps function in a try-except block and raises a custom exception if an exception is caught.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise CustomException(e, exc_info())
    return wrapper


def wrap_with_custom_exception(cls):
    """
    Wraps all methods of a class in a try-except block and raises a custom exception.
    """
    def wrapper(method):
        def wrapped(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                raise CustomException(e, exc_info())
        return wrapped

    for name, method in vars(cls).items():
        if callable(method):
            setattr(cls, name, wrapper(method))
    return cls


@exception_wrapper
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


@exception_wrapper
def to_yaml(fp: Path, data: dict):
    """ Function for Data Validation process. """
    fp.parent.mkdir(exist_ok=True)

    with open(fp, 'w') as f:
        yaml.dump(data, f)


@exception_wrapper
def dump_object(fp: Path, obj: object) -> None:
    """ Function for Data Transformation process. """
    try:
        logging.info('Enter in the save_object function of utils.')
        with open(fp, 'wb') as f:
            dill.dump(obj, f)
        logging.info('Exit from save_object function of utils.')
    except Exception as e:
        raise CustomException(e, exc_info()) from e


@exception_wrapper
def load_object(fp: Path):
    """ Function for Data Transformation process. """
    try:
        if not fp.exists():
            raise FileNotFoundError(fp)
        with open(fp, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e, exc_info()) from e


@exception_wrapper
def dump_array(fp: Path, array):
    with open(fp, "wb") as f:
        np.save(f, array)


@exception_wrapper
def load_array(fp: Path):
    with open(fp, "rb") as f:
        return np.load(f)
