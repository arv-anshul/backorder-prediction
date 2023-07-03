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


def wrap_with_custom_exception(cls):
    """
    Wraps all methods of a class in a try-except block and raises a custom exception.
    """
    def wrapper(method):
        def wrapped(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                raise CustomException(e, exc_info()) from e
        return wrapped

    for name, method in vars(cls).items():
        if callable(method):
            setattr(cls, name, wrapper(method))
    return cls


def read_dataset(fp: Path) -> DataFrame:
    """ Mostly supports `csv` and `parquet`. """
    # Extract pandas attribute from file extension
    suffix = fp.suffix[1:]

    # Print and log the warning
    if suffix not in ['csv', 'parquet']:
        warn_msg = 'utils.read_dataset: Supports CSV and parquet files easily.'
        warn(warn_msg)
        logging.warn(warn_msg)

    pd_attr = 'read_' + suffix
    df: DataFrame = getattr(pd, pd_attr)(fp)
    return df


def to_yaml(fp: Path, data: dict):
    fp.parent.mkdir(exist_ok=True)
    with open(fp, 'w') as f:
        yaml.dump(data, f)


def dump_object(fp: Path, obj: object) -> None:
    logging.info('Dumping object at %s', fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, 'wb') as f:
        dill.dump(obj, f)


def load_object(fp: Path):
    logging.info('Loading object from %s', fp)
    if not fp.exists():
        raise FileNotFoundError(fp)
    with open(fp, 'rb') as f:
        return dill.load(f)


def dump_array(fp: Path, array):
    logging.info('Dumping array at %s', fp)
    with open(fp, "wb") as f:
        np.save(f, array)


def load_array(fp: Path):
    logging.info('Loading array from %s', fp)
    with open(fp, "rb") as f:
        return np.load(f)
