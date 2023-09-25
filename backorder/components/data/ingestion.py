""" Divides data for pipeline. """

from pathlib import Path
from typing import Tuple

import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from backorder import utils
from backorder.config import TARGET_COLUMN
from backorder.entity import DataIngestionArtifact, DataIngestionConfig
from backorder.logger import logging


@utils.wrap_with_custom_exception
class DataIngestion(DataIngestionConfig):
    def __init__(self):
        """Divides and store the main data into train, test and validation data."""
        super().__init__()
        logging.info(f"{'>>'*10} Data Ingestion {'<<'*10}")

    def _import_data(self, fp: Path | None = None) -> DataFrame:
        """Import the data from the provided path."""
        import_path = self.base_data_fp if fp is None else fp

        log_msg = 'Importing main data from "%s"'
        logging.info(log_msg, import_path)

        df = utils.read_dataset(import_path)
        return df

    def _clean_df(self, df: DataFrame) -> DataFrame:
        """Custom cleaning of the df if requires."""
        df = df[:-1]

        # Drop columns
        df = df.drop(columns=['sku'])

        return df

    def _df_to_parquet(self, df: DataFrame, fp: Path) -> None:
        """Convert DataFrame to parquet format and store it."""
        # Change different file extension to parquet
        fp = fp.with_suffix('.parquet')

        logging.info('Saving DataFrame at "%s"', fp)
        df.to_parquet(fp, index=False)

    def upsample_data(self, X: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        X_majority, X_minority = X[y == 'No'], X[y == 'Yes']
        y_majority, y_minority = y[y == 'No'], y[y == 'Yes']

        X_upsampled, y_upsampled = resample(
            X_minority, y_minority, replace=True, n_samples=len(X_majority)
        )

        X_upsampled = np.concatenate((X_majority, X_upsampled))
        y_upsampled = np.concatenate((y_majority, y_upsampled))

        return (DataFrame(X_upsampled, columns=X.columns), Series(y_upsampled, name=TARGET_COLUMN))

    def initiate(
        self,
        main_data_fp: Path | None = None,
        upsample: bool = True,
    ) -> DataIngestionArtifact:
        """Initiate the Data Ingestion process."""
        df = self._import_data(main_data_fp)
        logging.info('Shape of imported raw data %s', df.shape)
        df = self._clean_df(df)

        # Up-sample the data to maintain balance
        if upsample:
            X_train_df, y_train_df = self.upsample_data(
                df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
            )
            X_train_df[y_train_df.name] = y_train_df
            df = X_train_df
            logging.info('Shape of upsampled raw data %s', df.shape)

        logging.info('Split DataFrame into train and test.')
        train_df, test_df = train_test_split(df, test_size=self.test_size)
        logging.info('Train data shape: %s', train_df.shape)
        logging.info('Test data shape: %s', test_df.shape)

        # Save train and test df
        self._df_to_parquet(train_df, self.train_path)
        self._df_to_parquet(test_df, self.test_path)

        # Prepare artifact
        artifact = DataIngestionArtifact(self.feature_store_fp, self.train_path, self.test_path)
        logging.info(f"Data ingestion artifact: {artifact}")
        return artifact
