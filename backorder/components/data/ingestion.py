""" Divides data for pipeline. """

from pathlib import Path
from sys import exc_info
from typing import Literal

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from backorder import utils
from backorder.entity.artifact_entity import DataIngestionArtifact
from backorder.entity.config_entity import DataIngestionConfig
from backorder.exception import CustomException
from backorder.logger import logging


class DataIngestion(DataIngestionConfig):
    def __init__(self):
        """ Divides and store the main data into train, test and validation data. """
        super().__init__()
        logging.info(f"{'>>'*10} Data Ingestion {'<<'*10}")

    def _import_data(self, fp: Path | None = None) -> DataFrame:
        """ Import the data from the provided path. """
        import_path = self.base_data_fp if fp is None else fp

        log_msg = 'Importing main data from "%s"'
        logging.info(log_msg, import_path)

        df = utils.read_dataset(import_path)
        return df

    def _clean_df(self, df: DataFrame) -> DataFrame:
        """ Custom cleaning of the df if requires. """
        df = df[:-1]

        # Drop columns
        df = df.drop(columns=['sku'])

        return df

    def _split(self, df: DataFrame) -> list[DataFrame]:
        """ Split DataFrame using `train_test_split` and returns. """
        logging.info('Split DataFrame into train and test.')
        return train_test_split(df, test_size=self.test_size)

    def _df_to_parquet(self, df: DataFrame, fp: Path) -> None:
        """ Convert DataFrame to parquet format and store it. """
        # Change different file extension to parquet
        fp = fp.with_suffix('.parquet')

        logging.info('Saving DataFrame at "%s"', fp)
        df.to_parquet(fp, index=False)

    def initiate(self, main_data_fp: Path | None = None) -> DataIngestionArtifact:
        """ Initiate the Data Ingestion process. """
        try:
            df = self._import_data(main_data_fp)
            # df = self._clean_df(df)
            train_df, test_df = self._split(df)

            # Save train and test df
            self._df_to_parquet(train_df, self.train_path)
            self._df_to_parquet(test_df, self.test_path)

            # Prepare artifact
            artifact = DataIngestionArtifact(
                self.feature_store_fp, self.train_path, self.test_path
            )
            logging.info(f"Data ingestion artifact: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e, exc_info())
