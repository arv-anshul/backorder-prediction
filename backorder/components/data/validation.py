""" Data Validation """

from pandas import DataFrame
from scipy.stats import ks_2samp

from backorder import utils
from backorder.entity import DataValidationArtifact, DataValidationConfig
from backorder.logger import logging


@utils.wrap_with_custom_exception
class DataValidation(DataValidationConfig):
    def __init__(self):
        """
        To initiate validation process between base, train and test dataset.
        """
        super().__init__()
        logging.info(f"{'>>'*10} Data Validation {'<<'*10}")
        self.validation_report = {}

    def _drop_missing_values_cols(
        self, df: DataFrame, report_name: str,
    ) -> DataFrame | None:
        """
        This function will drop column which contains missing value more than specified threshold.

        df: Accepts a pandas DataFrame
        threshold: Percentage criteria to drop a column

        Returns
        ------
        DataFrame if at least a single column is available after dropping missing columns else None.
        """

        threshold = self.missing_threshold
        null_report = df.isna().sum().div(df.shape[0])

        logging.info('Select columns which has null values more than %s',
                     threshold)
        drop_col = list(null_report[null_report > threshold].index)

        logging.info(f'Columns to drop: %s', drop_col)
        self.validation_report[report_name] = drop_col
        df.drop(drop_col, axis=1, inplace=True)

        if len(df.columns) == 0:
            return None
        return df

    def _is_required_cols_exists(
        self, base_df: DataFrame, curr_df: DataFrame, report_name: str,
    ) -> bool:
        base_cols = base_df.columns
        curr_cols = curr_df.columns

        missing_cols = []
        for base_col in base_cols:
            if base_col not in curr_cols:
                logging.info('Column: %s is not available.', base_col)
                missing_cols.append(base_col)

        if len(missing_cols) > 0:
            self.validation_report[report_name] = missing_cols
            return False
        return True

    def _data_drift(
        self, base_df: DataFrame, curr_df: DataFrame, report_name: str,
    ) -> None:
        drift_report = dict()
        for base_col in base_df.columns:
            base_data, curr_data = base_df[base_col], curr_df[base_col]

            logging.info('Hypothesis %s: %s, %s',
                         base_col, base_data.dtype, curr_data.dtype)
            distribution = ks_2samp(base_data, curr_data)
            pvalue = float(distribution.pvalue)    # type: ignore

            if pvalue > 0.05:
                drift_report[base_col] = {
                    'pvalues': pvalue, 'same_distribution': True
                }
            else:
                drift_report[base_col] = {
                    'pvalues': pvalue, 'same_distribution': False
                }
        self.validation_report[report_name] = drift_report

    def initiate(self) -> DataValidationArtifact:
        # --- --- Base Dataset --- --- #
        logging.info('Reading base DataFrame')
        base_df = utils.read_dataset(self.base_data_fp)

        logging.info('Drop null values columns from base df')
        base_df = self._drop_missing_values_cols(
            base_df, 'missing_values_within_base_dataset')

        # --- --- Train dataset --- --- #
        logging.info('Reading train DataFrame')
        train_df = utils.read_dataset(self.train_path)

        logging.info('Drop null values columns from train df')
        train_df = self._drop_missing_values_cols(
            train_df, 'missing_values_within_train_dataset')

        # --- --- Test Dataset --- --- #
        logging.info('Reading test DataFrame')
        test_df = utils.read_dataset(self.test_path)

        logging.info('Drop null values columns from test df')
        test_df = self._drop_missing_values_cols(
            test_df, 'missing_values_within_test_dataset')

        # --- --- Check datasets exists --- --- #
        if base_df is None:
            raise ValueError('Base Dataset cannot be None.')
        if train_df is None:
            raise ValueError('Train Dataset cannot be None.')
        if test_df is None:
            raise ValueError('Test Dataset cannot be None.')

        # --- --- Checking Data Drift --- --- #
        drift_log_msg = 'All columns are available in %s df hence detecting data drift'
        # --- --- Train Dataset --- --- #
        logging.info('Is all required columns present in train df')
        train_df_cols_status = self._is_required_cols_exists(
            base_df, train_df, 'missing_cols_within_train_dataset')
        if train_df_cols_status:
            logging.info(drift_log_msg % 'train')
            self._data_drift(base_df, train_df,
                             'data_drift_within_train_dataset')

        # --- --- Test Dataset --- --- #
        logging.info('Is all required columns present in test df')
        test_df_cols_status = self._is_required_cols_exists(
            base_df, test_df, 'missing_cols_within_test_dataset')
        if test_df_cols_status:
            logging.info(drift_log_msg % 'test')
            self._data_drift(base_df, test_df,
                             'data_drift_within_test_dataset')

        # Write report to YAML file
        logging.info('Writing report in yaml file')
        utils.to_yaml(self.report_fp, self.validation_report)

        artifact = DataValidationArtifact(
            self.feature_store_fp,
            self.train_path,
            self.test_path,
            self.report_fp,
        )
        logging.info('Data validation artifact: %s', artifact)
        return artifact
