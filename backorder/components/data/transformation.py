""" Data Transformation """

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

from backorder import utils
from backorder.config import TARGET_COLUMN
from backorder.entity import (DataTransformationArtifact,
                              DataTransformationConfig)
from backorder.logger import logging


@utils.wrap_with_custom_exception
class DataTransformation(DataTransformationConfig):
    def __init__(self):
        """ To initiate transformation process with train and test dataset. """
        super().__init__()
        logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")

    @classmethod
    def get_transformer_object(cls) -> Pipeline:
        simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
        robust_scaler = RobustScaler()

        pipeline = Pipeline(steps=[
            ('Imputer', simple_imputer),
            ('RobustScaler', robust_scaler),
        ])

        return pipeline

    def initiate(self) -> DataTransformationArtifact:
        # Reading training and testing file
        train_df = utils.read_dataset(self.train_path)
        test_df = utils.read_dataset(self.test_path)

        # Selecting input feature from train and test data
        X_train_df = train_df.drop(TARGET_COLUMN, axis=1)
        X_test_df = test_df.drop(TARGET_COLUMN, axis=1)

        # Selecting target feature from train and test data
        y_train_df = train_df[TARGET_COLUMN]
        y_test_df = test_df[TARGET_COLUMN]

        label_enc = LabelEncoder()

        # Transformation on target columns
        y_train_arr = y_train_df.squeeze()
        y_test_arr = y_test_df.squeeze()

        # Transformation on categorical columns
        for col in X_train_df.columns:
            if X_test_df[col].dtype == 'O':
                X_train_df[col] = label_enc.fit_transform(X_train_df[col])
                X_test_df[col] = label_enc.fit_transform(X_test_df[col])

        trf_pipeline = DataTransformation.get_transformer_object()
        trf_pipeline.fit(X_train_df)

        # Transforming input features
        X_train_arr = trf_pipeline.transform(X_train_df)
        X_test_arr = trf_pipeline.transform(X_test_df)

        train_arr = np.c_[X_train_arr, y_train_arr]
        test_arr = np.c_[X_test_arr, y_test_arr]

        # Objects dumping
        utils.dump_array(self.train_npz_path, train_arr)
        utils.dump_array(self.test_npz_path, test_arr)

        utils.dump_object(self.transformer_pkl_fp, trf_pipeline)
        utils.dump_object(self.target_enc_fp, label_enc)

        artifact = DataTransformationArtifact(
            self.transformer_pkl_fp, self.target_enc_fp,
            self.train_npz_path, self.test_npz_path,
        )

        logging.info('Data transformation object %s', artifact)
        return artifact
