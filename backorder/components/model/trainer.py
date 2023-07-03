""" Train models and store it. """

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from backorder import utils
from backorder.config import PREDICTION_TYPE
from backorder.entity import (DataTransformationConfig, ModelTrainerArtifact,
                              ModelTrainerConfig)
from backorder.logger import logging


@utils.wrap_with_custom_exception
class ModelTrainer(ModelTrainerConfig):
    def __init__(self):
        """ To initiate model training process with dumped datasets. """

        super().__init__()
        self.data_trf_config = DataTransformationConfig()
        self.prediction_type = PREDICTION_TYPE

        logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
        logging.info('Prediction type: %s', self.prediction_type)

    def _get_train_test_data(self):
        logging.info('Loading train and test array.')
        train_arr = utils.load_array(self.data_trf_config.train_npz_path)
        test_arr = utils.load_array(self.data_trf_config.test_npz_path)

        logging.info('Splitting into X and y from train and test array.')
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        return X_train, X_test, y_train, y_test

    def _fine_tune(self):
        ...

    def _evaluate(self, model, X_train, X_test, y_train, y_test):
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)

        train_score = accuracy_score(y_train, y_hat_train)
        test_score = accuracy_score(y_test, y_hat_test)

        logging.info('Train score: %s', train_score)
        logging.info('Tests score: %s', test_score)
        return train_score, test_score

    def _check_model_fitting(self, train_score, test_score):
        logging.info(f'Checking if our model is underfit or not')
        if test_score < self.expected_score:
            error_msg = (f'Expected score: {self.expected_score}\n'
                         f'Actual score: {test_score}')
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.info(f'Checking if our model is overfit or not')
        diff = abs(train_score - test_score)
        if diff > self.overfitting_threshold:
            error_msg = (
                f'Train-Test score diff: {diff}\n'
                f'Overfitting threshold: {self.overfitting_threshold}'
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

    def initiate(self) -> ModelTrainerArtifact:
        X_train, X_test, y_train, y_test = self._get_train_test_data()

        logging.info('Train the model')
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        train_score, test_score = self._evaluate(
            model, X_train, X_test, y_train, y_test
        )
        self._check_model_fitting(train_score, test_score)

        logging.info('Dumping trained model object.')
        utils.dump_object(self.model_path, model)

        artifact = ModelTrainerArtifact(
            self.model_path, train_score, test_score    # type: ignore
        )
        logging.info(f'Model trainer artifact: {artifact}')
        return artifact
