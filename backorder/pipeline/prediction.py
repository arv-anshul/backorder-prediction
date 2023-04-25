""" Predict the input file and store. """

from datetime import datetime as dt
from pathlib import Path

from pandas import DataFrame

from backorder import utils
from backorder.entity import StoredModelConfig
from backorder.logger import logging

PREDICTION_DIR = Path('prediction')


@utils.wrap_with_custom_exception
class Prediction:
    def __init__(self, input_fp: Path) -> None:
        """ Prediction using transformed model. """
        logging.info(f"{'>>'*20} Prediction {'<<'*20}")

        self.input_data_path = input_fp
        PREDICTION_DIR.mkdir(exist_ok=True)
        self.predicted_csv_fp = (
            PREDICTION_DIR /
            self.input_data_path.name
        ).with_stem(f'{dt.now():%m%d%Y__%H%M%S}')

    @classmethod
    def predict_one(cls, one_row_df: DataFrame):
        df = one_row_df

        logging.info('Loading pickled transformers to transform dataset.')
        model, transformer, target_enc = cls.__get_stored_transformers()

        input_feature_names = list(transformer.feature_names_in_)
        for i in input_feature_names:
            if df[i].dtype == 'O':
                df[i] = target_enc.fit_transform(df[i])
        input_arr = transformer.transform(df[input_feature_names])
        prediction = model.predict(input_arr)

        df['prediction'] = prediction
        return df

    @classmethod
    def __get_stored_transformers(cls):
        stored_models_config = StoredModelConfig()

        transformer_path = stored_models_config.stored_transformer_path
        target_enc_path = stored_models_config.stored_target_enc_path
        model_path = stored_models_config.stored_model_path

        transformer = utils.load_object(transformer_path)
        target_enc = utils.load_object(target_enc_path)
        model = utils.load_object(model_path)

        return model, transformer, target_enc

    def __clean_df(self, df: DataFrame):
        """ Custom cleaning for df. """
        df = df.drop(columns=['sku'])

        return df

    def initiate(self):
        logging.info('Reading file for prediction: %s', self.input_data_path)
        df = utils.read_dataset(self.input_data_path)
        df = self.__clean_df(df)

        logging.info('Loading pickled transformers to transform dataset.')
        model, transformer, target_enc = self.__get_stored_transformers()

        # TODO We need to create label encoder object for each categorical variable.
        input_feature_names = list(transformer.feature_names_in_)
        for i in input_feature_names:
            if df[i].dtype == 'O':
                df[i] = target_enc.fit_transform(df[i])
        input_arr = transformer.transform(df[input_feature_names])
        prediction = model.predict(input_arr)

        df['prediction'] = prediction
        df.to_csv(self.predicted_csv_fp, index=False, header=True)
        return self.predicted_csv_fp
