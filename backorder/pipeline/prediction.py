""" Predict the input file and store. """

from pathlib import Path

from pandas import DataFrame

from backorder import utils
from backorder.entity import StoredModelConfig
from backorder.logger import logging

PREDICTION_DIR = Path('prediction')


@utils.wrap_with_custom_exception
class Prediction:
    def __init__(self) -> None:
        """Prediction using transformed model."""
        logging.info(f"{'>>'*20} Prediction {'<<'*20}")

        PREDICTION_DIR.mkdir(exist_ok=True)
        self.predicted_csv_fp = PREDICTION_DIR / 'prediction.csv'

    @staticmethod
    def one_prediction(df: DataFrame):
        logging.info('Loading pickled transformers to transform dataset.')
        model, transformer, target_enc = Prediction.get_stored_transformers()

        input_arr = transformer.transform(df[transformer.feature_names_in_])
        prediction = model.predict(input_arr)

        df['backorder_prediction'] = target_enc.inverse_transform(prediction.astype(int))

        fp = Path(f'{PREDICTION_DIR}/pred.csv')
        fp.parent.mkdir(exist_ok=True)
        df.to_csv(fp, index=False)

        return df

    @staticmethod
    def get_stored_transformers():
        stored_models_config = StoredModelConfig()

        transformer_path = stored_models_config.stored_transformer_path
        target_enc_path = stored_models_config.stored_target_enc_path
        model_path = stored_models_config.stored_model_path

        transformer = utils.load_object(transformer_path)
        target_enc = utils.load_object(target_enc_path)
        model = utils.load_object(model_path)

        return model, transformer, target_enc

    def batch_prediction(self, df: DataFrame = ..., csv_fp: Path = ...) -> Path:
        if isinstance(csv_fp, Path) and csv_fp.suffix == '.csv':
            logging.info('Reading file for prediction: %s', csv_fp)
            df = utils.read_dataset(csv_fp)
        elif isinstance(df, DataFrame):
            ...
        else:
            raise ValueError('Pass either df or csv_path.')

        logging.info('Loading pickled transformers to transform dataset.')
        model, transformer, target_enc = Prediction.get_stored_transformers()

        input_arr = transformer.transform(df[transformer.feature_names_in_])
        prediction = model.predict(input_arr)

        df['backorder_prediction'] = target_enc.inverse_transform(prediction.astype(int))
        df.to_csv(self.predicted_csv_fp, index=False, header=True)
        return self.predicted_csv_fp
