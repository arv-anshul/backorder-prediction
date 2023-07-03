""" Stored Model entity to track recently stored trained model. """

from pathlib import Path

from backorder.config import STORED_MODEL_PATH
from backorder.logger import logging


class StoredModelConfig:
    def __init__(self) -> None:
        self.model_registry = STORED_MODEL_PATH
        self.model_registry.mkdir(exist_ok=True)

        self.latest_stored_dir = self.__get_latest_stored_dir_path()
        self.new_dir_to_store_models = self.__get_new_dir_path_to_store()

    def __get_latest_stored_dir_path(self) -> Path | None:
        dir_names = [int(i.name) for i in self.model_registry.iterdir()]
        if len(dir_names) == 0:
            return None
        return self.model_registry / str(max(dir_names))

    def __get_new_dir_path_to_store(self) -> Path:
        if self.latest_stored_dir == None:
            return self.model_registry / str(0)
        latest_dir_number = int(self.latest_stored_dir.name) + 1
        return self.model_registry / str(latest_dir_number)

    @property
    def stored_model_path(self):
        if self.latest_stored_dir is None:
            error_msg = 'Model is not available.'
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        return self.latest_stored_dir / 'model.pkl'

    @property
    def stored_transformer_path(self):
        if self.latest_stored_dir is None:
            error_msg = 'Transformer is not available.'
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        return self.latest_stored_dir / 'transformer.pkl'

    @property
    def stored_target_enc_path(self):
        if self.latest_stored_dir is None:
            error_msg = 'Target encoder is not available.'
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        return self.latest_stored_dir / 'target_encoder.pkl'

    @property
    def path_to_store_model(self):
        return self.new_dir_to_store_models / 'model.pkl'

    @property
    def path_to_store_transformer(self):
        return self.new_dir_to_store_models / 'transformer.pkl'

    @property
    def path_to_store_target_enc(self):
        return self.new_dir_to_store_models / 'target_encoder.pkl'
