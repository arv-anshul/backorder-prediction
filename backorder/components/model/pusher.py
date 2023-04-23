""" Model Pusher """

from backorder import utils
from backorder.entity import (DataTransformationArtifact, ModelPusherArtifact,
                              ModelPusherConfig, ModelTrainerArtifact,
                              StoredModelConfig)
from backorder.logger import logging


@utils.wrap_with_custom_exception
class ModelPusher(ModelPusherConfig):
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        """
        Dump the transformed models to `./stored_models` and `./artifacts/model_pusher` directory.
        """
        super().__init__()
        logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
        self.data_trf_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.stored_model_config = StoredModelConfig()

    def initiate(self) -> ModelPusherArtifact:
        logging.info('Loading dumped models from artifacts directory.')
        transformer = utils.load_object(self.data_trf_artifact.transformer_pkl)
        model = utils.load_object(self.model_trainer_artifact.model_path)
        target_enc = utils.load_object(self.data_trf_artifact.target_enc_fp)

        logging.info('Dumping models to `./artifacts/model_pusher` directory.')
        utils.dump_object(self.transformer_path, transformer)
        utils.dump_object(self.model_path, model)
        utils.dump_object(self.target_enc_path, target_enc)

        logging.info('Dumping models to `./stored_models` directory.')
        model_path = self.stored_model_config.path_to_store_model
        transformer_path = self.stored_model_config.path_to_store_transformer
        target_enc_path = self.stored_model_config.path_to_store_target_enc

        utils.dump_object(transformer_path, transformer)
        utils.dump_object(model_path, model)
        utils.dump_object(target_enc_path, target_enc)

        artifact = ModelPusherArtifact(self.dir, self.root_stored_model_dir)
        logging.info(artifact)
        return artifact
