""" Trained Model Evaluation """

from sklearn.metrics import accuracy_score, r2_score

from backorder import utils
from backorder.config import PREDICTION_TYPE, TARGET_COLUMN
from backorder.entity.artifact_entity import (DataIngestionArtifact,
                                              DataTransformationArtifact,
                                              ModelEvaluationArtifact,
                                              ModelTrainerArtifact)
from backorder.entity.config_entity import ModelEvaluationConfig
from backorder.entity.stored_model_entity import StoredModelConfig
from backorder.logger import logging


@utils.wrap_with_custom_exception
class ModelEvaluation(ModelEvaluationConfig):
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> None:
        """
        Evaluate the new model with older model and store the new model;
        iff new model is better than than older one.
        """
        logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
        self.data_ingestion_artifact = data_ingestion_artifact
        self.trf_artifact = data_transformation_artifact
        self.trainer_artifact = model_trainer_artifact
        self.stored_models = StoredModelConfig()
        self.prediction_type = PREDICTION_TYPE

    def __load_stored_objects(self, model_fp, transformer_fp, target_enc_fp) -> tuple:
        model = utils.load_object(model_fp)
        transformer = utils.load_object(transformer_fp)
        target_enc = utils.load_object(target_enc_fp)
        return model, transformer, target_enc

    def __evaluate_score(self, y_true, y_pred):
        if self.prediction_type == 'regression':
            return r2_score(y_true, y_pred)
        else:
            return accuracy_score(y_true, y_pred)

    def initiate(self) -> ModelEvaluationArtifact:
        # If stored model folder has model the we will compare
        # which model is best trained or the model from saved model folder

        if self.stored_models.latest_stored_dir == None:
            artifact = ModelEvaluationArtifact(True, 0)
            logging.info(artifact)
            return artifact

        logging.info('Importing stored trained objects.')
        model, transformer, target_enc = self.__load_stored_objects(
            self.stored_models.stored_model_path,
            self.stored_models.stored_transformer_path,
            self.stored_models.stored_target_enc_path,
        )

        logging.info('Newly trained model objects')
        new_model, new_transformer, _ = self.__load_stored_objects(
            self.trainer_artifact.model_path,
            self.trf_artifact.transformer_pkl,
            self.trf_artifact.target_enc_fp,
        )

        # --- --- Old Model Evaluation --- --- #
        logging.info(f"{'---'*10} Old Model Evaluation {'---'*10}")
        test_df = utils.read_dataset(self.data_ingestion_artifact.test_path)
        y_true = test_df[TARGET_COLUMN]

        input_feature_name = list(transformer.feature_names_in_)
        for i in input_feature_name:
            if test_df[i].dtype == 'O':
                test_df[i] = target_enc.fit_transform(test_df[i])

        input_arr = transformer.transform(test_df[input_feature_name])
        y_pred = model.predict(input_arr)

        old_score = self.__evaluate_score(y_true, y_pred)
        logging.info('Score of old model: %s', old_score)

        # --- --- New Model Evaluation --- --- #
        logging.info(f"{'---'*10} New Model Evaluation {'---'*10}")
        input_feature_name = list(new_transformer.feature_names_in_)
        input_arr = new_transformer.transform(test_df[input_feature_name])
        y_pred = new_model.predict(input_arr)

        current_score = self.__evaluate_score(y_true, y_pred)
        logging.info('Score of current model: %s', current_score)

        if current_score <= old_score:
            error_msg = 'New trained model is not better than old model'
            logging.error(error_msg)
            raise ValueError(error_msg)

        score_diff = current_score - old_score
        artifact = ModelEvaluationArtifact(True, score_diff)  # type: ignore
        logging.info(artifact)
        return artifact
