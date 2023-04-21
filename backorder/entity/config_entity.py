from datetime import datetime as dt
from pathlib import Path

from backorder.config import BASE_DATA_NAME


class TrainingPipelineConfig:
    def __init__(self):
        self.root = Path.cwd()
        self.artifact_dir = Path('artifacts', dt.now().strftime('%m%d%y__%H'))
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


class DataIngestionConfig(TrainingPipelineConfig):
    def __init__(self):
        super().__init__()
        self.base_data_fp = self.root / 'data' / BASE_DATA_NAME
        self.dir = self.artifact_dir / 'data_ingestion'
        self.feature_store_fp = self.dir / 'feature_store' / BASE_DATA_NAME
        self.train_path = self.dir / 'dataset' / 'train.parquet'
        self.test_path = self.dir / 'dataset' / 'test.parquet'
        self.test_size = 0.2
        self.__create_all_dirs()

    def to_dict(self) -> dict:
        """ Convert data into dict """
        return self.__dict__

    def __create_all_dirs(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.feature_store_fp.parent.mkdir(exist_ok=True)
        self.train_path.parent.mkdir(exist_ok=True)
        self.test_path.parent.mkdir(exist_ok=True)


class DataValidationConfig(DataIngestionConfig):
    def __init__(self):
        super().__init__()
        self.dir = self.artifact_dir / 'data_validation'
        self.report_fp = self.dir / 'report.yaml'
        self.missing_threshold = 0.2
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(exist_ok=True)


class DataTransformationConfig(DataIngestionConfig):
    def __init__(self):
        super().__init__()
        self.dir = self.artifact_dir / 'data_transformation'
        self.transformer_pkl_fp = self.dir / 'transformer.pkl'
        self.target_enc_fp = self.dir / 'target_encoder.pkl'
        self.train_npz_path = self.dir / 'transformed' / 'train.npz'
        self.test_npz_path = self.dir / 'transformed' / 'test.npz'
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(exist_ok=True)
        self.train_npz_path.parent.mkdir(parents=True, exist_ok=True)


class ModelTrainerConfig(TrainingPipelineConfig):
    def __init__(self):
        super().__init__()
        self.dir = self.artifact_dir / 'model_trainer'
        self.model_path = self.dir / 'model.pkl'
        self.expected_score = 0.7
        self.overfitting_threshold = 0.3
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(exist_ok=True)


class ModelEvaluationConfig(TrainingPipelineConfig):
    def __init__(self):
        super().__init__()
        self.change_threshold = 0.01


class ModelPusherConfig(TrainingPipelineConfig):
    def __init__(self):
        super().__init__()
        self.dir = self.artifact_dir / 'model_pusher'
        self.model_path = self.dir / 'model.pkl'
        self.transformer_path = self.dir / 'transformer.pkl'
        self.target_enc_path = self.dir / 'target_encoder.pkl'
        # Store the latest models and datasets at root directory
        self.root_saved_model_dir = self.root / 'saved_models'
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(exist_ok=True)
        self.root_saved_model_dir.mkdir(exist_ok=True)
