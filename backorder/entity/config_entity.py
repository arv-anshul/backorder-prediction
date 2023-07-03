from datetime import datetime as dt
from pathlib import Path

from backorder.config import BASE_DATA_NAME, STORED_MODEL_PATH


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
        self.num_cols = [
            'national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month',
            'forecast_6_month', 'forecast_9_month', 'sales_1_month',
            'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank',
            'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg',
            'local_bo_qty'
        ]
        self.obj_cols = [
            'potential_issue', 'deck_risk',
            'oe_constraint', 'ppap_risk',
            'stop_auto_buy', 'rev_stop'
        ]
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


class ModelEvaluationConfig:
    def __init__(self):
        self.change_threshold = 0.01


class ModelPusherConfig(TrainingPipelineConfig):
    def __init__(self):
        super().__init__()
        self.dir = self.artifact_dir / 'model_pusher'
        self.model_path = self.dir / 'model.pkl'
        self.transformer_path = self.dir / 'transformer.pkl'
        self.target_enc_path = self.dir / 'target_encoder.pkl'
        # Store the latest models and datasets at root directory
        self.root_stored_model_dir = STORED_MODEL_PATH
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(exist_ok=True)
