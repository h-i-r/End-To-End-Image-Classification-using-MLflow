import os.path
from image_classifier.utils.common import read_yaml, create_dir
from image_classifier.entity.config_entity import PrepareBaseModelConfig, TrainingConfig, EvaluateConfig
from image_classifier.constants import *
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_dir([self.config.artifacts_root])

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.base_model
        create_dir([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path= Path(config.base_model_path),
            updated_base_model_path= Path(config.updated_base_model_path),
            params_image_size= self.params.IMAGE_SIZE,
            params_learning_rate= self.params.LEARNING_RATE,
            params_include_top= self.params.INCLUDE_TOP,
            params_weights= self.params.WEIGHTS,
            params_classes= self.params.CLASSES)
        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.base_model
        params = self.params
        project_path = os.path.join('src', 'image_classifier', 'components', 'data')
        training_data = os.path.join(os.getcwd(), project_path)
        create_dir([training.root_dir])
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path= Path(training.trained_model_path),
            updated_base_model_path= Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs= params.EPOCHS,
            params_batch_size= params.BATCH_SIZE,
            params_is_augmented= params.AUGMENTATION,
            params_image_size= params.IMAGE_SIZE)
        return training_config

    def get_evaluation_config(self) -> EvaluateConfig:
        project_path = os.path.join('src', 'image_classifier', 'components', 'data')
        training_data = os.path.join(os.getcwd(), project_path)
        eval_config = EvaluateConfig(
            path_to_model=Path("artifacts/train/model.h5"),
            training_data=Path(training_data),
            all_params= self.params,
            mlflow_uri=os.environ["MLFLOW_TRACKING_URI"],
            params_image_size= self.params.IMAGE_SIZE,
            params_batch_size= self.params.BATCH_SIZE
        )
        return eval_config




