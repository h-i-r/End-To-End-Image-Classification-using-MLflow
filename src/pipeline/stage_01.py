from image_classifier.config.configuration import ConfigurationManager
from image_classifier.components.model import PrepareBaseModel
from image_classifier import logger


STAGE_NAME = "Data Load"
class PrepareBaseModelTrainingPipeline:
    def __int__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

if __name__ == '__main__':
    try:
        logger.info(f"starting...{STAGE_NAME}")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f"completed...{STAGE_NAME}")
    except Exception as e:
        raise e

