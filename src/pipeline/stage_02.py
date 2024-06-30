from image_classifier.config.configuration import ConfigurationManager
from image_classifier.components.train import Training
from image_classifier import logger


STAGE_NAME = "Train Model"
class TrainingPipeline:
    def __int__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

if __name__ == '__main__':
    try:
        logger.info(f"starting...{STAGE_NAME}")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f"completed...{STAGE_NAME}")
    except Exception as e:
        raise e

