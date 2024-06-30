from image_classifier.config.configuration import ConfigurationManager
from image_classifier.components.evaluate import Evaluation
from image_classifier import logger


STAGE_NAME = "Evaluate Model"
class EvaluationPipeline:
    def __int__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f"starting...{STAGE_NAME}")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"completed...{STAGE_NAME}")
    except Exception as e:
        raise e

