from image_classifier import logger
from image_classifier.pipeline.stage_01 import PrepareBaseModelTrainingPipeline
from image_classifier.pipeline.stage_02 import TrainingPipeline
from image_classifier.pipeline.stage_03 import EvaluationPipeline
from image_classifier.pipeline.stage_04 import PredictPipeline
import os

os.environ["MLFLOW_TRACKING_PASSWORD"]=MLFLOW_TRACKING_PASSWORD
os.environ["MLFLOW_TRACKING_USERNAME"]=MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_URI"]=MLFLOW_TRACKING_URI

STAGE_NAME = "Load Data"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} initiating <<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage  {STAGE_NAME} successfully completed <<<<<")

except Exception as e:
    raise e

STAGE_NAME = "Train Model"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} initiating <<<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage  {STAGE_NAME} successfully completed <<<<<")
except Exception as e:
    raise e

STAGE_NAME = "Evaluate Model"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} initiating <<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>> Stage  {STAGE_NAME} successfully completed <<<<<")
except Exception as e:
    raise e

STAGE_NAME = "Make Predictions"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} initiating <<<<<")
    obj = PredictPipeline()
    obj.main()
    logger.info(f">>>>> Stage  {STAGE_NAME} successfully completed <<<<<")
except Exception as e:
    raise e

