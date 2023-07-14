from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src. components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging
import sys

try:
    data_ingestor = DataIngestion()
    train_data, test_data, _ = data_ingestor.initiate_data_ingestion()

    data_transormation = DataTransformation()
    train_array, test_array, obj = data_transormation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    score = model_trainer.initial_model_trainer(train_array, test_array, obj)
    logging.info(f"best model score found: {score}")
except Exception as e:
    raise CustomException(e, sys)