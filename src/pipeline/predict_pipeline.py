import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self, model_path:str = "artifacts/model.pkl", preprocessor_path: str = "artifacts/preprocessor.pkl"):
        self.model = load_object(model_path)
        self.preprocessor = load_object(preprocessor_path)
        logging.info("loaded model and preprocessor")

    def predict(self, features):
        try:
            data_preprocessed = self.preprocessor.transform(features)
            logging.info("sent feautures was preprocessed")
            prediction = self.model.predict(data_preprocessed)
            logging.info("made predictions from given features")
            return prediction
        except Exception as e:
            raise CustomException(e, sys)




class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str,
        test_preparation_course: int,
        reading_score: int,
        writing_score: int):
        self.gender = gender
        self.race_ethinicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'race/ethnicity': [self.race_ethinicity],
                'parental level of education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test preparation course': [self.test_preparation_course],
                'reading score': [self.reading_score],
                'writing score': [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

