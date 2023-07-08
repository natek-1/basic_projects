import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_trasformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data tranformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # Handling missing values
                    ('scaler', StandardScaler()) # Scaling the data
                ]
            )

            logging.info('Numerical column scaling colpleted')

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ('scaler', StandardScaler()) 
                ]
            )
            logging.info("Categorical coumns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipline', num_pipeline, numerical_columns),
                    ('cat_pipline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
