import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config =ModelTrainerConfig()
    
    def initial_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                    "Linear Regression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "XGBRegressor": XGBRegressor(),
                    #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    'AdaBoost Regressor': AdaBoostRegressor()
                }
                
            
            params={
                "Linear Regression":{},
                "Lasso": {
                    'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,0.1, 0.2, 0.3, 0.5, 0.8,1]
                },
                "Ridge":{
                    'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
               "K-Neighbors Regressor": {
                    'n_neighbors':[5, 4, 8, 3, 6, 10],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                    'max_depth': range(10, 150, 10) + None,
                    'max_leaf_nodes': range(50, 150) + None,
                },
                "Random Forest Regressor":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth': range(10, 150, 10) + None,
                    'max_leaf_nodes': range(50, 150) + None,
                },                
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                } 
            }
            model_report, best_models = evaluate_model(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test , models=models, params=params)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = best_models[best_model_name]
            logging.info(model_report)

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info('best model found for the dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    model_trainer = ModelTrainer()
    
    data_ingestor = DataIngestion()
    train_data, test_data, _ = data_ingestor.initiate_data_ingestion()
    data_transormation = DataTransformation()
    train_array, test_array, obj = data_transormation.initiate_data_transformation(train_data, test_data)
    score = model_trainer.initial_model_trainer(train_array, test_array, obj)
    logging.info(f"best model score found: {score}")



