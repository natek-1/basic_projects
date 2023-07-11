import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise ChildProcessError(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_models = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.key)[i]]

            grid_search = GridSearchCV(model, param, cv=3, scoring='r2', n_jobs=-1, verbose=2)

            grid_search.fit(X_train, y_train)

            y_train_pred = grid_search.best_score_
            y_test_pred = grid_search.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(model.keys())[i]] = test_model_score, 
            best_models[list(model.keys())[i]]  = grid_search.best_estimator_
        return report, best_models
    except Exception as e:
        raise CustomException(e, sys)
