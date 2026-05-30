import os 
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        import pickle
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):

    try:

        report = {}
        best_models = {}

        for model_name, model in models.items():

            para = param[model_name]

            gs = GridSearchCV(
                model,
                para,
                cv=3,
                n_jobs=-1,
                scoring="f1"
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            best_model.fit(X_train, y_train)

            y_test_pred = best_model.predict(X_test)

            test_score = f1_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_score

            best_models[model_name] = best_model

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)