import os
import sys

from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing data"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                ),

                "Random Forest": RandomForestClassifier(
                    random_state=42
                ),

                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    random_state=42
                )

            }

            params = {

                "Decision Tree": {

                    "criterion": [
                        "gini",
                        "entropy"
                    ],

                    "max_depth": [
                        3,
                        5,
                        10,
                        15,
                        None
                    ],

                    "min_samples_split": [
                        2,
                        5,
                        10
                    ],

                    "min_samples_leaf": [
                        1,
                        2,
                        4
                    ]
                },

                "Random Forest": {

                    "n_estimators": [
                        100,
                        200,
                        300
                    ],

                    "max_depth": [
                        5,
                        10,
                        20,
                        None
                    ],

                    "min_samples_split": [
                        2,
                        5,
                        10
                    ],

                    "min_samples_leaf": [
                        1,
                        2,
                        4
                    ]
                },
                "XGBoost": {

                    "n_estimators": [
                        100,
                        200,
                        300
                    ],

                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1,
                        0.2
                    ],

                    "max_depth": [
                        3,
                        5,
                        7,
                        10
                    ],

                    "subsample": [
                        0.8,
                        1.0
                    ],

                    "colsample_bytree": [
                        0.8,
                        1.0
                    ]
                }

            }

            model_report, best_models = evaluate_models(

                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,

                models=models,
                param=params

            )

            logging.info("All Model Scores")

            for name, score in model_report.items():

                logging.info(
                    f"{name} : {score}"
                )

            print(model_report)

            best_model_score = max(
                model_report.values()
            )

            best_model_name = list(
                model_report.keys()
            )[
                list(
                    model_report.values()
                ).index(
                    best_model_score
                )
            ]

            best_model = models[
                best_model_name
            ]

            logging.info(
                f"Best Model Name : {best_model_name}"
            )

            logging.info(
                f"Best Model Score : {best_model_score}"
            )

            best_model.fit(
                X_train,
                y_train
            )

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model

            )

            predicted = best_model.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predicted
            )

            precision = precision_score(
                y_test,
                predicted
            )

            recall = recall_score(
                y_test,
                predicted
            )

            f1 = f1_score(
                y_test,
                predicted
            )

            cm = confusion_matrix(
                y_test,
                predicted
            )

            logging.info(
                f"Accuracy : {accuracy}"
            )

            logging.info(
                f"Precision : {precision}"
            )

            logging.info(
                f"Recall : {recall}"
            )

            logging.info(
                f"F1 Score : {f1}"
            )

            logging.info(
                f"Confusion Matrix :\n{cm}"
            )

            print("\n========== FINAL RESULTS ==========")
            print("Accuracy :", accuracy)
            print("Precision :", precision)
            print("Recall :", recall)
            print("F1 Score :", f1)
            print("Confusion Matrix :")
            print(cm)

            return f1

        except Exception as e:

            raise CustomException(
                e,
                sys
            )