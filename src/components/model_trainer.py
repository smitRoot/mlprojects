import os
import sys
from dataclasses import dataclass


from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split trainibg and test input data " )
            xtrain,ytrain,xtest,ytest=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
            "Random forest ": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "LinearRegression": LinearRegression(),
            "K-neoghbour classifier": KNeighborsRegressor(),
            "XGB classifier": XGBRegressor(),
            "cataboosting classifier ": CatBoostRegressor(verbose=False),
            "AdaBoostRegressor": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_model(xtrain,ytrain,xtest,ytest,models)

            best_model_score= max(sorted(model_report.values()))

            best_model_names=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_names]

            if best_model_score<0.6:
                raise CustomException("No best model found ")
            logging.info(f"best model found on both training and test data set")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model            )
            
            predicted = best_model.predict(xtest )
            r2_square=r2_score(ytest,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)

