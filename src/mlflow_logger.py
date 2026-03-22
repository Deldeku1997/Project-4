import os
import sys
import mlflow
import mlflow.sklearn
import joblib

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_PATH

mlflow.set_experiment("RealEstateAdvisor")

clf,reg,_=joblib.load(MODEL_PATH)

with mlflow.start_run():
    mlflow.sklearn.log_model(clf,"classification_model")
    mlflow.sklearn.log_model(reg,"regression_model")
    mlflow.log_param("algorithm","RandomForest")