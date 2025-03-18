import os
import pickle
import warnings

import pandas as pd
import xgboost as xgb

from feat_eng import feat_eng
from model import optimize_xgboost

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

def train_model(data_path, train_flag, n_trials=20, model_path="model.pkl"):
    if not train_flag:
        raise ValueError("train_flag must be True for training mode")
    
    # Load data
    try:
        train_data = pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Failed to load training data: {str(e)}")
    
    # Preprocess data
    try:
        X_train, X_test, y_train, y_test = feat_eng(train_data, train=True)
    except Exception as e:
        raise Exception(f"Feature engineering failed: {str(e)}")
    
    # Train model
    try:
        final_model, best_params, study = optimize_xgboost(
            X_train, y_train, X_test, y_test, n_trials=n_trials
        )
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
            
        return final_model, best_params, study
    except Exception as e:
        raise Exception(f"Model training failed: {str(e)}")

def inference(data_path, model_path="model.pkl"):
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load data
    try:
        inference_data = pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Failed to load inference data: {str(e)}")
    
    # Preprocess data (with train=False)
    try:
        X = feat_eng(inference_data, train=False)
    except Exception as e:
        raise Exception(f"Feature engineering failed: {str(e)}")
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")
    
    # Make predictions
    try:
        predictions = pd.DataFrame({'Predicted': model.predict(X)})
        return predictions
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")