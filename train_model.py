import pandas as pd

from feat_eng import feat_eng
from model import optimize_xgboost


def train_model(train_data_path, n_trials=20):
    # Load data
    train_data = pd.read_csv(train_data_path)

    # Preprocess data
    X_train, X_test, y_train, y_test = feat_eng(train_data, train=True)

    # Train model
    final_model, best_params, study = optimize_xgboost(
        X_train, y_train, X_test, y_test, n_trials=n_trials
    )

    return final_model, best_params, study