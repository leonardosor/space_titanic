import pandas as pd
import argparse
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

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train XGBoost model on Space Titanic data")
    parser.add_argument("train_data_path", type=str, nargs='?', default="data/train.csv", 
                        help="Path to training data CSV file")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials for hyperparameter optimization")
    args = parser.parse_args()
    
    # Train model using parsed arguments
    final_model, best_params, study = train_model(args.train_data_path, args.n_trials)
    
    print("\nTraining complete!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {study.best_value:.4f}")