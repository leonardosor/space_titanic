import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

from functools import partial

import optuna
import xgboost as xgb


def objective(trial, X_train, y_train):
    """
    Objective function for Optuna to optimize XGBoost hyperparameters.

    Parameters:
    -----------
    :param trial: Trial object for hyperparameter optimization
    :param X_train: array-like training features
    :param y_train : array-like training target
    :returns model: mean CV score to be maximized
    """
    param = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "alpha": trial.suggest_float("alpha", 0, 10),
        "lambda": trial.suggest_float("lambda", 0, 10),
        # Fixed parameters (not optimized)
        "objective": "binary:logistic",
        "eval_metric": ["error", "auc"],
        "tree_method": "gpu_hist",
        "device": "cuda",
        "seed": 42,
    }

    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Cross-validation
    cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=1000,
        nfold=5,
        early_stopping_rounds=50,
        metrics=["auc"],
        seed=42,
        verbose_eval=False,
    )

    # Return best score (Optuna will maximize this)
    return cv_results["test-auc-mean"].values[-1]


def optimize_xgboost(X_train, y_train, X_test, y_test, n_trials=15):
    """
    Performs hyperparameter optimization for XGBoost using Optuna and trains a final model
    with the best parameters found.

    :param X_train : array-like training features
    :param y_train : array-like Training target
    :param X_test : array-like test features
    :param y_test : array-like test target
    :n_trials : int, default=15 # of Optuna trials for hyperparameter optimization
    :return final_model : xgb.Booster Trained XGBoost model with best parameters
    :return best_params : dict Dictionary of best hyperparameters
    :return study : optuna.study.Study Optuna study object containing optimization
    """
    # Step 1: Create and run Optuna study
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(objective, X_train=X_train, y_train=y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Step 2: Get best parameters
    best_params = study.best_params
    # Add fixed parameters to best_params
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": ["error", "auc"],
            "tree_method": "gpu_hist",
            "device": "cuda",
            "seed": 42,
        }
    )

    # Print optimization results
    print("\nBest parameters:", best_params)
    print(f"Best CV score: {study.best_value:.4f}")

    # Step 3: Train final model with best parameters
    print("\nTraining final model with best parameters...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=10000,
        early_stopping_rounds=50,
        evals=[(dval, "validation")],
        verbose_eval=100,
    )

    print(f"\nFinal model trained with {final_model.best_iteration} boosting rounds")
    print(f"Best validation AUC: {final_model.best_score:.4f}")

    return final_model, best_params, study
