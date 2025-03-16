import xgboost as xgb
import optuna
from functools import partial

def objective(trial, X_train, y_train):
    """
    Objective function for Optuna to optimize XGBoost hyperparameters.
    
    Parameters:
    -----------
    trial : optuna.trial.Trial
        Trial object for hyperparameter optimization
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    float
        Mean CV score to be maximized
    """
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_float('alpha', 0, 10),
        'lambda': trial.suggest_float('lambda', 0, 10),
       
        # Fixed parameters (not optimized)
        'objective': 'binary:logistic',
        'eval_metric': ['error', 'auc'],
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'seed': 42
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
        metrics=['auc'],
        seed=42,
        verbose_eval=False
    )
   
    # Return best score (Optuna will maximize this)
    return cv_results['test-auc-mean'].values[-1]

def optimize_xgboost(X_train, y_train, X_test, y_test, n_trials=15):
    """
    Performs hyperparameter optimization for XGBoost using Optuna and trains a final model
    with the best parameters found.
   
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    n_trials : int, default=15
        Number of Optuna trials for hyperparameter optimization
   
    Returns:
    --------
    final_model : xgb.Booster
        Trained XGBoost model with best parameters
    best_params : dict
        Dictionary of best hyperparameters
    study : optuna.study.Study
        Optuna study object containing optimization results
    """
    # Step 1: Create and run Optuna study
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        partial(objective, X_train=X_train, y_train=y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Step 2: Get best parameters
    best_params = study.best_params
    # Add fixed parameters to best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': ['error', 'auc'],
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'seed': 42
    })
    
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
        evals=[(dval, 'validation')],
        verbose_eval=100
    )
    
    print(f"\nFinal model trained with {final_model.best_iteration} boosting rounds")
    print(f"Best validation AUC: {final_model.best_score:.4f}")
    
    return final_model, best_params, study