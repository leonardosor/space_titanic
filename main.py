import argparse

from train_model import train_model

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train XGBoost model on Space Titanic data"
    )
    parser.add_argument(
        "train_data_path",
        type=str,
        help="Path to training data CSV file",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials for hyperparameter optimization",
    )
    args = parser.parse_args()

    # Train model using parsed arguments
    final_model, best_params, study = train_model(args.train_data_path, args.n_trials)

    print("\nTraining complete!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {study.best_value:.4f}")
