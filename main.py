import os

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings

warnings.filterwarnings('ignore')


import argparse
import sys

from train_model import inference, train_model

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train or run inference with XGBoost model on Space Titanic data"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to data CSV file (training or inference)",
    )
    parser.add_argument(
        "train_flag",
        type=str,
        choices=["True", "False"],
        help="True for training, False for inference"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials for hyperparameter optimization (training only)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.pkl",
        help="Path to save/load model (save when training, load when inferring)",
    )
    args = parser.parse_args()
    
    # Convert string flag to boolean
    train_mode = args.train_flag.lower() == "true"
    
    try:
        if train_mode:
            # Train model
            final_model, best_params, study = train_model(
                args.data_path, 
                train_mode, 
                args.n_trials,
                args.model_path
            )
            print("\nTraining complete!")
            print(f"Best parameters: {best_params}")
            print(f"Best score: {study.best_value:.4f}")
            print(f"Model saved to: {args.model_path}")
        else:
            # Run inference
            predictions = inference(args.data_path, args.model_path)
            print("\nInference complete!")
            print(f"Predictions shape: {predictions.shape}")
            # Save predictions to CSV
            output_path = "submissions/predictions.csv"
            predictions.to_csv(output_path, index=False)
            print(f"Predictions saved to: {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)