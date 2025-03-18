# Space Titanic XGBoost Classifier

This script executes XGBoost model for a Kaggle competition that ultimately classifies the survivors of the Titanic shipwreck except that it has been modified for a space version of the disaster. https://www.kaggle.com/competitions/spaceship-titanic/overview

The feat_eng.py file will clean up and perform basic feature engineering. Whereas the model.py file initiate the XGBoost model. Finally, the train_model.py will consume the training data and fit the model

The output of the train_model.py includes a trained model that can be used to run inference on unseen data. The required libraries are: numpy pandas sklearn xgboost optuna argparse

Usage: The main.py file takes one mandatory argument that is the path to the training file, and an optional argument that is the number of trials to run, by default the program will run 20 trials

python main.py "C:\Users\Leo\Documents\kaggle_comp\space_titanic\data\train.csv" --n_trials 15
