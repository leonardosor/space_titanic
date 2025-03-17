import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def feat_eng(df, train=False):
    """
    preprocess the data needed to train an XGBoost model
    :param data: dataframe
    :param train_flag: whether to preprocess the data for training or testing
    :return: preprocessed data
    """

    cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    df["VIP"] = df["VIP"].astype(bool)
    df[["FN", "LN"]] = df["Name"].str.split(" ", expand=True)
    df["Family"] = (
        df["LN"].duplicated(keep=False).map({True: "family", False: "not family"})
    )
    df["sp_cols"] = (df[cols] != 0).sum(axis=1)
    df[["deck", "room", "side"]] = df["Cabin"].str.split("/", expand=True)
    df.loc[(df["CryoSleep"].isna()) & (df[cols].isna().any(axis=1)), "CryoSleep"] = 0
    df.loc[df["CryoSleep"] == True, cols] = df.loc[
        df["CryoSleep"] == True, cols
    ].fillna(0)
    # Define a nested function to calculate the rate
    try:
        vip = (df["VIP"] == True).sum()
        transp = (df["sp_cols"] == True).sum()

        df["Rate"] = np.where(
            (df.get("VIP", 0) == True) & (df.get("Family", 0) == "family"),
            1 - (vip / transp),
            vip / transp,
        )
    except:
        pass

    numeric_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    categorical_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Family",
        "deck",
        "side",
    ]

    num_cols = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "sp_cols",
        "room",
        "Rate",
    ]

    # Create a preprocessing pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    if train:
        X = df.drop(columns=["PassengerId", "Name", "FN", "LN", "Cabin", "Transported"])
        y = df["Transported"].astype("Int32")

        # Split the data into training and testing sets
        n = len(X)
        cutoff = int(0.8 * n)
        train_idx = np.arange(cutoff)
        test_idx = np.arange(cutoff, n)

        X_train, X_test, y_train, y_test = (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
        )
        # y_train = y_train.to_numpy()
        # y_test = y_test.to_numpy()
        # Fit and transform the training data
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.fit_transform(X_test)

        categorical_feature_names = preprocessor.transformers_[1][1].get_feature_names(
            input_features=categorical_cols
        )

        # Use the original numerical column names (no changes here)
        numeric_feature_names = num_cols

        # Combine the feature names into one list
        all_feature_names = np.concatenate(
            [numeric_feature_names, categorical_feature_names]
        )

        # Convert the transformed data to a DataFrame with the new column names
        # X_transformed = pd.DataFrame(X_train_preprocessed, columns=all_feature_names)

        return X_train_preprocessed, X_test_preprocessed, y_train, y_test

    else:
        X = df.drop(
            columns=[
                "PassengerId",
                "Name",
                "FN",
                "LN",
                "Cabin",
            ]
        )
        X_blind = preprocessor.fit_transform(X)

        return X_blind
