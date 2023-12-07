import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.tree import plot_tree
import time


class DiagnosisPrediction:
    """
    A class for predicting the total number of concurrent HIV/AIDS diagnoses.
+
    Methods:
    - drop_columns: Drops specified columns from the DataFrame.
    - encode_categorical: Performs One-Hot Encoding for the 'RACE/ETHNICITY' column.
    - convert_to_numeric: Converts a specific column to float format.
    - train_model: Trains a Random Forest Regressor model for prediction and performs cross-validation.
    - visualize_trees: Visualizes decision trees from the trained Random Forest Regressor.
    """

    def __init__(self):
        pass

    def drop_columns(self, df):
        """
        Drops specified columns from the DataFrame.

        Args:
        - df: Input DataFrame

        Returns:
        - Modified DataFrame after dropping specified columns.
        """
        df.drop(
            columns=[
                "NEIGHBORHOOD",
                "AGE",
                "HIV DIAGNOSES PER 100,000 POPULATION",
                "TOTAL NUMBER OF HIV DIAGNOSES",
                "PROPORTION OF CONCURRENT HIV/AIDS DIAGNOSES AMONG ALL HIV DIAGNOSES",
                "TOTAL NUMBER OF AIDS DIAGNOSES",
                "AIDS DIAGNOSES PER 100,000 POPULATION",
                "Borough",
            ],
            inplace=True,
        )
        return df

    def encode_categorical(self, df):
        """
        Performs One-Hot Encoding for the 'RACE/ETHNICITY' column.

        Args:
        - df: Input DataFrame

        Returns:
        - DataFrame after performing One-Hot Encoding for the specified column.
        """
        df_prediction_set = pd.get_dummies(df, columns=["RACE/ETHNICITY"])
        return df_prediction_set

    def convert_to_numeric(self, df):
        """
        Converts a specific column to float format.

        Args:
        - df: Input DataFrame

        Returns:
        - DataFrame with the specified column converted to float format.
        """
        df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"] = (
            pd.to_numeric(
                df["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"], errors="coerce"
            )
            .fillna(0)
            .astype(float)
        )
        return df

    def train_model(self, df_prediction_set):
        """
        Trains a Random Forest Regressor model for prediction and performs cross-validation.

        Args:
        - df_prediction_set: DataFrame with features and target variable.

        Returns:
        - Average Mean Squared Error (MSE) from cross-validation.
        """
        start_time = time.time()

        X = df_prediction_set.drop(
            columns=["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"]
        )
        y = df_prediction_set["TOTAL NUMBER OF CONCURRENT HIV/AIDS DIAGNOSES"]

        numeric_features = X.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
        average_mse = -scores.mean()

        elapsed_time = time.time() - start_time
        print(f"Time taken for training: {elapsed_time:.2f} seconds")

        model.fit(X, y)

        # Printing details about the trained model
        print("Trained Random Forest Regressor details:")
        print(model.named_steps["regressor"])

        return average_mse

    def visualize_trees(self):
        """
        Visualizes decision trees from the trained Random Forest Regressor.
        """
        X, y = make_regression(n_samples=100, n_features=4, noise=0.1)

        rf = RandomForestRegressor(n_estimators=10)
        rf.fit(X, y)

        num_trees_to_visualize = 3
        for i in range(num_trees_to_visualize):
            tree = rf.estimators_[i]
            plt.figure(figsize=(10, 5))
            plot_tree(
                tree,
                filled=True,
                feature_names=[f"Feature {i}" for i in range(X.shape[1])],
            )
            plt.title(f"Decision Tree {i+1}")
            plt.show()
