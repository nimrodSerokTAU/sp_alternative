import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from typing import Literal

class Regressor:
    def __init__(self, features_file: str, test_size: float) -> None:
        self.features_file = features_file
        self.test_size = test_size
        # self.num_estimators = n_estimators
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X, self.y, self.y_pred = None, None, None
        self.prediction = None

        df = pd.read_csv(self.features_file)
        # Check for missing values
        print("Missing values in each column:\n", df.isnull().sum())

        # Handle missing values (if any)
        # Example: Filling missing values with the mean (for numerical columns)
        # df['orig_tree_ll'] = df['orig_tree_ll'].fillna(df['orig_tree_ll'].mean())

        # target score
        self.y = df["score"]
        # features
        # self.X = df.drop(columns=['score', 'code'])
        self.X = df.drop(columns=['score', 'code', 'taxa_num', 'pypythia_msa_difficulty','msa_path','entropy_min','entropy_pct_25'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)

    def random_forest(self, n_estimators: int = 100) -> float:
        # Create and fit the Random Forest Regressor
        self.regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.regressor.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.regressor.predict(self.X_test)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    def gradient_boost(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3) -> float:
        # Create and fit the Gradient Boosting Regressor
        self.regressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        self.regressor.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.regressor.predict(self.X_test)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    def support_vector(self, kernel: str = 'rbf', c_param: int = 100, gamma: float = 0.1) -> float:
        # Create and fit the Support Vector Regressor
        self.regressor = SVR(kernel=kernel, C=c_param, gamma=gamma)
        self.regressor.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.regressor.predict(self.X_test)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    def k_nearest_neighbors(self, n_neighbors: int = 5) -> float:
        # Create and fit the K-Nearest Neighbors Regressor
        self.regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.regressor.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.regressor.predict(self.X_test)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    def plot_results(self, model_name: Literal["svr", "rf", "knn-r", "gbr"]) -> None:
        # Plot results for many features
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, color='blue', edgecolor='k', alpha=0.7)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', linestyle='--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        if model_name == "svr":
            title = "Support Vector Regression"
        elif model_name == "rf":
            title = "Random Forest Regression"
        elif model_name == "knn-r":
            title = "K-Nearest Neighbors Regression"
        elif model_name == "gbr":
            title = "Gradient Boosting Regression"
        plt.title(f'{title}: Predicted vs. True Values')
        plt.grid(True)
        plt.show()

    # def get_features_importances(self, model_name: str) -> None:
        if model_name == "rf" or model_name == "gbr":
            importances = self.regressor.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.title(f'{title} Feature Importances')
            plt.bar(range(self.X.shape[1]), importances[indices], align='center')
            plt.xticks(range(self.X.shape[1]), self.X.columns[indices], rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.show()