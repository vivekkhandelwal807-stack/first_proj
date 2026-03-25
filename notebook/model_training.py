# 2_MODEL_TRAINING.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Ignore warnings
warnings.filterwarnings('ignore')


def evaluate_model(true, predicted):
    """Function to calculate evaluation metrics"""
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


def main():
    # 1. Import Data
    # Ensure the path 'data/raw.csv' exists relative to your script
    try:
        df = pd.read_csv('data/raw.csv')
    except FileNotFoundError:
        print("Error: 'data/raw.csv' not found. Please ensure the data file is in the correct directory.")
        return

    # 2. Preparing X and Y variables
    X = df.drop(columns=['math_score'], axis=1)
    y = df['math_score']

    # 3. Create Column Transformer
    num_features = X.select_dtypes(exclude="object").columns
    cat_features = X.select_dtypes(include="object").columns

    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features),
        ]
    )

    # Transform X
    X = preprocessor.fit_transform(X)

    # 4. Separate dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Training and Evaluation
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "XGBRegressor": XGBRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor()
    }

    model_list = []
    r2_list = []

    print("\n--- Model Performance Comparison ---\n")
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train model

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate metrics
        model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
        model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

        print(name)
        model_list.append(name)

        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))
        print('----------------------------------')
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
        r2_list.append(model_test_r2)
        print('=' * 35 + '\n')

    # 6. Display Results Summary
    results_df = pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2_Score']).sort_values(
        by=["R2_Score"], ascending=False)
    print("Results Summary (Sorted by R2 Score):")
    print(results_df)

    # 7. Final Model (Linear Regression Example)
    print("\n--- Final Model: Linear Regression ---")
    lin_model = LinearRegression(fit_intercept=True)
    lin_model = lin_model.fit(X_train, y_train)
    y_pred = lin_model.predict(X_test)
    score = r2_score(y_test, y_pred) * 100
    print(" Accuracy of the model is %.2f" % score)


if __name__ == "__main__":
    main()