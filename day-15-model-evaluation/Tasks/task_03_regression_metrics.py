
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load the dataset
DATA_PATH = Path(__file__).parent / "House Price Prediction Dataset.csv"
df = pd.read_csv(DATA_PATH)

# 2. Clean column names
df.columns = df.columns.str.strip()

# Remove unnecessary ID column
df = df.drop(columns=["Id"], errors="ignore")

# Remove rows with missing target prices
df = df.dropna(subset=["Price"])

# Convert target price to numeric
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 3. Separate features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# 4. Identify numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 5. Preprocess missing values and categorical features
numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_columns),
        ("categorical", categorical_transformer, categorical_columns)
    ]
)

# 6. Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# 7. Create and train the regression model
model = make_pipeline(
    preprocessor,
    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
)

model.fit(X_train, y_train)

# 8. Predict prices for unseen test houses
y_pred = model.predict(X_test)

# Convert actual and predicted prices to NumPy arrays
y_actual = y_test.to_numpy(dtype=float)
y_predicted = np.asarray(y_pred, dtype=float)

# 9. Calculate metrics manually using NumPy
errors = y_actual - y_predicted

mae_numpy = np.mean(np.abs(errors))
mse_numpy = np.mean(errors ** 2)
rmse_numpy = np.sqrt(mse_numpy)

ss_res = np.sum(errors ** 2)
ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

r2_numpy = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

# 10. Verify with Scikit-learn
mae_sklearn = mean_absolute_error(y_actual, y_predicted)
mse_sklearn = mean_squared_error(y_actual, y_predicted)
rmse_sklearn = np.sqrt(mse_sklearn)
r2_sklearn = r2_score(y_actual, y_predicted)

# 11. Display results
print("\n" + "=" * 55)
print("HOUSE PRICE MODEL EVALUATION")
print("=" * 55)

print("\nManual calculation using NumPy:")
print(f"MAE  : {mae_numpy:.2f}")
print(f"MSE  : {mse_numpy:.2f}")
print(f"RMSE : {rmse_numpy:.2f}")
print(f"R²   : {r2_numpy:.4f}")

print("\nVerification using Scikit-learn:")
print(f"MAE  : {mae_sklearn:.2f}")
print(f"MSE  : {mse_sklearn:.2f}")
print(f"RMSE : {rmse_sklearn:.2f}")
print(f"R²   : {r2_sklearn:.4f}")

print("\nVerification checks:")
print("MAE matches :", np.isclose(mae_numpy, mae_sklearn))
print("MSE matches :", np.isclose(mse_numpy, mse_sklearn))
print("RMSE matches:", np.isclose(rmse_numpy, rmse_sklearn))
print("R² matches  :", np.isclose(r2_numpy, r2_sklearn))

# 12. Compare actual and predicted prices
comparison = pd.DataFrame({
    "Actual Price": y_actual,
    "Predicted Price": np.round(y_predicted, 2),
    "Absolute Error": np.round(np.abs(errors), 2)
})

print("\nActual vs Predicted Prices (first 10 test houses):")
print(comparison.head(10).to_string(index=False))
