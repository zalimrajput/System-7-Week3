
from pathlib import Path

import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. Load the house-price dataset
DATA_PATH = Path(__file__).parent / "House Price Prediction Dataset.csv"
df = pd.read_csv(DATA_PATH)

# 2. Clean column names and prepare data
df.columns = df.columns.str.strip()
df = df.drop(columns=["Id"], errors="ignore")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])

X = df.drop(columns=["Price"])
y = df["Price"]

# 3. Identify numerical and categorical columns
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 4. Preprocessing
preprocessor = ColumnTransformer([
    (
        "numeric",
        make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        ),
        numeric_columns
    ),
    (
        "categorical",
        make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        ),
        categorical_columns
    )
])

# 5. Define three regression models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}

# 6. Configure 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 7. Evaluate each model using R²
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)
print("Scoring metric: R²")
print("Higher R² is generally better.\n")

for name, model in models.items():
    pipeline = make_pipeline(preprocessor, model)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    print(name)
    print("Fold scores:", scores.round(4))
    print(f"Mean R²: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print("-" * 60)
