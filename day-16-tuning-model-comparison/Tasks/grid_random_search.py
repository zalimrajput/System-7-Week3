
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from scipy.stats import loguniform

# 1. Load dataset
DATA_PATH = Path(__file__).parent / "House Price Prediction Dataset.csv"
df = pd.read_csv(DATA_PATH)


# 2. Drop Id column
df = df.drop("Id", axis=1)

# 3. Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# 4. Identify columns
numeric_cols = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt"
]

categorical_cols = [
    "Location",
    "Condition",
    "Garage"
]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        ), categorical_cols)
    ]
)

# 7. Create Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", SVR())
])

# 8. Define GridSearchCV parameters
param_grid = {
    "model__kernel": ["linear", "rbf"],
    "model__C": [0.1, 1, 10, 100],
    "model__epsilon": [0.1, 0.5, 1]
}

# 9. GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 10. Define RandomizedSearchCV parameters
param_dist = {
    "model__kernel": ["linear", "rbf"],
    "model__C": loguniform(0.1, 100),
    "model__epsilon": loguniform(0.01, 2)
}

# 11. RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# 12. Compare results
print("\n========== GRID SEARCH RESULTS ==========")

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

print("\n======= RANDOMIZED SEARCH RESULTS =======")

print("Best Parameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

# 13. Compare both methods
comparison = pd.DataFrame({
    "Method": ["GridSearchCV", "RandomizedSearchCV"],
    "Best CV Score": [
        grid_search.best_score_,
        random_search.best_score_
    ]
})

print("\n========== FINAL COMPARISON ==========")
print(comparison)

# 14. Test evaluation using best models
grid_test_score = grid_search.best_estimator_.score(
    X_test, y_test
)

random_test_score = random_search.best_estimator_.score(
    X_test, y_test
)

print("\nGridSearch Test R2:", grid_test_score)
print("RandomizedSearch Test R2:", random_test_score)
