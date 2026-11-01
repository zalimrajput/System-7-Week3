
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from pathlib import Path


# 1. Load dataset
DATA_PATH = Path(__file__).parent / "House Price Prediction Dataset.csv"
df = pd.read_csv(DATA_PATH)


# 2. Remove ID column
df = df.drop("Id", axis=1)

# 3. Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# 4. Identify categorical columns
categorical_cols = ["Location", "Condition", "Garage"]

# 5. Identify numerical columns
numeric_cols = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 7. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# 8. Store scores
train_scores = []
test_scores = []
depth_values = range(1, 16)

# 9. Train models with different depths
for depth in depth_values:

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Training score
    train_score = pipeline.score(X_train, y_train)

    # Testing score
    test_score = pipeline.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

    print(
        f"Depth: {depth:2d} | "
        f"Train R2: {train_score:.4f} | "
        f"Test R2: {test_score:.4f}"
    )

# 10. Plot results
plt.figure(figsize=(10, 6))

plt.plot(depth_values, train_scores, marker="o", label="Training Score")
plt.plot(depth_values, test_scores, marker="o", label="Testing Score")

plt.xlabel("Maximum Tree Depth")
plt.ylabel("R2 Score")
plt.title("Decision Tree: Training vs Testing Score")
plt.xticks(range(1, 16))
plt.legend()
plt.grid(True)

plt.show()
