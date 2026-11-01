
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# 1. Load dataset
DATA_PATH = Path(__file__).parent / "House Price Prediction Dataset.csv"
df = pd.read_csv(DATA_PATH)


# 2. Remove Id because it is only an identifier
df = df.drop("Id", axis=1)

# 3. Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# 4. Identify numeric and categorical columns
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

# 6. Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 7. Fit and transform training data
X_train_transformed = preprocessor.fit_transform(X_train)

# 8. Transform testing data
X_test_transformed = preprocessor.transform(X_test)

# 9. Display results
print("Original Training Shape:", X_train.shape)
print("Transformed Training Shape:", X_train_transformed.shape)

print("Original Testing Shape:", X_test.shape)
print("Transformed Testing Shape:", X_test_transformed.shape)

print("\nTransformed Feature Names:")
print(preprocessor.get_feature_names_out())
