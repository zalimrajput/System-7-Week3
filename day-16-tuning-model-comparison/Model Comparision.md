# Day 2 – Module A: Machine Learning

## Topics Covered

1. Overfitting and Underfitting
2. Feature Scaling
   - StandardScaler
   - MinMaxScaler
3. Categorical Encoding
   - LabelEncoder
   - OneHotEncoder
4. Hyperparameter Tuning
   - GridSearchCV
   - RandomizedSearchCV
5. Model Comparison

---

## 1. Overfitting and Underfitting

### Overfitting

Overfitting occurs when a model learns the training data too well, including noise and unnecessary patterns. It performs very well on training data but poorly on unseen testing data.

### Underfitting

Underfitting occurs when a model is too simple to learn the underlying patterns in the dataset. It performs poorly on both training and testing data.

### Python Example

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Different tree depths
depths = [1, 2, 3, 5, 10, 15, None]

for depth in depths:

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Max Depth: {depth}")
    print(f"Training R2: {train_score:.4f}")
    print(f"Testing R2: {test_score:.4f}")
    print("-" * 40)
```

### Understanding the Results

| Condition | Training Score | Testing Score |
|---|---|---|
| Underfitting | Low | Low |
| Good Fit | Good | Good |
| Overfitting | Very High | Low |

**Important:** A good model should generalize well to unseen data, not just memorize training data.

---

## 2. Feature Scaling

Feature scaling brings numerical features to a similar scale so that features with larger numerical ranges do not dominate algorithms that are sensitive to feature magnitude.

### A. StandardScaler

StandardScaler transforms data so that it has approximately:

- Mean = 0
- Standard Deviation = 1

Formula:

\[
z = \frac{x-\mu}{\sigma}
\]

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Before Scaling:")
print(X_train.head())

print("After Scaling:")
print(X_train_scaled[:5])
```

### B. MinMaxScaler

MinMaxScaler scales values into a specified range, usually between 0 and 1.

Formula:

\[
X_{scaled} = \frac{X-X_{min}}{X_{max}-X_{min}}
\]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled[:5])
```

### StandardScaler vs MinMaxScaler

| StandardScaler | MinMaxScaler |
|---|---|
| Mean approximately 0 | Minimum 0 |
| Standard deviation approximately 1 | Maximum 1 |
| Does not restrict values to a fixed range | Usually scales between 0 and 1 |
| Useful for many linear and distance-based models | Useful when a bounded range is desired |

**Important:** Always fit the scaler on training data only, then transform testing data to avoid data leakage.

---

## 3. Categorical Encoding

Machine learning algorithms generally require numerical input. Encoding converts categorical values into numerical representations.

### A. LabelEncoder

LabelEncoder assigns numerical labels to categories. It is mainly designed for encoding target labels (`y`).

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data = ["Good", "Average", "Poor", "Good", "Poor"]

encoded_data = encoder.fit_transform(data)

print("Original:", data)
print("Encoded:", encoded_data)
```

### B. OneHotEncoder

OneHotEncoder creates separate binary columns for categorical values.

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

data = pd.DataFrame({
    "Location": ["Urban", "Rural", "Suburban", "Urban"]
})

encoder = OneHotEncoder(sparse_output=False)

encoded_data = encoder.fit_transform(data[["Location"]])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(["Location"])
)

print(encoded_df)
```

### Output

| Location_Rural | Location_Suburban | Location_Urban |
|---|---|---|
| 0 | 0 | 1 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

**Note:** For older Scikit-learn versions, use `sparse=False` instead of `sparse_output=False`.

---

## 4. GridSearchCV

GridSearchCV is a hyperparameter tuning technique that tests all specified combinations of hyperparameters using cross-validation.

### Python Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
```

### Explanation

- `n_estimators`: Number of decision trees.
- `max_depth`: Maximum depth of each tree.
- `min_samples_split`: Minimum samples required to split a node.
- `cv=5`: Performs five-fold cross-validation.
- `scoring="r2"`: Uses R² to evaluate model performance.
- `best_params_`: Returns the best parameter combination.

---

## 5. RandomizedSearchCV

RandomizedSearchCV tests a specified number of randomly selected hyperparameter combinations instead of testing every combination.

### Python Example

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

model = RandomForestRegressor(random_state=42)

param_dist = {
    "n_estimators": randint(50, 200),
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": randint(2, 10)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

best_model = random_search.best_estimator_
```

### GridSearchCV vs RandomizedSearchCV

| GridSearchCV | RandomizedSearchCV |
|---|---|
| Tests every specified combination | Tests randomly selected combinations |
| Can be computationally expensive | Usually faster with a limited search budget |
| Suitable for smaller parameter grids | Suitable for larger parameter spaces |
| Exhaustive within the specified grid | Does not guarantee finding the global best |

---

## 6. Model Comparison

Model comparison helps identify which algorithm performs best on unseen data.

### Models Used

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Python Example

```python
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

results_df = pd.DataFrame(results)

print(results_df.sort_values(
    by="R2 Score",
    ascending=False
))
```

### Evaluation Metrics

| Metric | Meaning | Better Result |
|---|---|---|
| MAE | Mean Absolute Error | Lower |
| MSE | Mean Squared Error | Lower |
| RMSE | Root Mean Squared Error | Lower |
| R² | Coefficient of Determination | Higher |


**Technologies and Libraries**

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Scikit-learn` · `SciPy`

