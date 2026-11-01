# Module B: Advanced Machine Learning Concepts

This module covers important machine learning techniques used to improve data quality, reduce dimensionality, handle imbalanced datasets, prevent data leakage, and build reliable machine learning workflows.

## 1. Principal Component Analysis (PCA)

### What is PCA?

Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique.

It reduces the number of features in a dataset while preserving as much important information (variance) as possible.

For example:
- Original dataset: 10 features.
- After PCA: 2 principal components.
- The model can work with fewer dimensions.

### Why Use PCA?

- Reduce the number of features.
- Remove redundant information.
- Reduce computational complexity.
- Help visualize high-dimensional data.
- Reduce noise in some datasets.

### Code Example: PCA

```python
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA and reduce to 2 components
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# 4. Display explained variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance:")
print(pca.explained_variance_ratio_.sum())

# 5. Visualize PCA results
plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Iris Dataset")
plt.colorbar(label="Class")
plt.show()
```

### Important PCA Parameters

| Parameter | Description |
|---|---|
| `n_components=2` | Reduces data to two principal components. |
| `explained_variance_ratio_` | Shows the proportion of variance explained by each component. |
| `fit_transform()` | Learns principal components and transforms the data. |
| `transform()` | Transforms new data using previously learned components. |

**Important:** Standardize features before applying PCA when they have different scales.

---

## 2. Feature Engineering

### What is Feature Engineering?

Feature engineering is the process of creating, transforming, or selecting features to improve machine learning model performance.

A model's performance often depends on the quality of its input features.

### Common Feature Engineering Techniques

1. Creating new features.
2. Encoding categorical variables.
3. Extracting date and time information.
4. Handling missing values.
5. Scaling numerical features.
6. Removing irrelevant features.

### Code Example: Feature Engineering on House Price Dataset

Dataset columns:

`Id, Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage, Price`

```python
import pandas as pd

# 1. Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

# 2. Remove unnecessary ID column
df = df.drop("Id", axis=1)

# 3. Create new features
df["HouseAge"] = 2026 - df["YearBuilt"]

df["TotalRooms"] = (
    df["Bedrooms"] + df["Bathrooms"]
)

df["AreaPerRoom"] = (
    df["Area"] / df["TotalRooms"].replace(0, 1)
)

# 4. Display new features
print(df[
    ["Area", "Bedrooms", "Bathrooms",
     "HouseAge", "TotalRooms", "AreaPerRoom"]
].head())

# 5. Check dataset information
print("\nDataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns)
```

### Explanation

| New Feature | Meaning |
|---|---|
| `HouseAge` | Current year minus construction year. |
| `TotalRooms` | Bedrooms plus bathrooms. |
| `AreaPerRoom` | House area divided by total rooms. |

Feature engineering helps the model discover useful relationships that may not be obvious from the original features.

---

## 3. Handling Imbalanced Data

### What is Imbalanced Data?

An imbalanced dataset occurs when one class has significantly more samples than another class.

Example:

| Class | Number of Samples |
|---|---:|
| Not Fraud (0) | 950 |
| Fraud (1) | 50 |

Here, 95% of the data belongs to class 0 and only 5% belongs to class 1.

A model may predict every transaction as not fraud and still achieve 95% accuracy, while completely failing to detect fraud.

### Common Techniques

- Random Oversampling: increases minority-class samples by duplicating existing examples.
- Random Undersampling: reduces majority-class samples.
- SMOTE: generates synthetic minority-class examples.
- Class Weights: assigns greater importance to minority-class errors.

### Code Example: Handling Imbalanced Data with SMOTE

Install imbalanced-learn:

```bash
pip install imbalanced-learn
```

```python
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

# 1. Generate an imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],
    random_state=42
)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Check original class distribution
print("Before SMOTE:")
print(pd.Series(y_train).value_counts())

# 4. Apply SMOTE only to training data
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train, y_train
)

# 5. Check new class distribution
print("\nAfter SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# 6. Train model
model = RandomForestClassifier(random_state=42)

model.fit(X_train_resampled, y_train_resampled)

# 7. Predict on original test data
y_pred = model.predict(X_test)

# 8. Evaluate model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

**Important:** Apply SMOTE only to the training data. Never apply it to the test data because this can cause data leakage.

---

## 4. Data Leakage

### What is Data Leakage?

Data leakage occurs when information from outside the training data is unintentionally used during model training.

It can make a model appear more accurate during evaluation than it will be on genuinely unseen data.

### Common Types of Data Leakage

**1. Scaling Before Train-Test Split**

Incorrect:

```python
from sklearn.preprocessing import StandardScaler

# Incorrect: learns statistics from the entire dataset
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Split after scaling
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

Why is this incorrect?

The scaler learns the mean and standard deviation using both training and testing data.

**Correct:**

```python
# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit scaler only on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using training statistics
X_test_scaled = scaler.transform(X_test)
```

**2. Applying SMOTE Before Train-Test Split**

Incorrect:

```python
# Incorrect: creates synthetic samples before splitting
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)
```

Why is this incorrect?

Synthetic samples can be generated using information from the entire dataset. Related samples may then appear in both training and testing sets.

**Correct:** Split first and apply SMOTE only to the training set.

**3. Using Target Information as an Input Feature**

If the target variable or information derived from it is included in the input features, the model can learn information that would not be available at prediction time.

### How to Prevent Data Leakage

- Split the data before preprocessing.
- Fit scalers and encoders only on training data.
- Apply SMOTE only to training data.
- Remove target-derived or future information from input features.
- Use Pipeline and cross-validation for safer workflows.

---

## 5. Machine Learning Pipelines

### What is a Pipeline?

A Pipeline combines multiple machine learning steps into one workflow.

For example:

1. Handle missing values.
2. Scale numerical features.
3. Encode categorical features.
4. Train a machine learning model.

Instead of performing each operation separately, Pipeline executes them in the correct order.

### Why Use Pipeline?

- Keeps preprocessing and model training organized.
- Reduces repetitive code.
- Helps prevent data leakage.
- Makes cross-validation easier.
- Ensures consistent preprocessing during prediction.

### Code Example: Pipeline with House Price Dataset

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

# 2. Remove ID column
df = df.drop("Id", axis=1)

# 3. Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# 4. Define feature columns
numeric_cols = [
    "Area", "Bedrooms", "Bathrooms",
    "Floors", "YearBuilt"
]

categorical_cols = [
    "Location", "Condition", "Garage"
]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6. Numerical preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 7. Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# 9. Create complete ML pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# 10. Train model
model_pipeline.fit(X_train, y_train)

# 11. Make predictions
y_pred = model_pipeline.predict(X_test)

# 12. Evaluate model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

### Understanding Pipeline

```python
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])
```

| Component | Purpose |
|---|---|
| `Pipeline` | Combines preprocessing and model training. |
| `ColumnTransformer` | Applies different preprocessing to different columns. |
| `SimpleImputer` | Handles missing values. |
| `StandardScaler` | Scales numerical features. |
| `OneHotEncoder` | Converts categorical features into numerical features. |
| `RandomForestRegressor` | Predicts house prices. |

---

# 6. Practical Task: Customer Segmentation Using K-Means and PCA

### Task Description

Build a customer segmentation model using K-Means clustering.

Customer segmentation means dividing customers into groups based on similar characteristics, such as annual income and spending score.

### Dataset

For this task, use the Mall Customers dataset, which contains customer information.

Dataset features:

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier. |
| Gender | Customer gender. |
| Age | Customer age. |
| Annual Income (k$) | Annual income in thousands of dollars. |
| Spending Score (1-100) | Spending score assigned to each customer. |

Dataset: [Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

### Task Requirements

1. Load and explore the dataset.
2. Check missing values and duplicate records.
3. Remove the `CustomerID` column because it is an identifier.
4. Select `Annual Income (k$)` and `Spending Score (1-100)` for clustering.
5. Apply StandardScaler.
6. Use the Elbow Method to find a suitable number of clusters.
7. Train a K-Means clustering model.
8. Assign cluster labels to customers.
9. Visualize customer segments.
10. Apply PCA and visualize the clusters in two dimensions.

### Complete Code Example

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")

# 2. Explore dataset
print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# 3. Remove unnecessary columns
df = df.drop("CustomerID", axis=1)

# 4. Select features for clustering
X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

# 5. Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 6. Find optimal K using Elbow Method
inertia_values = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(model.inertia_)

# 7. Plot Elbow Method
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# 8. Train K-Means model
# Select K after inspecting the elbow plot
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# 9. Add cluster labels to dataset
df["Cluster"] = clusters

print("\nCustomer Segments:")
print(df.head())

print("\nCustomers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# 10. Visualize customer segments
plt.figure(figsize=(8, 5))

plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=clusters
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation Using K-Means")

plt.show()

# 11. Apply PCA for dimensionality reduction
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# 12. Visualize clusters using PCA
plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation Using PCA")

plt.show()

# 13. Display explained variance
print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal Explained Variance:",
    pca.explained_variance_ratio_.sum()
)
```

### Understanding Customer Segments

After clustering, inspect the average income and spending score of each cluster:

```python
cluster_summary = df.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print(cluster_summary)
```

Example customer segment interpretations:

| Segment | Income | Spending | Interpretation |
|---|---|---|---|
| 1 | High | High | Target customers |
| 2 | High | Low | Potential customers |
| 3 | Low | High | Frequent spenders |
| 4 | Low | Low | Low spending customers |
| 5 | Medium | Medium | Average customers |

*These are example interpretations. Actual segment characteristics depend on the cluster summary produced by your model.*


## Module B: Summary

| Topic | Main Concept | Scikit-learn / Library |
|---|---|---|
| PCA | Dimensionality reduction | `PCA` |
| Feature Engineering | Creating useful features | Pandas |
| Imbalanced Data | Handling unequal class distributions | `SMOTE` |
| Data Leakage | Preventing test-data information from entering training | `train_test_split` |
| Pipelines | Combining preprocessing and model training | `Pipeline` |
| Customer Segmentation | Grouping similar customers | `KMeans` |

### Installation

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
```
