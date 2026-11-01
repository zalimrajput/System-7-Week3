# Module B: SVM, Naive Bayes, K-Means, Hierarchical Clustering & DBSCAN

This module covers supervised and unsupervised machine learning algorithms using Python and Scikit-learn.

## 1. Support Vector Machine (SVM)

### What is SVM?

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression.

It tries to find the best decision boundary (hyperplane) that separates different classes.

**Important concepts:**
- **Hyperplane:** Decision boundary that separates classes.
- **Support Vectors:** Data points closest to the decision boundary.
- **Kernel:** Helps SVM handle non-linear data.
- **C:** Controls the penalty for classification errors.
- **Gamma:** Controls the influence of individual training points for RBF and other applicable kernels.

### SVM for Classification (SVC)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1, gamma="scale"))
])

# 4. Train model
model.fit(X_train, y_train)

# 5. Predictions
predictions = model.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
```

### SVM for Regression (SVR)

SVR predicts continuous numerical values, such as house prices.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=10, epsilon=0.1))
])

# 4. Train model
model.fit(X_train, y_train)

# 5. Predictions
predictions = model.predict(X_test)

# 6. Evaluation
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
```

---

## 2. Naive Bayes

### What is Naive Bayes?

Naive Bayes is a supervised machine learning algorithm based on Bayes' theorem.

It calculates the probability of a class based on the input features.

It is called "Naive" because it assumes that features are conditionally independent given the class.

**Types of Naive Bayes:**

| Algorithm | Usage |
|---|---|
| GaussianNB | Continuous numerical data |
| MultinomialNB | Word counts and text classification |
| BernoulliNB | Binary features (0 and 1) |

### Gaussian Naive Bayes Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create model
model = GaussianNB()

# 4. Train model
model.fit(X_train, y_train)

# 5. Predictions
predictions = model.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
```

### Multinomial Naive Bayes for Text Classification

Example: Classifying SMS messages as spam or ham.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Sample messages
messages = [
    "Win a free prize now",
    "Congratulations you won money",
    "Hello how are you",
    "Let's meet tomorrow",
    "Claim your free reward",
    "Are you coming to class?"
]

# 2. Labels (1 = Spam, 0 = Ham)
labels = [1, 1, 0, 0, 1, 0]

# 3. Create pipeline
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# 4. Train model
model.fit(messages, labels)

# 5. Predict new messages
new_messages = [
    "Win free money",
    "Hello see you tomorrow"
]

predictions = model.predict(new_messages)

# 6. Display predictions
for message, prediction in zip(new_messages, predictions):
    result = "Spam" if prediction == 1 else "Ham"
    print(message, "->", result)
```

---

## 3. K-Means Clustering

### What is K-Means?

K-Means is an unsupervised machine learning algorithm that groups similar data points into clusters.

It works by assigning data points to the nearest cluster centroid and updating the centroids until the clusters stabilize.

**Important concepts:**
- **K:** Number of clusters.
- **Centroid:** Center point of a cluster.
- **Inertia:** Sum of squared distances between points and their assigned centroids.
- **Elbow Method:** Helps select the number of clusters.

### K-Means Example

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Generate sample data
X, y = make_blobs(
    n_samples=300,
    centers=4,
    random_state=42
)

# 2. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Create K-Means model
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# 4. Train model
model.fit(X_scaled)

# 5. Get cluster labels
labels = model.labels_

# 6. Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    s=40
)

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
```

### Elbow Method

The Elbow Method helps determine a suitable number of clusters.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertia_values = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    inertia_values.append(model.inertia_)

plt.plot(range(1, 11), inertia_values, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
```

---

## 4. Hierarchical Clustering

### What is Hierarchical Clustering?

Hierarchical Clustering is an unsupervised learning algorithm that creates a hierarchy of clusters.

It can be performed in two ways:

1. **Agglomerative:** Starts with individual data points and gradually merges them into larger clusters.
2. **Divisive:** Starts with one large cluster and repeatedly divides it into smaller clusters.

**Linkage methods:**

| Method | Description |
|---|---|
| Single | Uses the minimum distance between clusters. |
| Complete | Uses the maximum distance between clusters. |
| Average | Uses the average distance between clusters. |
| Ward | Merges clusters to minimize the increase in within-cluster variance. |

### Agglomerative Clustering Example

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# 1. Generate sample data
X, y = make_blobs(
    n_samples=200,
    centers=3,
    random_state=42
)

# 2. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Create model
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

# 4. Perform clustering
labels = model.fit_predict(X_scaled)

# 5. Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    s=40
)

plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

### Dendrogram Example

A dendrogram shows how clusters are merged at different distances.

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Generate linkage matrix
linkage_matrix = linkage(
    X_scaled,
    method="ward"
)

# 2. Plot dendrogram
plt.figure(figsize=(12, 6))

dendrogram(linkage_matrix)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
```

---

## 5. DBSCAN Clustering

### What is DBSCAN?

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.

It groups data points based on their density and identifies points that do not belong to any cluster as noise.

Unlike K-Means, DBSCAN does not require the number of clusters in advance.

**Important concepts:**

- **eps:** Maximum distance between two points for them to be considered neighbors.
- **min_samples:** Minimum number of points required in an eps-neighborhood for a core point.
- **Core Point:** A point with at least min_samples points in its neighborhood, including itself.
- **Border Point:** A point that is not a core point but lies within the neighborhood of a core point.
- **Noise Point:** A point that is not assigned to any cluster.

### DBSCAN Example

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. Generate sample data
X, y = make_moons(
    n_samples=300,
    noise=0.08,
    random_state=42
)

# 2. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Create DBSCAN model
model = DBSCAN(
    eps=0.3,
    min_samples=5
)

# 4. Perform clustering
labels = model.fit_predict(X_scaled)

# 5. Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    s=40
)

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 6. Count clusters and noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("Number of Clusters:", n_clusters)
print("Number of Noise Points:", n_noise)
```

**Note:** In DBSCAN, label `-1` represents noise points or outliers.

---

## 6. Algorithm Comparison

| Algorithm | Learning Type | Main Purpose | Requires Number of Clusters? |
|---|---|---|---|
| SVM | Supervised | Classification and regression | No |
| Naive Bayes | Supervised | Classification | No |
| K-Means | Unsupervised | Clustering | Yes |
| Hierarchical | Unsupervised | Hierarchical clustering | Usually specified when cutting the hierarchy |
| DBSCAN | Unsupervised | Density-based clustering | No |

## 8. Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, classification_report
```
