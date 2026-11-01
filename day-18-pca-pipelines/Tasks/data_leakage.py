
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. LOAD DATASET
# ==========================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 3. INCORRECT APPROACH: DATA LEAKAGE
# SCALING BEFORE TRAIN-TEST SPLIT
# ==========================================

scaler_leak = StandardScaler()

# WRONG: Scaler learns from the entire dataset
X_scaled = scaler_leak.fit_transform(X)

# Split already-scaled data
X_train_leak = X_scaled[X_train.index]
X_test_leak = X_scaled[X_test.index]

# Train classifier
model_leak = LogisticRegression(max_iter=5000)

model_leak.fit(X_train_leak, y_train)

y_pred_leak = model_leak.predict(X_test_leak)

leak_accuracy = accuracy_score(y_test, y_pred_leak)

print("\n========== DATA LEAKAGE APPROACH ==========")
print("Accuracy:", leak_accuracy)

# ==========================================
# 4. CORRECT APPROACH: BUILD PIPELINE
# SCALER + PCA + CLASSIFIER
# ==========================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),

    ("pca", PCA(n_components=0.95)),

    ("classifier", LogisticRegression(max_iter=5000))
])

# Pipeline learns only from training data
pipeline.fit(X_train, y_train)

# Predict on untouched test data
y_pred_pipeline = pipeline.predict(X_test)

pipeline_accuracy = accuracy_score(y_test, y_pred_pipeline)

print("\n========== CORRECT PIPELINE APPROACH ==========")
print("Accuracy:", pipeline_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_pipeline))

# ==========================================
# 5. COMPARE BOTH APPROACHES
# ==========================================

comparison = pd.DataFrame({
    "Approach": [
        "Scaling Before Split (Leakage)",
        "Pipeline (Correct)"
    ],

    "Accuracy": [
        leak_accuracy,
        pipeline_accuracy
    ]
})

print("\n========== ACCURACY COMPARISON ==========")
print(comparison)

# ==========================================
# 6. VISUALIZE COMPARISON
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Approach"],
    comparison["Accuracy"]
)

plt.ylabel("Accuracy")
plt.xlabel("Approach")
plt.title("Data Leakage vs Correct Pipeline")

plt.ylim(0, 1.05)
plt.xticks(rotation=10)

plt.grid(axis="y")
plt.tight_layout()
plt.show()
