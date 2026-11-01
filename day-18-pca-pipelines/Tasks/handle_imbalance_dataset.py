import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    recall_score,
    precision_score,
    classification_report
)

# ==========================================
# 1. LOAD DATASET
# ==========================================


DATA_PATH = Path(__file__).parent / "creditcard.csv"
df = pd.read_csv(DATA_PATH)


print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# 2. CHECK CLASS DISTRIBUTION
# ==========================================

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 5. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

# Fit scaler only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# ==========================================
# MODEL 1: WITHOUT CLASS WEIGHT
# ==========================================

model_before = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train model
model_before.fit(X_train_scaled, y_train)

# Make predictions
y_pred_before = model_before.predict(X_test_scaled)

# Calculate recall
recall_before = recall_score(
    y_test,
    y_pred_before,
    pos_label=1
)

precision_before = precision_score(
    y_test,
    y_pred_before,
    zero_division=0
)

# ==========================================
# MODEL 2: WITH CLASS WEIGHT
# ==========================================

model_after = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train model
model_after.fit(X_train_scaled, y_train)

# Make predictions
y_pred_after = model_after.predict(X_test_scaled)

# Calculate recall
recall_after = recall_score(
    y_test,
    y_pred_after,
    pos_label=1
)

precision_after = precision_score(
    y_test,
    y_pred_after,
    zero_division=0
)

# ==========================================
# 6. COMPARE MODEL PERFORMANCE
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Without Class Weight",
        "With Class Weight"
    ],
    "Recall": [
        recall_before,
        recall_after
    ],
    "Precision": [
        precision_before,
        precision_after
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

# ==========================================
# 7. CLASSIFICATION REPORTS
# ==========================================

print("\n========== WITHOUT CLASS WEIGHT ==========")
print(classification_report(y_test, y_pred_before))

print("\n========== WITH CLASS WEIGHT ==========")
print(classification_report(y_test, y_pred_after))

# ==========================================
# 8. VISUALIZE RECALL COMPARISON
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Model"],
    comparison["Recall"]
)

plt.xlabel("Model")
plt.ylabel("Recall Score")
plt.title("Recall Before and After Applying Class Weight")

plt.ylim(0, 1.1)
plt.grid(axis="y")

plt.show()