
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import precision_score

# 1. Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create Naive Bayes model
nb_model = GaussianNB()

# 4. Train Naive Bayes
nb_model.fit(X_train, y_train)

# 5. Naive Bayes predictions
nb_predictions = nb_model.predict(X_test)

# 6. Create SVM model with scaling
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# 7. Train SVM
svm_model.fit(X_train, y_train)

# 8. SVM predictions
svm_predictions = svm_model.predict(X_test)

# 9. Calculate precision
nb_precision = precision_score(
    y_test,
    nb_predictions,
    average="weighted"
)

svm_precision = precision_score(
    y_test,
    svm_predictions,
    average="weighted"
)

# 10. Compare results
comparison = pd.DataFrame({
    "Model": ["Naive Bayes", "SVM"],
    "Precision": [nb_precision, svm_precision]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison)

# 11. Visualize precision scores
plt.figure(figsize=(7, 5))

plt.bar(
    comparison["Model"],
    comparison["Precision"]
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Precision Score")
plt.title("Naive Bayes vs SVM Precision Comparison")
plt.ylim(0, 1.1)
plt.grid(axis="y")

plt.show()

# 12. Identify the better model
if nb_precision > svm_precision:
    print("Naive Bayes has higher precision.")
elif svm_precision > nb_precision:
    print("SVM has higher precision.")
else:
    print("Both models have equal precision.")
