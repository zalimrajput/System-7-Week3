
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Create a sample classification dataset
data = {
    "study_hours": [2, 5, 1, 6, 3, 8, 4, 7, 2, 9,
                    1, 6, 3, 8, 5, 4, 7, 2, 9, 6],
    "attendance": [60, 85, 50, 90, 70, 95, 75, 92, 65, 98,
                   55, 88, 72, 96, 80, 78, 91, 62, 99, 87],
    "result": ["Fail", "Pass", "Fail", "Pass", "Fail",
               "Pass", "Pass", "Pass", "Fail", "Pass",
               "Fail", "Pass", "Fail", "Pass", "Pass",
               "Fail", "Pass", "Fail", "Pass", "Pass"]
}

df = pd.DataFrame(data)

# 2. Separate features (X) and target (y)
X = df.drop(columns=["result"])
y = df["result"]

# 3. Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Print the shapes
print("Original dataset shape:", df.shape)
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

# 5. Print class balance before splitting
print("\nOriginal class balance:")
print(y.value_counts())
print("\nOriginal class percentages:")
print((y.value_counts(normalize=True) * 100).round(2))

# 6. Print class balance in training data
print("\nTraining class balance:")
print(y_train.value_counts())
print("\nTraining class percentages:")
print((y_train.value_counts(normalize=True) * 100).round(2))

# 7. Print class balance in testing data
print("\nTesting class balance:")
print(y_test.value_counts())
print("\nTesting class percentages:")
print((y_test.value_counts(normalize=True) * 100).round(2))
