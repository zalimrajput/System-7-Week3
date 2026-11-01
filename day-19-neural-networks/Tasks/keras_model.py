
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Generate sample tabular dataset
# ==========================================

X, y = make_classification(
    n_samples=2000,
    n_features=8,
    n_informative=6,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)

# ==========================================
# 2. Split dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 3. Feature scaling
# ==========================================

scaler = StandardScaler()

# Fit only on training data
X_train = scaler.fit_transform(X_train)

# Use the same scaler for test data
X_test = scaler.transform(X_test)

# ==========================================
# 4. Build Keras Sequential model
# ==========================================

def build_model(optimizer):

    # Same random initialization for fair comparison
    tf.keras.utils.set_random_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),

        tf.keras.layers.Dense(32, activation="relu"),

        tf.keras.layers.Dense(16, activation="relu"),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================
# 5. Train model using SGD
# ==========================================

sgd_model = build_model(
    tf.keras.optimizers.SGD(learning_rate=0.01)
)

print("\nTraining with SGD...")

sgd_history = sgd_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# ==========================================
# 6. Train model using Adam
# ==========================================

adam_model = build_model(
    tf.keras.optimizers.Adam(learning_rate=0.001)
)

print("\nTraining with Adam...")

adam_history = adam_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# ==========================================
# 7. Evaluate both models
# ==========================================

sgd_loss, sgd_accuracy = sgd_model.evaluate(
    X_test, y_test, verbose=0
)

adam_loss, adam_accuracy = adam_model.evaluate(
    X_test, y_test, verbose=0
)

print("\n========== MODEL COMPARISON ==========")

print("\nSGD Results:")
print("Test Loss:", round(sgd_loss, 4))
print("Test Accuracy:", round(sgd_accuracy, 4))

print("\nAdam Results:")
print("Test Loss:", round(adam_loss, 4))
print("Test Accuracy:", round(adam_accuracy, 4))

# ==========================================
# 8. Plot training and validation loss curves
# ==========================================

plt.figure(figsize=(12, 5))

# SGD loss
plt.subplot(1, 2, 1)

plt.plot(
    sgd_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    sgd_history.history["val_loss"],
    label="Validation Loss"
)

plt.title("SGD Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Adam loss
plt.subplot(1, 2, 2)

plt.plot(
    adam_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    adam_history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Adam Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 9. Compare final test results
# ==========================================

results = pd.DataFrame({
    "Optimizer": ["SGD", "Adam"],
    "Test Loss": [sgd_loss, adam_loss],
    "Test Accuracy": [sgd_accuracy, adam_accuracy]
})

print("\nFinal Comparison:")
print(results)
