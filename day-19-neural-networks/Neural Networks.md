# Module C: Neural Networks and Deep Learning

This module focuses on the fundamentals of neural networks and deep learning using Python, NumPy, Matplotlib, Scikit-learn, and TensorFlow/Keras.

## Topics Covered

1. Perceptrons
2. Backpropagation
3. Gradient Descent
4. Optimizers
5. Keras Workflow
6. Student Results Prediction Task

---

## 1. Perceptron

A perceptron is the simplest type of artificial neural network. It takes input features, multiplies them by weights, adds a bias, and applies an activation function to produce an output.

### Mathematical Formula

Weighted sum:

\[
z = w_1x_1 + w_2x_2 + b
\]

Step activation function:

\[
f(z) =
\begin{cases}
1 & z \geq 0 \\
0 & z < 0
\end{cases}
\]

### Example Code: Simple Perceptron Using NumPy

```python
import numpy as np

# Input features
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# AND gate expected outputs
y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0

learning_rate = 0.1
epochs = 10

# Perceptron training
for epoch in range(epochs):

    for i in range(len(X)):

        # Calculate weighted sum
        z = np.dot(X[i], weights) + bias

        # Step activation
        prediction = 1 if z >= 0 else 0

        # Calculate error
        error = y[i] - prediction

        # Update weights and bias
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Test predictions
print("Final Weights:", weights)
print("Final Bias:", bias)

print("\nPredictions:")

for sample in X:
    z = np.dot(sample, weights) + bias
    prediction = 1 if z >= 0 else 0
    print(sample, "Prediction:", prediction)
```

**Key Concepts:**
- Weights determine the importance of input features.
- Bias shifts the decision boundary.
- Learning rate controls the size of updates.
- Epoch represents one complete pass through the training dataset.

---

## 2. Backpropagation

Backpropagation is an algorithm used to train neural networks. It calculates how much each weight contributes to the prediction error and propagates the error backward through the network.

### Backpropagation Workflow

1. Forward propagation: Calculate predictions.
2. Calculate loss: Measure prediction error.
3. Backward propagation: Calculate gradients.
4. Update weights: Reduce the loss.
5. Repeat until training is complete.

### Important Formulas

Mean Squared Error:

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

Gradient descent weight update:

\[
w = w - \alpha \frac{\partial L}{\partial w}
\]

Where:
- \(w\) = Weight
- \(\alpha\) = Learning rate
- \(L\) = Loss function
- \(\frac{\partial L}{\partial w}\) = Gradient of the loss with respect to the weight

### Example Code: Backpropagation Using Keras

```python
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Sample input data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
], dtype=float)

# Target output
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Build neural network
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mse"
)

# Train model
history = model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)

print("Final Loss:", history.history["loss"][-1])

# Prediction
prediction = model.predict(
    np.array([[6, 7]], dtype=float),
    verbose=0
)

print("Prediction:", prediction[0][0])
```

**Note:** Keras automatically performs backpropagation during model training. We do not manually calculate every gradient in this example.

---

## 3. Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function by updating model parameters in the direction of the negative gradient.

### Mathematical Formula

\[
w_{new} = w_{old} - \alpha \times gradient
\]

### Example Code: Gradient Descent Using NumPy

```python
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Initialize parameters
weight = 0.0
bias = 0.0

learning_rate = 0.01
epochs = 100

loss_history = []

# Gradient descent training
for epoch in range(epochs):

    # Forward propagation
    y_pred = weight * X + bias

    # Calculate loss
    loss = np.mean((y - y_pred) ** 2)

    loss_history.append(loss)

    # Calculate gradients
    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    # Update parameters
    weight -= learning_rate * dw
    bias -= learning_rate * db

print("Final Weight:", weight)
print("Final Bias:", bias)
print("Final Loss:", loss_history[-1])

# Plot loss
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Loss")
plt.grid()
plt.show()
```

**Key Concepts:**
- Gradient descent updates weights to reduce loss.
- A very small learning rate can make training slow.
- A very large learning rate can make training unstable.
- Training should progressively reduce the loss when optimization is working properly.

---

## 4. Optimizers

Optimizers update neural network weights and biases during training to minimize the loss function.

### Common Optimizers

| Optimizer | Description |
|---|---|
| SGD | Updates parameters using gradient descent |
| Momentum | Uses previous updates to improve optimization |
| RMSprop | Adapts learning rates using moving averages of squared gradients |
| Adam | Combines momentum and adaptive learning rates |

### Example Code: Compare Optimizers

```python
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# Sample dataset
X = np.random.rand(500, 2)

y = (
    X[:, 0] * 2 +
    X[:, 1] * 3
)

optimizers = {
    "SGD": keras.optimizers.SGD(learning_rate=0.01),
    "RMSprop": keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam": keras.optimizers.Adam(learning_rate=0.001)
}

histories = {}

for name, optimizer in optimizers.items():

    # Create a fresh model for each optimizer
    model = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=optimizer,
        loss="mse"
    )

    history = model.fit(
        X,
        y,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    histories[name] = history.history["loss"]

    print(name, "Final Loss:", history.history["loss"][-1])

# Plot comparison
for name, losses in histories.items():
    plt.plot(losses, label=name)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")
plt.legend()
plt.grid()
plt.show()
```

**Note:** Optimizer performance depends on the dataset, model architecture, learning rate, and other training settings. There is no optimizer that is always best.

---

## 5. Keras Workflow

Keras provides a high-level API for building, compiling, training, and evaluating neural networks.

### Main Steps

1. Load the dataset.
2. Prepare and preprocess the data.
3. Split into training and testing sets.
4. Build the neural network.
5. Compile the model.
6. Train the model.
7. Evaluate the model.
8. Make predictions.
9. Save the trained model.

### Example Code: Binary Classification

```python
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers

# Generate sample dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions
predictions = model.predict(X_test)

predicted_classes = (predictions >= 0.5).astype(int)

print("First 10 Predictions:")
print(predicted_classes[:10].flatten())

# Save model
model.save("classification_model.keras")
```

---

## 6. Student Results Prediction Task

### Task Description

Build a neural network that predicts whether a student will pass or fail based on study hours and attendance.

### Dataset

We will generate a sample dataset containing:

| Feature | Description |
|---|---|
| Study Hours | Number of hours studied |
| Attendance | Student attendance percentage |
| Result | 0 = Fail, 1 = Pass |

### Complete Example Code

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# 1. GENERATE SAMPLE DATASET
# ==========================================

np.random.seed(42)

study_hours = np.random.uniform(0, 10, 500)
attendance = np.random.uniform(40, 100, 500)

# Generate results using sample conditions
result = (
    (study_hours * 5 + attendance * 0.4) >= 35
).astype(int)

X = np.column_stack((study_hours, attendance))
y = result

print("Dataset Shape:", X.shape)

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
# 3. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 4. BUILD NEURAL NETWORK
# ==========================================

model = keras.Sequential([
    layers.Input(shape=(2,)),

    layers.Dense(16, activation="relu"),

    layers.Dense(8, activation="relu"),

    layers.Dense(1, activation="sigmoid")
])

# ==========================================
# 5. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. TRAIN MODEL
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ==========================================
# 7. EVALUATE MODEL
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)

# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_probability = model.predict(X_test, verbose=0)

y_pred = (y_probability >= 0.5).astype(int).flatten()

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 9. PLOT TRAINING HISTORY
# ==========================================

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")

plt.legend()
plt.grid()

# Loss plot
plt.subplot(1, 2, 2)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# ==========================================
# 10. TEST A NEW STUDENT
# ==========================================

new_student = np.array([[6, 85]])

# Apply the same fitted scaler
new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled, verbose=0)[0][0]

print("\nNew Student Prediction:")

if prediction >= 0.5:
    print("Result: PASS")
else:
    print("Result: FAIL")

print("Pass Probability:", round(float(prediction), 4))

# ==========================================
# 11. SAVE MODEL
# ==========================================

model.save("student_results_model.keras")

print("\nModel Saved Successfully!")
```
