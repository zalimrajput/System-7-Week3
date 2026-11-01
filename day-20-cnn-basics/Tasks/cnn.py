
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix, classification_report


# ==========================================
# 2. LOAD CIFAR-10 DATASET
# ==========================================

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

print("Training Images Shape:", x_train.shape)
print("Training Labels Shape:", y_train.shape)

print("Testing Images Shape:", x_test.shape)
print("Testing Labels Shape:", y_test.shape)


# ==========================================
# 3. DEFINE CLASS NAMES
# ==========================================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ==========================================
# 4. NORMALIZE IMAGE DATA
# ==========================================

# Convert pixel values from 0-255 to 0-1

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Normalized Training Shape:", x_train.shape)
print("Normalized Testing Shape:", x_test.shape)


# ==========================================
# 5. DISPLAY SAMPLE IMAGES
# ==========================================

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 6. BUILD CNN MODEL
# ==========================================

model = models.Sequential([

    # First Convolution Block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(32, 32, 3)
    ),

    layers.MaxPooling2D((2, 2)),

    # Second Convolution Block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Third Convolution Block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Flatten Feature Maps
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(128, activation="relu"),

    # Output Layer: 10 CIFAR-10 Classes
    layers.Dense(10, activation="softmax")
])


# ==========================================
# 7. DISPLAY MODEL ARCHITECTURE
# ==========================================

model.summary()


# ==========================================
# 8. COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# 9. TRAIN MODEL
# ==========================================

history = model.fit(

    x_train,
    y_train,

    epochs=10,

    batch_size=64,

    validation_split=0.2
)


# ==========================================
# 10. EVALUATE MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n========== TEST RESULTS ==========")

print("Test Loss:", round(test_loss, 4))

print("Test Accuracy:", round(test_accuracy * 100, 2), "%")


# ==========================================
# 11. PLOT TRAINING AND VALIDATION ACCURACY
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Training vs Validation Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# 12. PLOT TRAINING AND VALIDATION LOSS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("CNN Training vs Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# 13. MAKE PREDICTIONS
# ==========================================

y_pred_prob = model.predict(x_test)

# Get class with highest probability
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert actual labels into 1D array
y_true = y_test.flatten()


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names

)

plt.title("CIFAR-10 CNN Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.show()


# ==========================================
# 15. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)
