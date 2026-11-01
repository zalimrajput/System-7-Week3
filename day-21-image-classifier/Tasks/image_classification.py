
# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# 2. CONFIGURATION
# =====================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"


# =====================================================
# 3. LOAD DATASET
# =====================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_dataset.class_names

print("Classes:", class_names)


# Improve dataset loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# =====================================================
# 4. CNN FROM SCRATCH
# =====================================================

cnn_model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    # Normalize pixel values
    layers.Rescaling(1.0 / 255),

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Classification layers
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")

])


# =====================================================
# 5. COMPILE CNN MODEL
# =====================================================

cnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN FROM SCRATCH")
cnn_model.summary()


# =====================================================
# 6. TRAIN CNN FROM SCRATCH
# =====================================================

print("\nTraining CNN from Scratch...")

cnn_history = cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# =====================================================
# 7. BUILD MOBILENETV2 TRANSFER LEARNING MODEL
# =====================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False


# Build complete model
transfer_model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    # MobileNetV2 preprocessing: 0-255 to -1 to 1
    layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    ),

    # Pretrained MobileNetV2
    base_model,

    # Custom classification layers
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.3),

    # Binary classification
    layers.Dense(1, activation="sigmoid")

])


# =====================================================
# 8. COMPILE TRANSFER LEARNING MODEL
# =====================================================

transfer_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nMOBILENETV2 TRANSFER LEARNING MODEL")
transfer_model.summary()

print(
    "\nFrozen MobileNetV2:",
    not base_model.trainable
)


# =====================================================
# 9. TRAIN TRANSFER LEARNING MODEL
# =====================================================

print("\nTraining MobileNetV2...")

transfer_history = transfer_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# =====================================================
# 10. EVALUATE BOTH MODELS
# =====================================================

print("\n========== CNN FROM SCRATCH ==========")

cnn_loss, cnn_accuracy = cnn_model.evaluate(
    validation_dataset,
    verbose=1
)

print("CNN Test/Validation Loss:", cnn_loss)
print("CNN Test/Validation Accuracy:", cnn_accuracy * 100, "%")


print("\n========== MOBILENETV2 ==========")

transfer_loss, transfer_accuracy = transfer_model.evaluate(
    validation_dataset,
    verbose=1
)

print("MobileNetV2 Loss:", transfer_loss)
print(
    "MobileNetV2 Test/Validation Accuracy:",
    transfer_accuracy * 100,
    "%"
)


# =====================================================
# 11. COMPARE BOTH MODELS
# =====================================================

print("\n========== MODEL COMPARISON ==========")

print(
    f"CNN from Scratch Accuracy: {cnn_accuracy * 100:.2f}%"
)

print(
    f"MobileNetV2 Accuracy: {transfer_accuracy * 100:.2f}%"
)

if transfer_accuracy > cnn_accuracy:
    print("\nMobileNetV2 performed better.")

elif cnn_accuracy > transfer_accuracy:
    print("\nCNN from Scratch performed better.")

else:
    print("\nBoth models achieved the same accuracy.")


# =====================================================
# 12. VISUALIZE ACCURACY COMPARISON
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    cnn_history.history["val_accuracy"],
    label="CNN from Scratch"
)

plt.plot(
    transfer_history.history["val_accuracy"],
    label="MobileNetV2"
)

plt.title("Validation Accuracy Comparison")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.show()


# =====================================================
# 13. VISUALIZE TRAINING LOSS COMPARISON
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    cnn_history.history["val_loss"],
    label="CNN from Scratch"
)

plt.plot(
    transfer_history.history["val_loss"],
    label="MobileNetV2"
)

plt.title("Validation Loss Comparison")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.show()


# =====================================================
# 14. VISUALIZE SAMPLE PREDICTIONS
# =====================================================

# Get one batch of validation images
sample_images, sample_labels = next(
    iter(validation_dataset)
)

# Get predictions from both models
cnn_predictions = cnn_model.predict(sample_images)

transfer_predictions = transfer_model.predict(sample_images)


# Display first 9 images
plt.figure(figsize=(14, 14))

for i in range(min(9, len(sample_images))):

    actual_label = int(sample_labels[i][0])

    cnn_probability = float(cnn_predictions[i][0])
    transfer_probability = float(transfer_predictions[i][0])

    # Convert probabilities into class labels
    cnn_predicted = int(cnn_probability >= 0.5)

    transfer_predicted = int(
        transfer_probability >= 0.5
    )

    # Get class names
    actual_name = class_names[actual_label]

    cnn_name = class_names[cnn_predicted]

    transfer_name = class_names[transfer_predicted]

    plt.subplot(3, 3, i + 1)

    plt.imshow(
        tf.cast(sample_images[i], tf.uint8).numpy()
    )

    plt.title(
        f"Actual: {actual_name}\n"
        f"CNN: {cnn_name}\n"
        f"MobileNetV2: {transfer_name}"
    )

    plt.axis("off")


plt.tight_layout()
plt.show()


# =====================================================
# 15. CLASSIFICATION REPORT FOR BOTH MODELS
# =====================================================

# Collect all actual labels
y_true = np.concatenate([
    labels.numpy().flatten()
    for images, labels in validation_dataset
]).astype(int)


# CNN predictions for complete validation dataset
cnn_probabilities = cnn_model.predict(validation_dataset)

cnn_predicted_labels = (
    cnn_probabilities.flatten() >= 0.5
).astype(int)


# MobileNetV2 predictions
transfer_probabilities = transfer_model.predict(
    validation_dataset
)

transfer_predicted_labels = (
    transfer_probabilities.flatten() >= 0.5
).astype(int)


print("\n========== CNN CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        cnn_predicted_labels,
        target_names=class_names,
        zero_division=0
    )
)


print("\n========== MOBILENETV2 CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        transfer_predicted_labels,
        target_names=class_names,
        zero_division=0
    )
)
