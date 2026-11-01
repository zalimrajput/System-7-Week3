# Module D: Multiclass Classification, Transfer Learning & Model Evaluation

## Overview

This module focuses on advanced image classification techniques using Convolutional Neural Networks (CNNs) and Transfer Learning.

We learn how to classify images into multiple categories, use pretrained models, evaluate model performance, and build a practical image classifier.

## Learning Objectives

- Understand multiclass classification.
- Build image classification models using CNNs.
- Understand one-hot encoding and softmax activation.
- Learn the concept of Transfer Learning.
- Use pretrained models such as MobileNetV2.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Generate and interpret a confusion matrix.
- Build an image classifier using a custom dataset.

---

# 1. Multiclass Classification

### What is Multiclass Classification?

Multiclass classification is a machine learning task in which a model predicts one class from more than two possible classes.

For example, an image classifier may identify an image as:

- Cat
- Dog
- Horse
- Bird
- Car

Unlike binary classification, which predicts between two classes, multiclass classification predicts among multiple categories.

### Example: CIFAR-10 Dataset

CIFAR-10 contains 60,000 color images divided into 10 classes.

| Class | Description |
|---|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

### Load CIFAR-10 Dataset

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import datasets

# Load dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

print("Training Images:", x_train.shape)
print("Training Labels:", y_train.shape)

print("Testing Images:", x_test.shape)
print("Testing Labels:", y_test.shape)
```

### Expected Output

```text
Training Images: (50000, 32, 32, 3)
Training Labels: (50000, 1)

Testing Images: (10000, 32, 32, 3)
Testing Labels: (10000, 1)
```

### Normalize Image Data

Pixel values range from 0 to 255. We divide by 255 to convert them into the range 0 to 1.

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

### Build a Multiclass Classification CNN

```python
from tensorflow.keras import layers, models

model = models.Sequential([

    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dense(10, activation="softmax")
])

model.summary()
```

### Compile the Model

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

**Why sparse categorical crossentropy?**

CIFAR-10 labels are integer values from 0 to 9. Sparse categorical crossentropy is suitable when multiclass labels are stored as integers rather than one-hot encoded vectors.

### Train the Model

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)
```

---

# 2. Transfer Learning

### What is Transfer Learning?

Transfer Learning is a technique in which we use knowledge learned by a model on one dataset to solve a different but related task.

Instead of training a CNN from scratch, we use a pretrained model that has already learned useful image features.

For example:

- A pretrained model learns edges, shapes, textures, and patterns.
- We reuse these learned features.
- We add a new classification layer for our own image categories.
- We train the new classification layer on our dataset.

### Popular Pretrained Models

| Model | Description |
|---|---|
| VGG16 | Deep CNN architecture |
| ResNet50 | Uses residual connections |
| MobileNetV2 | Lightweight and efficient |
| EfficientNet | Efficient model scaling |
| InceptionV3 | Uses multiple convolution operations |

### Example: Transfer Learning with MobileNetV2

We will use MobileNetV2 pretrained on ImageNet.

For this example, create the following dataset structure:

```text
Module-D/
│
├── transfer_learning.py
│
└── dataset/
    │
    ├── train/
    │   ├── cats/
    │   └── dogs/
    │
    └── validation/
        ├── cats/
        └── dogs/
```

Place your cat and dog images in their respective folders.

### Load Training and Validation Images

```python
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Classes:", train_dataset.class_names)
```

### Load Pretrained MobileNetV2

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False
```

**Understanding the parameters:**

- `weights="imagenet"`: Loads pretrained weights learned from ImageNet.
- `include_top=False`: Removes the original classification layers.
- `input_shape=(224, 224, 3)`: Accepts 224×224 RGB images.
- `base_model.trainable=False`: Keeps pretrained weights frozen during initial training.

### Build Transfer Learning Model

```python
model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    # MobileNetV2 preprocessing
    layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    ),

    # Pretrained feature extractor
    base_model,

    # Classification layers
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.3),

    # Binary output: cat or dog
    layers.Dense(1, activation="sigmoid")
])

model.summary()
```

### Compile the Model

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

We use binary crossentropy because this particular example has two classes: cats and dogs.

### Train the Transfer Learning Model

```python
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)
```

### Optional: Fine-Tuning

After training the classification layers, we can unfreeze some pretrained layers and train them with a smaller learning rate.

```python
base_model.trainable = True

# Freeze most pretrained layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)
```

Fine-tuning allows the pretrained model to adapt some of its learned features to our own dataset.

---

# 3. Model Evaluation

### What is Model Evaluation?

Model evaluation measures how well a trained model performs on unseen data.

For classification problems, we commonly use:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. Confusion Matrix

### 3.1 Accuracy

Accuracy measures the percentage of correctly predicted samples.

Formula:

Accuracy = Correct Predictions / Total Predictions

### 3.2 Precision

Precision measures how many positive predictions were actually correct.

Precision = TP / (TP + FP)

### 3.3 Recall

Recall measures how many actual positive samples were correctly identified.

Recall = TP / (TP + FN)

### 3.4 F1-Score

F1-score is the harmonic mean of precision and recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

### 3.5 Confusion Matrix

A confusion matrix compares actual labels against predicted labels.

For multiclass classification:

- Rows represent actual classes.
- Columns represent predicted classes.
- Diagonal values represent correct predictions.
- Off-diagonal values represent incorrect predictions.

### Evaluate CIFAR-10 CNN Model

Use the trained CIFAR-10 model from Section 1.

```python
# Evaluate model
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")
```

### Generate Predictions

```python
import numpy as np

y_pred_prob = model.predict(x_test)

# Convert probabilities into predicted class labels
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert actual labels into 1D array
y_true = y_test.flatten()
```

### Classification Report

```python
from sklearn.metrics import classification_report

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

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

plt.title("CIFAR-10 Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")

plt.show()
```

---
