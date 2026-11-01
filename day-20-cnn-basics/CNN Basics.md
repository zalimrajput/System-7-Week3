# Module D: Image Processing and CNN Basics

This module introduces the fundamental concepts of image processing and Convolutional Neural Networks (CNNs). It covers image preprocessing, data augmentation, convolution operations, pooling layers, and the basic architecture of CNNs using Python and TensorFlow/Keras.

## 1. Image Preprocessing

Image preprocessing is the process of preparing raw images before feeding them into a deep learning model.

Images often have different dimensions, pixel values, and color formats. Preprocessing makes the data consistent and suitable for model training.

### Common preprocessing techniques

- **Resizing:** Convert images into a fixed width and height.
- **Normalization:** Scale pixel values from 0–255 to 0–1.
- **Color conversion:** Convert RGB images to grayscale when needed.
- **Reshaping:** Adjust image dimensions according to model requirements.

### Example 1: Load and Display an Image

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Load an image
image = tf.keras.utils.load_img("image.jpg")

# Convert image into NumPy array
image_array = tf.keras.utils.img_to_array(image)

print("Image Shape:", image_array.shape)

# Display image
plt.imshow(image_array.astype("uint8"))
plt.axis("off")
plt.show()
```

### Example 2: Resize an Image

```python
# Resize image to 128x128
resized_image = tf.image.resize(image_array, (128, 128))

print("Original Shape:", image_array.shape)
print("Resized Shape:", resized_image.shape)

plt.imshow(resized_image.numpy().astype("uint8"))
plt.axis("off")
plt.show()
```

### Example 3: Normalize Image

Pixel values normally range from 0 to 255.

Normalization converts them into values between 0 and 1.

Formula:

```text
Normalized Pixel = Original Pixel / 255
```

```python
normalized_image = resized_image / 255.0

print("Minimum Pixel Value:", tf.reduce_min(normalized_image).numpy())
print("Maximum Pixel Value:", tf.reduce_max(normalized_image).numpy())
```

## 2. Image Augmentation

Image augmentation is a technique used to artificially increase the diversity of training images by applying random transformations.

It helps reduce overfitting and improves the model's ability to generalize to unseen images.

### Common augmentation techniques

1. Rotation – Rotates images.
2. Flipping – Flips images horizontally or vertically.
3. Zooming – Zooms into or out of images.
4. Translation – Shifts images horizontally or vertically.
5. Contrast adjustment – Changes image contrast.

### Example 1: Image Augmentation Using Keras

```python
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

# Add batch dimension
image_batch = tf.expand_dims(normalized_image, axis=0)

# Apply augmentation
augmented_image = data_augmentation(image_batch, training=True)

# Display augmented image
plt.imshow(augmented_image[0].numpy())
plt.axis("off")
plt.show()
```

### Example 2: Display Multiple Augmented Images

```python
plt.figure(figsize=(10, 8))

for i in range(9):
    augmented = data_augmentation(image_batch, training=True)

    plt.subplot(3, 3, i + 1)
    plt.imshow(augmented[0].numpy())
    plt.axis("off")

plt.tight_layout()
plt.show()
```

**Important:** Apply augmentation to training images only, not validation or test images.

## 3. Convolution Operation

Convolution is the main operation used in CNNs to extract important features from images.

A small matrix called a **filter (kernel)** moves across an image and performs mathematical calculations to detect features.

CNN filters can learn to detect:

- Edges
- Corners
- Shapes
- Textures
- Patterns

### How convolution works

1. Take an image.
2. Apply a filter to a small region.
3. Multiply corresponding values.
4. Add the multiplication results.
5. Move the filter across the image.
6. Generate a feature map.

### Example 1: Basic Convolution Using NumPy

```python
import numpy as np
from scipy.signal import correlate2d

# Input image (5x5)
image = np.array([
    [1, 2, 3, 0, 1],
    [4, 5, 6, 1, 2],
    [7, 8, 9, 2, 3],
    [1, 2, 3, 4, 5],
    [4, 5, 6, 7, 8]
])

# 3x3 filter
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Apply convolution-style cross-correlation
feature_map = correlate2d(
    image,
    kernel,
    mode="valid"
)

print("Input Image:")
print(image)

print("\nFeature Map:")
print(feature_map)
```

**Note:** Deep learning frameworks conventionally implement cross-correlation rather than mathematically flipped convolution. The example uses cross-correlation.

### Example 2: Convolution Using TensorFlow

```python
# Create an input image
input_image = tf.random.normal((1, 5, 5, 1))

# Create a 3x3 filter
filter_kernel = tf.ones((3, 3, 1, 1))

# Apply convolution
output = tf.nn.conv2d(
    input_image,
    filter_kernel,
    strides=1,
    padding="VALID"
)

print("Input Shape:", input_image.shape)
print("Output Shape:", output.shape)
```

## 4. Pooling Operation

Pooling is used in CNNs to reduce the spatial dimensions (height and width) of feature maps while retaining important information.

It helps:

- Reduce computational complexity.
- Reduce the number of parameters in later layers.
- Make feature extraction more efficient.
- Provide some tolerance to small spatial changes.

### Types of Pooling

**1. Max Pooling**

Selects the maximum value from each region.

Example:

```text
Input:
[[1, 3, 2, 4],
 [5, 6, 1, 2],
 [7, 2, 8, 3],
 [4, 1, 5, 9]]

Output:
[[6, 4],
 [7, 9]]
```

**2. Average Pooling**

Calculates the average value of each region.

Example:

```text
Input:
[[1, 3, 2, 4],
 [5, 6, 1, 2],
 [7, 2, 8, 3],
 [4, 1, 5, 9]]

Output:
[[3.75, 2.25],
 [3.5, 6.25]]
```

### Example 1: Max Pooling

```python
input_image = tf.constant([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [7, 2, 8, 3],
    [4, 1, 5, 9]
], dtype=tf.float32)

input_image = tf.reshape(input_image, (1, 4, 4, 1))

max_pool = tf.nn.max_pool2d(
    input_image,
    ksize=2,
    strides=2,
    padding="VALID"
)

print(max_pool.numpy().reshape(2, 2))
```

### Example 2: Average Pooling

```python
avg_pool = tf.nn.avg_pool2d(
    input_image,
    ksize=2,
    strides=2,
    padding="VALID"
)

print(avg_pool.numpy().reshape(2, 2))
```

## 5. CNN Basics (Convolutional Neural Networks)

A Convolutional Neural Network (CNN) is a deep learning architecture mainly used for processing image data.

Unlike traditional neural networks, CNNs automatically learn spatial features from images through convolution and pooling operations.

### Main CNN Layers

| Layer | Purpose |
|---|---|
| Input Layer | Receives image data |
| Convolution Layer | Extracts image features |
| ReLU Activation | Introduces non-linearity |
| Pooling Layer | Reduces feature map dimensions |
| Flatten Layer | Converts feature maps into a 1D vector |
| Dense Layer | Learns classification patterns |
| Output Layer | Produces final predictions |

### Basic CNN Architecture

```text
Input Image (128x128x3)
          |
          v
Convolution Layer
          |
          v
      ReLU
          |
          v
     Max Pooling
          |
          v
Convolution Layer
          |
          v
      ReLU
          |
          v
     Max Pooling
          |
          v
       Flatten
          |
          v
    Dense Layer
          |
          v
    Output Layer
          |
          v
   Final Prediction
```

### Example: Build a Basic CNN Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Create CNN model
model = models.Sequential([

    # Input image: 128x128 RGB
    layers.Input(shape=(128, 128, 3)),

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Flatten feature maps
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(128, activation="relu"),

    # Binary classification output
    layers.Dense(1, activation="sigmoid")
])

# Display model architecture
model.summary()
```

### Compile the CNN Model

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Train the CNN Model

For this example, assume `train_ds` and `validation_ds` are already prepared TensorFlow image datasets containing two classes.

```python
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10
)
```

### Evaluate the Model

```python
loss, accuracy = model.evaluate(validation_ds)

print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)
```

## 6. Important CNN Concepts

| Concept | Explanation |
|---|---|
| Filter / Kernel | Small matrix used to extract features |
| Stride | Number of pixels the filter moves each time |
| Padding | Adds extra pixels around an image |
| Feature Map | Output generated after convolution |
| Activation Function | Adds non-linearity, commonly ReLU |
| Pooling | Reduces feature map dimensions |
| Flattening | Converts multidimensional data into a vector |
| Epoch | One complete pass through training data |
| Batch Size | Number of samples processed together |

### Padding Types

- **Valid Padding:** No extra pixels are added around the image.
- **Same Padding:** Padding is added to maintain the spatial dimensions when stride is 1.

### Stride Example

```python
# Stride 1
output_1 = tf.nn.conv2d(
    input_image,
    filter_kernel,
    strides=1,
    padding="VALID"
)

# Stride 2
output_2 = tf.nn.conv2d(
    input_image,
    filter_kernel,
    strides=2,
    padding="VALID"
)

print("Stride 1 Shape:", output_1.shape)
print("Stride 2 Shape:", output_2.shape)
```
