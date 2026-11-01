import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers

# ==========================================
# 1. LOAD IMAGE
# ==========================================

image = tf.keras.utils.load_img("cat.jpg")

# Convert image into NumPy array
image = tf.keras.utils.img_to_array(image)

print("Original Image Shape:", image.shape)

# ==========================================
# 2. RESIZE IMAGE TO 64x64
# ==========================================

resized_image = tf.image.resize(image, (64, 64))

print("Resized Image Shape:", resized_image.shape)

# ==========================================
# 3. NORMALIZE IMAGE
# ==========================================

# Convert pixel values from 0-255 to 0-1
normalized_image = resized_image / 255.0

print("Minimum Pixel Value:",
      tf.reduce_min(normalized_image).numpy())

print("Maximum Pixel Value:",
      tf.reduce_max(normalized_image).numpy())

# ==========================================
# 4. DISPLAY ORIGINAL PROCESSED IMAGE
# ==========================================

plt.figure(figsize=(5, 5))

plt.imshow(normalized_image.numpy())
plt.title("Resized and Normalized Image")
plt.axis("off")

plt.show()

# ==========================================
# 5. CREATE AUGMENTATION LAYERS
# ==========================================

data_augmentation = tf.keras.Sequential([

    # Random horizontal flipping
    layers.RandomFlip("horizontal"),

    # Random rotation
    layers.RandomRotation(0.2),

    # Random zoom
    layers.RandomZoom(0.2),

    # Random contrast
    layers.RandomContrast(0.2)

])

# ==========================================
# 6. ADD BATCH DIMENSION
# ==========================================

# CNN expects 4D input:
# (Batch, Height, Width, Channels)

image_batch = tf.expand_dims(
    normalized_image,
    axis=0
)

print("Batch Image Shape:", image_batch.shape)

# ==========================================
# 7. DISPLAY 9 AUGMENTED SAMPLES
# ==========================================

plt.figure(figsize=(10, 10))

for i in range(9):

    # Apply random augmentation
    augmented_image = data_augmentation(
        image_batch,
        training=True
    )

    # Remove batch dimension
    augmented_image = augmented_image[0]

    # Display image
    plt.subplot(3, 3, i + 1)

    plt.imshow(
        tf.clip_by_value(
            augmented_image, 0, 1
        ).numpy()
    )

    plt.title(f"Augmented Image {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()