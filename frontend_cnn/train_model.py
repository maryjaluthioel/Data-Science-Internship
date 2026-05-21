import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =====================================================================
# 1. DEFINE PATHS AND CONSTANTS
# =====================================================================
# This points directly to your local Windows directory path
DATASET_DIR = r"C:\Users\Lenovo\Downloads\imagedataset\Snake Images" 
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# =====================================================================
# 2. LOAD DATASETS DIRECTLY FROM YOUR FOLDERS
# =====================================================================
print("Loading training dataset...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary'  # 0 for Non-Venomous, 1 for Venomous
)

print("Loading test dataset...")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Optimize pipeline streaming performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =====================================================================
# 3. CONSTRUCT THE CNN ARCHITECTURE
# =====================================================================
model = models.Sequential([
    # Standardize pixel color values from 0-255 down to 0-1 range
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Convolutional Blocks
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten & Dense Classification Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary probability output node
])

# =====================================================================
# 4. COMPILE AND TRAIN
# =====================================================================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nStarting training process...")
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)

# =====================================================================
# 5. SAVE THE MODEL
# =====================================================================
model.save("snake_model.h5")
print("\nModel trained and successfully saved as 'snake_model.h5'!")