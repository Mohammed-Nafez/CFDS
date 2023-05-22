import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras


# Load the face dataset
face_dir = r"C:\Users\Mohammd Nafez Aloul\Documents\GUI-PyQt5\Faces"
face_files = os.listdir(face_dir)
X_face = []
for file in face_files:
    img = cv2.imread(os.path.join(face_dir, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # resize image to (64, 64)
    X_face.append(img)
y_face = [1] * len(X_face)

# Load the non-face dataset
nonface_dir = r"C:\Users\Mohammd Nafez Aloul\Documents\GUI-PyQt5\Non-faces"
nonface_files = os.listdir(nonface_dir)
X_nonface = []
for file in nonface_files:
    img = cv2.imread(os.path.join(nonface_dir, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # resize image to (64, 64)
    X_nonface.append(img)
y_nonface = [0] * len(X_nonface)

# Combine the datasets
X = X_face + X_nonface
y = y_face + y_nonface

# Convert the data to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(100, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Train the model
model.fit(X_train.reshape(-1, 64, 64, 1), y_train, validation_data=(X_test.reshape(-1, 64, 64, 1), y_test), epochs=10, batch_size=32)

# Save the model
model.save("face_recognition_model.h5")