import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# Load data
with open("cnn_data.pkl", "rb") as f:
    data = pickle.load(f)

x = np.array(data["data"]) / 255.0  # Normalize
y = np.array(data["labels"])        # Still in string format like 'A', 'B', ...

# Encode labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)  # One-hot encode

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')  # output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[INFO] Training CNN model...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model and label encoder
model.save("cnn_model.h5")
with open("cnn_labels.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("[INFO] Model saved as cnn_model.h5")
