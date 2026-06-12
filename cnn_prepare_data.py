import os
import cv2
import pickle
import numpy as np

# Dataset path
DATASET_PATH = "archive/asl_alphabet_train"

data = []
labels = []

print("[INFO] Loading images...")

label_folders = sorted(os.listdir(DATASET_PATH))

for label in label_folders:
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue

    print(f"[INFO] Processing label: {label}")

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # resize for CNN

        data.append(img)
        labels.append(label)

print(f"[INFO] Total images: {len(data)}")

# Save the image dataset
with open("cnn_data.pkl", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("[INFO] cnn_data.pkl created successfully!")
