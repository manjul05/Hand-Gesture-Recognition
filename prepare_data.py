import os
import cv2
import pickle

# Path to your ASL dataset
DATASET_PATH = "archive/asl_alphabet_train"

data = []
labels = []

print("[INFO] Loading images...")

# List all folders like A, B, ..., Z, space, nothing, del
label_folders = sorted(os.listdir(DATASET_PATH))
print("[INFO] Found folders:", label_folders)

# Loop through each label folder
for label in label_folders:
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"[INFO] Processing label: {label}")

    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)

        # Read the image
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize to 64x64 and flatten
        img = cv2.resize(img, (64, 64))
        data.append(img.flatten())
        labels.append(label)

print(f"[INFO] Total samples: {len(data)}")

# Save data
with open("data.pkl", "wb") as f:
    pickle.dump((data, labels), f)

print("[INFO] Data saved to 'data.pkl'")
