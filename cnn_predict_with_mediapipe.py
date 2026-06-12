import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# Load CNN model
model = load_model('cnn_model.h5')  # Make sure path is correct

# Load labels from training dataset directory
labels = sorted(os.listdir('archive/asl_alphabet_train'))
print(f"Loaded {len(labels)} labels: {labels}")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    # Flip and convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmarks
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Add some margin
            margin = 20
            x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
            x_max, y_max = min(x_max + margin, w), min(y_max + margin, h)

            # Extract hand ROI and preprocess
            roi = frame[y_min:y_max, x_min:x_max]
            try:
                roi_resized = cv2.resize(roi, (64, 64))  # Match your model input
                roi_normalized = roi_resized / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=0)

                # Predict gesture
                prediction = model.predict(roi_input)
                idx = np.argmax(prediction)
                print(f"Predicted index: {idx}")

                # Avoid index error
                if idx < len(labels):
                    label = labels[idx]
                else:
                    label = "Unknown"
                    print("Warning: Predicted index out of range")

                # Show prediction
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            except Exception as e:
                print(f"Error processing ROI: {e}")

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
