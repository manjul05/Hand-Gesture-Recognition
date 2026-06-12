import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)
# Labels used during training (must match the training labels exactly)
labels = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space' ]

# Setup MediaPipe
mp_hands = mp.solutions.hands            
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Gesture Recognition")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip to mirror view
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            try:
                # Extract landmarks
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # Now 21 × 3 = 63 features


                landmarks = np.array(landmarks).reshape(1, -1)

                # Predict
                prediction = model.predict(landmarks)
                predicted_label = prediction[0]

                cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            except Exception as e:
                cv2.putText(frame, f"Prediction failed", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("[ERROR]", str(e))
    else:
        cv2.putText(frame, "Prediction: Unknown", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
