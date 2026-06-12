import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load CNN model
model = load_model("cnn_model.h5")

# Label map (used in training order)
labels = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space' ]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI) for prediction (center square box)
    x1, y1, x2, y2 = 200, 100, 450, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    # Predict
    prediction = model.predict(roi)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = prediction[0][predicted_index]

    # Show prediction
    cv2.putText(frame, f'{predicted_label} ({confidence:.2f})', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Display
    cv2.imshow("Hand Gesture Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
