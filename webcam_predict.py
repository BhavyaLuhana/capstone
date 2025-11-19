import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model('traffic_sign_model.h5')

# Load labels
with open('data8.pickle', 'rb') as f:
    data = pickle.load(f)
labels = data['labels']

# Webcam setup
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    size = 250  # bigger region for better visibility
    x1, y1 = w//2 - size//2, h//2 - size//2
    x2, y2 = x1 + size, y1 + size

    roi = frame[y1:y2, x1:x2]

    # ✅ Denoise + preprocess
    roi = cv2.bilateralFilter(roi, 5, 75, 75)
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    label = labels[class_idx] if class_idx < len(labels) else f"Class {class_idx}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Traffic Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
