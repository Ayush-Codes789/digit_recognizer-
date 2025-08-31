import cv2
import numpy as np
import tensorflow as tf

# Load trained CNN model
model = tf.keras.models.load_model("mnist_cnn.h5")

# Start webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define ROI (200x200 center box)
    x, y, w, h = 200, 100, 200, 200
    roi = gray[y:y+h, x:x+w]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (28, 28))
    _, roi_thresh = cv2.threshold(roi_resized, 128, 255, cv2.THRESH_BINARY_INV)
    roi_normalized = roi_thresh / 255.0
    roi_final = roi_normalized.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(roi_final)
    digit = np.argmax(prediction)

    # Show ROI + prediction
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(frame, f"Prediction: {digit}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Digit Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

