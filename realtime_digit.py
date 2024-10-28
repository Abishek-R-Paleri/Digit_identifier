import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np

# Load and train the MNIST model
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Initialize webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Main loop for capturing and predicting
try:
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Create a region of interest for drawing
        roi = frame[50:300, 50:300]
        cv2.rectangle(frame, (50, 50), (300, 300), (0, 255, 0), 2)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Preprocess the ROI for model input
        _, roi_thresh = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY_INV)
        digit = cv2.resize(roi_thresh, (28, 28), interpolation=cv2.INTER_AREA)
        digit = digit.reshape(1, 28, 28) / 255.0  # Normalize to match training input

        # Make a prediction
        prediction = model.predict(digit)
        predicted_digit = np.argmax(prediction)
        
        # Display the predicted digit on the frame
        cv2.putText(frame, f"Prediction: {predicted_digit}", (50, 400), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the original frame with prediction
        cv2.imshow("Real-Time Digit Recognition", frame)
        cv2.imshow("Region of Interest", roi_thresh)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
