import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Initialize webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a mirror-like interaction
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI) for digit drawing
    x_start, y_start, box_size = 50, 50, 250
    roi = frame[y_start:y_start + box_size, x_start:x_start + box_size]
    cv2.rectangle(frame, (x_start, y_start), (x_start + box_size, y_start + box_size), (0, 255, 0), 2)
    
    # Process the ROI for prediction
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY_INV)
    digit = cv2.resize(roi_thresh, (28, 28), interpolation=cv2.INTER_AREA)
    digit = digit.reshape(1, 28, 28, 1) / 255.0  # Reshape and normalize

    # Make a prediction
    prediction = model.predict(digit)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display prediction with confidence score
    cv2.putText(frame, f"Prediction: {predicted_digit} ({confidence:.2f})", 
                (x_start, y_start + box_size + 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the thresholded ROI and the main frame
    cv2.imshow("Region of Interest", roi_thresh)
    cv2.imshow("Real-Time Digit Recognition", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
