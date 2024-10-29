import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import numpy as np

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # Expand dims for CNN

# Define the improved model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Test accuracy on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

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

    # Define a fixed ROI area where the user can draw the digit
    x_start, y_start, box_size = 50, 50, 250
    roi = frame[y_start:y_start + box_size, x_start:x_start + box_size]

    # Convert the ROI to grayscale and apply a binary threshold
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box around the detected contour
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region inside the bounding box, resize to 28x28, and normalize
        digit = roi_thresh[y:y + h, x:x + w]
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
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
