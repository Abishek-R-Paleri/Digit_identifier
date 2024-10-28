import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Display some sample images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),           # Flatten 28x28 images into a 784-dimensional vector
    layers.Dense(128, activation='relu'),           # Hidden layer with 128 neurons and ReLU activation
    layers.Dropout(0.2),                            # Dropout layer to prevent overfitting
    layers.Dense(10, activation='softmax')          # Output layer with 10 neurons (one for each digit)
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy:.4f}")

# Making Predictions
predictions = model.predict(x_test)

# Visualize a few predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"True: {y_test[i]}, Predicted: {predictions[i].argmax()}")
    plt.axis('off')
plt.show()
