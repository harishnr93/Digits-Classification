"""
Date: 14.Oct.2024
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(history):
    plt.figure(figsize=(12, 4))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    #plt.show()
    return

def plot_all_samples(x_train, y_train, x_val, y_val, x_test, y_test, num_images=5):
    """
    Plot sample images from the training, validation, and test datasets.
    Displays images in rows labeled by dataset type.
    """
    datasets = [("Training", x_train, y_train), 
                ("Validation", x_val, y_val), 
                ("Test", x_test, y_test)]
    
    plt.figure(figsize=(15, 6))
    for row, (name, x_data, y_data) in enumerate(datasets):
        for col in range(num_images):
            plt.subplot(len(datasets), num_images, row * num_images + col + 1)
            plt.imshow(x_data[col], cmap='gray')  # Display image in grayscale
            plt.title(f"{name}\nLabel: {y_data[col]}")
            plt.axis('off')
    
    plt.tight_layout()
    #plt.show()
    return

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Split training data into training and validation sets
validation_split = 0.2
split_index = int((1 - validation_split) * len(x_train))
x_train, x_val = x_train[:split_index], x_train[split_index:]
y_train, y_val = y_train[:split_index], y_train[split_index:]

# Plot images from training, validation, and test datasets
plot_all_samples(x_train, y_train, x_val, y_val, x_test, y_test, num_images=10)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # classes for digits 0-9
])

# Compile the model
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=32)

# Evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy:.2f}")

# Validate predictions
y_pred = np.argmax(model.predict(x_test), axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

plot_metrics(history)

print("Done!")
plt.show()