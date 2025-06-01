# Part 3: Image Classification with Generated Images

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import os

# --- Parameters ---
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 28, 28, 1
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
NUM_CLASSES = 10
BATCH_SIZE_CNN = 128
EPOCHS_CNN = 10 # Epochs for CNN training

# GAN Params (should match gan_mnist.py)
GAN_LATENT_DIM = 100
GAN_SAVE_DIR = 'gan_outputs'

# Number of GAN images to generate for training the second CNN
NUM_GAN_IMAGES_TO_GENERATE = 50000 # Adjust as needed, e.g., similar to MNIST train size

# Create a directory for CNN outputs if it doesn't exist
CNN_SAVE_DIR = 'cnn_outputs'
if not os.path.exists(CNN_SAVE_DIR):
    os.makedirs(CNN_SAVE_DIR)

# --- 1. Load and Preprocess Original MNIST Data ---
(x_train_real, y_train_real), (x_test_real, y_test_real) = mnist.load_data()

def preprocess_cnn_images(images):
    # Normalize images to [0, 1] for CNN
    images = images.astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1)
    return images

x_train_real_cnn = preprocess_cnn_images(x_train_real)
x_test_real_cnn = preprocess_cnn_images(x_test_real)

# Convert labels to one-hot encoding
y_train_real_cnn = keras.utils.to_categorical(y_train_real, NUM_CLASSES)
y_test_real_cnn = keras.utils.to_categorical(y_test_real, NUM_CLASSES)

print(f"x_train_real_cnn shape: {x_train_real_cnn.shape}")
print(f"y_train_real_cnn shape: {y_train_real_cnn.shape}")
print(f"x_test_real_cnn shape: {x_test_real_cnn.shape}")
print(f"y_test_real_cnn shape: {y_test_real_cnn.shape}")


# --- 2. Define CNN Classifier Model ---
def build_cnn_classifier():
    model = models.Sequential(name='cnn_classifier')
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax')) # Softmax for multi-class classification

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 3. Train CNN on Original MNIST Data (Baseline) ---
print("\n--- Training CNN on REAL MNIST data (Baseline) ---")
cnn_baseline = build_cnn_classifier()
cnn_baseline.summary()

history_baseline = cnn_baseline.fit(x_train_real_cnn, y_train_real_cnn,
                                    epochs=EPOCHS_CNN,
                                    batch_size=BATCH_SIZE_CNN,
                                    validation_data=(x_test_real_cnn, y_test_real_cnn))

baseline_loss, baseline_accuracy = cnn_baseline.evaluate(x_test_real_cnn, y_test_real_cnn, verbose=0)
print(f"\nBaseline CNN Test Accuracy (trained on real data): {baseline_accuracy*100:.2f}%")
cnn_baseline.save(os.path.join(CNN_SAVE_DIR, 'cnn_baseline_model.keras'))
print(f"Baseline CNN model saved to {os.path.join(CNN_SAVE_DIR, 'cnn_baseline_model.keras')}")

# Plot baseline CNN training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_baseline.history['accuracy'], label='Accuracy')
plt.plot(history_baseline.history['val_accuracy'], label='Val Accuracy')
plt.title('Baseline CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_baseline.history['loss'], label='Loss')
plt.plot(history_baseline.history['val_loss'], label='Val Loss')
plt.title('Baseline CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CNN_SAVE_DIR, 'cnn_baseline_training_history.png'))
print(f"Baseline CNN training history plot saved.")
plt.close()


# --- 4. Load Trained GAN Generator ---
def build_gan_generator_for_cnn(): # Renamed to avoid conflict if running in same notebook
    model = models.Sequential(name='gan_generator_loaded')
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(GAN_LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(IMG_CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

gan_generator_loaded = build_gan_generator_for_cnn()
try:
    gan_generator_loaded.load_weights(os.path.join(GAN_SAVE_DIR, 'gan_generator_weights.weights.h5'))
    print("\nGAN generator weights loaded successfully for image generation.")
except Exception as e:
    print(f"Error loading GAN generator weights: {e}. Make sure 'gan_mnist.py' was run.")
    exit()

# --- 5. Generate Synthetic Image Dataset using GAN ---
print(f"\n--- Generating {NUM_GAN_IMAGES_TO_GENERATE} synthetic images using GAN ---")
noise_for_gan_training_data = tf.random.normal([NUM_GAN_IMAGES_TO_GENERATE, GAN_LATENT_DIM])
x_train_gan_generated_raw = gan_generator_loaded.predict(noise_for_gan_training_data, batch_size=BATCH_SIZE_CNN, verbose=1)

# Preprocess GAN images: rescale from [-1, 1] (tanh output) to [0, 1] for CNN input
x_train_gan_generated_cnn = (x_train_gan_generated_raw * 0.5 + 0.5)
print(f"Shape of generated GAN images for CNN: {x_train_gan_generated_cnn.shape}")

# Display a few generated images to verify
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train_gan_generated_cnn[i].reshape(IMG_WIDTH, IMG_HEIGHT), cmap='gray')
    plt.axis('off')
plt.suptitle('Sample GAN-generated Images for CNN Training')
plt.savefig(os.path.join(CNN_SAVE_DIR, 'sample_gan_generated_for_cnn.png'))
plt.close()


# --- 6. Assign Pseudo-Labels to GAN-generated Images ---
# Using the baseline CNN (trained on real data) to predict labels for GAN images
print("\n--- Assigning pseudo-labels to GAN-generated images using the baseline CNN ---")
predicted_probabilities_gan = cnn_baseline.predict(x_train_gan_generated_cnn, batch_size=BATCH_SIZE_CNN, verbose=1)
y_train_gan_pseudo_labels = np.argmax(predicted_probabilities_gan, axis=1)
y_train_gan_pseudo_labels_cnn = keras.utils.to_categorical(y_train_gan_pseudo_labels, NUM_CLASSES)

print(f"Shape of GAN pseudo-labels: {y_train_gan_pseudo_labels_cnn.shape}")

# --- 7. Train CNN on GAN-generated Data ---
print("\n--- Training CNN on GAN-generated data (with pseudo-labels) ---")
cnn_gan_trained = build_cnn_classifier() # Fresh instance of the CNN model
cnn_gan_trained.summary() # Verify it's the same architecture

history_gan_trained = cnn_gan_trained.fit(x_train_gan_generated_cnn, y_train_gan_pseudo_labels_cnn,
                                          epochs=EPOCHS_CNN,
                                          batch_size=BATCH_SIZE_CNN,
                                          validation_data=(x_test_real_cnn, y_test_real_cnn)) # Validate on REAL test data

gan_trained_loss, gan_trained_accuracy = cnn_gan_trained.evaluate(x_test_real_cnn, y_test_real_cnn, verbose=0)
print(f"\nCNN Test Accuracy (trained on GAN data): {gan_trained_accuracy*100:.2f}%")
cnn_gan_trained.save(os.path.join(CNN_SAVE_DIR, 'cnn_gan_trained_model.keras'))
print(f"CNN model (trained on GAN data) saved to {os.path.join(CNN_SAVE_DIR, 'cnn_gan_trained_model.keras')}")


# Plot GAN-trained CNN training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_gan_trained.history['accuracy'], label='Accuracy')
plt.plot(history_gan_trained.history['val_accuracy'], label='Val Accuracy (on Real Test Data)')
plt.title('GAN-Trained CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_gan_trained.history['loss'], label='Loss')
plt.plot(history_gan_trained.history['val_loss'], label='Val Loss (on Real Test Data)')
plt.title('GAN-Trained CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CNN_SAVE_DIR, 'cnn_gan_trained_training_history.png'))
print(f"GAN-trained CNN training history plot saved.")
plt.close()

# --- 8. Comparison ---
print("\n--- Performance Comparison ---")
print(f"Baseline CNN Test Accuracy (trained on REAL data): {baseline_accuracy*100:.2f}%")
print(f"CNN Test Accuracy (trained on GAN-generated data): {gan_trained_accuracy*100:.2f}%")

print("\nCNN classification script finished.")

