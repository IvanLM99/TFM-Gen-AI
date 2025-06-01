# Part 1: Variational Autoencoder (VAE) Training and Analysis
# Part 2: VAE Latent Space Visualization

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.datasets import mnist
import os

# Parameters
IMG_WIDTH, IMG_HEIGHT = 28, 28
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
BATCH_SIZE = 128
EPOCHS = 30 # Adjust as needed for better results
LATENT_DIM = 2 # For easier 2D visualization, as in output.png
# LATENT_DIM = 16 # Or a higher dimension

# Visualization Params
SCATTER_SAMPLE_SIZE = 5000 # Number of test samples for VAE scatter plot
N_GRID_POINTS_VIZ = 15 # Number of points in each dimension for manifold plots

# Create a directory for VAE outputs
VAE_SAVE_DIR = 'vae_outputs'
if not os.path.exists(VAE_SAVE_DIR):
    os.makedirs(VAE_SAVE_DIR)

# 1. Data Loading and Preprocessing
(x_train, y_train_labels_full), (x_test, y_test_labels_full) = mnist.load_data()

def preprocess_images(images):
    images = images.astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1) # Add channel dimension
    return images

x_train_processed = preprocess_images(x_train)
x_test_processed = preprocess_images(x_test)

print(f"x_train_processed shape: {x_train_processed.shape}")
print(f"x_test_processed shape: {x_test_processed.shape}")

# 2. VAE Model Definition

# --- Encoder ---
encoder_inputs = keras.Input(shape=INPUT_SHAPE, name='encoder_input')
x_enc = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same', name='conv1')(encoder_inputs)
x_enc = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same', name='conv2')(x_enc)
x_enc = layers.Flatten(name='flatten')(x_enc)
x_enc = layers.Dense(16, activation='relu', name='dense_encoder')(x_enc)

z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x_enc)
z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x_enc)

def sampling(args):
    z_mean_arg, z_log_var_arg = args
    batch = K.shape(z_mean_arg)[0]
    dim = K.int_shape(z_mean_arg)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean_arg + K.exp(0.5 * z_log_var_arg) * epsilon

z = layers.Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])

encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# --- Decoder ---
latent_inputs = keras.Input(shape=(LATENT_DIM,), name='z_sampling')
x_dec = layers.Dense(7 * 7 * 64, activation='relu', name='dense_decoder_input')(latent_inputs)
x_dec = layers.Reshape((7, 7, 64), name='reshape_decoder')(x_dec)
x_dec = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same', name='deconv1')(x_dec)
x_dec = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same', name='deconv2')(x_dec)
decoder_outputs = layers.Conv2DTranspose(IMG_CHANNELS, 3, activation='sigmoid', padding='same', name='decoder_output')(x_dec)

decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')
decoder.summary()

# --- VAE Model (combining encoder and decoder) ---
class VAE(keras.Model):
    def __init__(self, enc, dec, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder_model = enc # Renamed to avoid conflict with keras.Model.encoder
        self.decoder_model = dec # Renamed to avoid conflict with keras.Model.decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x_input, _ = data # Assuming data is (x, y), we only need x for VAE
        with tf.GradientTape() as tape:
            z_mean_val, z_log_var_val, z_val = self.encoder_model(x_input)
            reconstruction = self.decoder_model(z_val)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x_input, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var_val - tf.square(z_mean_val) - tf.exp(z_log_var_val))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        z_mean_val, z_log_var_val, z_val = self.encoder_model(inputs)
        reconstruction = self.decoder_model(z_val)
        return reconstruction

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# 3. Model Training
print("\n--- Training VAE ---")
history = vae.fit(x_train_processed, x_train_processed, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test_processed, x_test_processed))

# 4. Plot Loss per Epoch
plt.figure(figsize=(12, 6)) # Adjusted size
plt.plot(history.history['loss'], label='Training Total Loss')
plt.plot(history.history['val_loss'], label='Validation Total Loss')
plt.plot(history.history['reconstruction_loss'], label='Training Reconstruction Loss')
# Keras might not add val_ prefix to custom metrics in history by default for custom train_step
# Check history.history.keys() if these are missing
if 'val_reconstruction_loss' in history.history:
    plt.plot(history.history['val_reconstruction_loss'], label='Validation Reconstruction Loss')
if 'val_kl_loss' in history.history:
    plt.plot(history.history['val_kl_loss'], label='Validation KL Loss')
plt.plot(history.history['kl_loss'], label='Training KL Loss')

plt.title('VAE Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(VAE_SAVE_DIR, 'vae_loss_plot.png'))
print(f"\nLoss plot saved as {os.path.join(VAE_SAVE_DIR, 'vae_loss_plot.png')}")
plt.close()

# 5. Save Connection Weights
encoder.save_weights(os.path.join(VAE_SAVE_DIR, 'vae_encoder_weights.weights.h5'))
decoder.save_weights(os.path.join(VAE_SAVE_DIR, 'vae_decoder_weights.weights.h5'))
print(f"\nVAE encoder and decoder weights saved in '{VAE_SAVE_DIR}'.")

# --- Part 2: VAE Latent Space Visualization (Integrated) ---
print("\n--- Visualizing VAE Latent Space ---")

# Reconstruct some test images (example of using the trained VAE)
n_display = 10
reconstructed_images = vae.predict(x_test_processed[:n_display], verbose=0)

plt.figure(figsize=(10, 3))
for i in range(n_display):
    # Original
    ax = plt.subplot(2, n_display, i + 1)
    plt.imshow(x_test_processed[i].reshape(IMG_WIDTH, IMG_HEIGHT), cmap='gray')
    plt.title("Orig")
    plt.axis('off')
    # Reconstructed
    ax = plt.subplot(2, n_display, i + 1 + n_display)
    plt.imshow(reconstructed_images[i].reshape(IMG_WIDTH, IMG_HEIGHT), cmap='gray')
    plt.title("Recon")
    plt.axis('off')
plt.suptitle('Original vs. Reconstructed Images (VAE)')
plt.savefig(os.path.join(VAE_SAVE_DIR, 'vae_reconstructions.png'))
print(f"VAE reconstructions plot saved in '{VAE_SAVE_DIR}'.")
plt.close()


if LATENT_DIM == 2:
    # 1. Scatter plot of encoded test images
    print("Generating VAE 2D latent space scatter plot...")
    z_mean_encoded, _, _ = encoder.predict(x_test_processed[:SCATTER_SAMPLE_SIZE], verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean_encoded[:, 0], z_mean_encoded[:, 1], c=y_test_labels_full[:SCATTER_SAMPLE_SIZE], cmap='viridis', s=5)
    plt.colorbar(label='Digit Label')
    plt.xlabel("Latent Dimension 1 (z_mean)")
    plt.ylabel("Latent Dimension 2 (z_mean)")
    plt.title(f'VAE 2D Latent Space of MNIST Test Samples (N={SCATTER_SAMPLE_SIZE})')
    plt.grid(True)
    plt.savefig(os.path.join(VAE_SAVE_DIR, 'vae_latent_space_scatter.png'))
    print(f"VAE latent space scatter plot saved to {os.path.join(VAE_SAVE_DIR, 'vae_latent_space_scatter.png')}")
    plt.close()

    # 2. Manifold plot (grid of decoded images)
    print("Generating VAE 2D latent space manifold...")
    figure_manifold = np.zeros((IMG_WIDTH * N_GRID_POINTS_VIZ, IMG_HEIGHT * N_GRID_POINTS_VIZ, IMG_CHANNELS))
    
    z1_min, z1_max = np.min(z_mean_encoded[:,0]) if 'z_mean_encoded' in locals() else -2.5, np.max(z_mean_encoded[:,0]) if 'z_mean_encoded' in locals() else 2.5
    z2_min, z2_max = np.min(z_mean_encoded[:,1]) if 'z_mean_encoded' in locals() else -2.5, np.max(z_mean_encoded[:,1]) if 'z_mean_encoded' in locals() else 2.5
    
    grid_x = np.linspace(z1_min, z1_max, N_GRID_POINTS_VIZ)
    grid_y = np.linspace(z2_min, z2_max, N_GRID_POINTS_VIZ)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
            figure_manifold[
                i * IMG_WIDTH : (i + 1) * IMG_WIDTH,
                j * IMG_HEIGHT : (j + 1) * IMG_HEIGHT,
                :
            ] = digit

    plt.figure(figsize=(10, 10))
    if IMG_CHANNELS == 1:
        plt.imshow(figure_manifold[:,:,0], cmap='gray')
    else:
        plt.imshow(figure_manifold)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title(f'VAE Latent Space Manifold ({LATENT_DIM}D)')
    plt.xticks(np.arange(0, N_GRID_POINTS_VIZ*IMG_HEIGHT, IMG_HEIGHT)[::3], [f"{val:.1f}" for val in grid_x[::3]])
    plt.yticks(np.arange(0, N_GRID_POINTS_VIZ*IMG_WIDTH, IMG_WIDTH)[::3], [f"{val:.1f}" for val in grid_y[::-1][::3]])
    plt.savefig(os.path.join(VAE_SAVE_DIR, f'vae_latent_manifold_{LATENT_DIM}d.png'))
    print(f"VAE latent manifold saved to {os.path.join(VAE_SAVE_DIR, f'vae_latent_manifold_{LATENT_DIM}d.png')}")
    plt.close()
else:
    print(f"VAE Latent Dimension is {LATENT_DIM}. For >2D scatter plot, dimensionality reduction (e.g., t-SNE) is typically used.")
    print("Generating random samples from prior N(0,1) instead of manifold for >2D.")
    random_latent_vectors = np.random.normal(size=(N_GRID_POINTS_VIZ * N_GRID_POINTS_VIZ, LATENT_DIM))
    generated_images_from_prior = decoder.predict(random_latent_vectors, verbose=0)
    
    plt.figure(figsize=(10, 10))
    for i in range(min(N_GRID_POINTS_VIZ**2, 225)): # Display up to 15x15 grid
        ax = plt.subplot(N_GRID_POINTS_VIZ, N_GRID_POINTS_VIZ, i + 1)
        plt.imshow(generated_images_from_prior[i].reshape(IMG_WIDTH, IMG_HEIGHT), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'VAE Generated Samples (from N(0,1), Latent Dim={LATENT_DIM})')
    plt.savefig(os.path.join(VAE_SAVE_DIR, f'vae_generated_samples_prior_{LATENT_DIM}d.png'))
    print(f"VAE generated samples from prior saved in '{VAE_SAVE_DIR}'.")
    plt.close()

print("\nVAE script (with integrated visualization) finished.")
