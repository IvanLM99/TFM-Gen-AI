# Part 1: Generative Adversarial Network (GAN) Training and Analysis
# Part 2: GAN Latent Space Visualization (Slice Manifold)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import os

# Parameters
IMG_WIDTH, IMG_HEIGHT = 28, 28
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
BATCH_SIZE = 128
EPOCHS = 50 # GANs often require more epochs to stabilize
LATENT_DIM_GAN = 100 # Latent dimension for GAN's noise input
LEARNING_RATE_DISC = 0.0002
LEARNING_RATE_GAN = 0.0002
BETA_1 = 0.5

# Visualization Params
N_GRID_POINTS_VIZ_GAN = 15 # Number of points in each dimension for GAN manifold plot

# Create a directory to save generated images and plots
GAN_SAVE_DIR = 'gan_outputs' # Same directory for all GAN related outputs
if not os.path.exists(GAN_SAVE_DIR):
    os.makedirs(GAN_SAVE_DIR)

# 1. Data Loading and Preprocessing
(x_train_gan, _), (_, _) = mnist.load_data()

def preprocess_images_gan(images):
    images = images.astype('float32')
    images = (images - 127.5) / 127.5 # Normalize to [-1, 1]
    images = np.expand_dims(images, axis=-1)
    return images

x_train_gan_processed = preprocess_images_gan(x_train_gan)
train_dataset_gan = tf.data.Dataset.from_tensor_slices(x_train_gan_processed).shuffle(len(x_train_gan_processed)).batch(BATCH_SIZE)

# 2. GAN Model Definition

# --- Generator ---
def build_generator():
    model = models.Sequential(name='generator')
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(LATENT_DIM_GAN,)))
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

generator = build_generator()
generator.summary()

# --- Discriminator ---
def build_discriminator():
    model = models.Sequential(name='discriminator')
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=INPUT_SHAPE))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

discriminator = build_discriminator()
discriminator.summary()

# --- Loss Functions and Optimizers ---
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_GAN, beta_1=BETA_1)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_DISC, beta_1=BETA_1)

@tf.function
def train_step(images):
    noise = tf.random.normal([tf.shape(images)[0], LATENT_DIM_GAN]) # Use dynamic batch size from images
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input, save_dir_gan):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(6, 6))
    for i in range(predictions.shape[0]):
        plt.subplot(6, 6, i + 1)
        img_to_show = (predictions[i, :, :, 0] * 127.5 + 127.5) / 255.0 # Rescale for display
        plt.imshow(img_to_show, cmap='gray')
        plt.axis('off')
    plt.suptitle(f'GAN Generated Images - Epoch {epoch+1}')
    plt.savefig(os.path.join(save_dir_gan, f'image_at_epoch_{epoch+1:04d}.png'))
    plt.close(fig)

num_examples_to_generate = 36
seed = tf.random.normal([num_examples_to_generate, LATENT_DIM_GAN])

# 3. GAN Training Loop
print("\n--- Training GAN ---")
gen_loss_history = []
disc_loss_history = []

for epoch in range(EPOCHS):
    epoch_gen_loss_avg = tf.keras.metrics.Mean()
    epoch_disc_loss_avg = tf.keras.metrics.Mean()
    for image_batch in train_dataset_gan:
        g_loss, d_loss = train_step(image_batch)
        epoch_gen_loss_avg.update_state(g_loss)
        epoch_disc_loss_avg.update_state(d_loss)
    current_gen_loss = epoch_gen_loss_avg.result().numpy()
    current_disc_loss = epoch_disc_loss_avg.result().numpy()
    gen_loss_history.append(current_gen_loss)
    disc_loss_history.append(current_disc_loss)
    print(f'Epoch {epoch+1}/{EPOCHS} - Gen Loss: {current_gen_loss:.4f}, Disc Loss: {current_disc_loss:.4f}')
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        generate_and_save_images(generator, epoch, seed, GAN_SAVE_DIR)

print(f"\nGenerated sample images during training saved in '{GAN_SAVE_DIR}'.")

# 4. Plot Loss per Epoch
plt.figure(figsize=(10, 5))
plt.plot(gen_loss_history, label='Generator Loss')
plt.plot(disc_loss_history, label='Discriminator Loss')
plt.title('GAN Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GAN_SAVE_DIR, 'gan_loss_plot.png'))
print(f"Loss plot saved to {os.path.join(GAN_SAVE_DIR, 'gan_loss_plot.png')}")
plt.close()

# 5. Save Connection Weights
generator.save_weights(os.path.join(GAN_SAVE_DIR, 'gan_generator_weights.weights.h5'))
discriminator.save_weights(os.path.join(GAN_SAVE_DIR, 'gan_discriminator_weights.weights.h5'))
print(f"GAN generator and discriminator weights saved in '{GAN_SAVE_DIR}'.")


# --- Part 2: GAN Latent Space Visualization (Integrated Slice Manifold) ---
print("\n--- Visualizing GAN Latent Space (2D Slice of Noise Input) ---")
dim1_to_vary = 0
dim2_to_vary = 1
base_noise = np.random.normal(0, 1, size=(1, LATENT_DIM_GAN)).astype('float32')
noise_range_gan = np.linspace(-2, 2, N_GRID_POINTS_VIZ_GAN)

gan_manifold_figure = np.zeros((IMG_WIDTH * N_GRID_POINTS_VIZ_GAN, IMG_HEIGHT * N_GRID_POINTS_VIZ_GAN, IMG_CHANNELS))

for i, val_dim2 in enumerate(noise_range_gan[::-1]):
    for j, val_dim1 in enumerate(noise_range_gan):
        current_noise = base_noise.copy()
        current_noise[0, dim1_to_vary] = val_dim1
        current_noise[0, dim2_to_vary] = val_dim2
        generated_image_gan = generator.predict(current_noise, verbose=0)
        img_to_show = (generated_image_gan[0] * 0.5 + 0.5) # Rescale [-1,1] to [0,1]
        gan_manifold_figure[
            i * IMG_WIDTH : (i + 1) * IMG_WIDTH,
            j * IMG_HEIGHT : (j + 1) * IMG_HEIGHT,
            :
        ] = img_to_show.reshape(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

plt.figure(figsize=(10, 10))
if IMG_CHANNELS == 1:
    plt.imshow(gan_manifold_figure[:,:,0], cmap='gray')
else:
    plt.imshow(gan_manifold_figure)
plt.xlabel(f"Noise Dimension {dim1_to_vary} Value")
plt.ylabel(f"Noise Dimension {dim2_to_vary} Value")
plt.title(f'GAN Latent Space Slice (Varying Dims {dim1_to_vary} & {dim2_to_vary})')
plt.xticks(np.arange(0, N_GRID_POINTS_VIZ_GAN*IMG_HEIGHT, IMG_HEIGHT)[::3], [f"{val:.1f}" for val in noise_range_gan[::3]])
plt.yticks(np.arange(0, N_GRID_POINTS_VIZ_GAN*IMG_WIDTH, IMG_WIDTH)[::3], [f"{val:.1f}" for val in noise_range_gan[::-1][::3]])
plt.savefig(os.path.join(GAN_SAVE_DIR, f'gan_latent_slice_manifold_dims_{dim1_to_vary}_{dim2_to_vary}.png'))
print(f"GAN latent slice manifold saved to {os.path.join(GAN_SAVE_DIR, f'gan_latent_slice_manifold_dims_{dim1_to_vary}_{dim2_to_vary}.png')}")
plt.close()

print("\nGAN script (with integrated visualization) finished.")
