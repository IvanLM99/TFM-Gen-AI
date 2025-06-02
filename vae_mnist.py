# vae_mnist_pytorch.py
# Part 1: Variational Autoencoder (VAE) Training and Analysis (PyTorch)
# Part 2: VAE Latent Space Visualization (PyTorch)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- Parameters ---
IMG_WIDTH, IMG_HEIGHT = 28, 28
IMG_CHANNELS = 1
INPUT_SIZE = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS # 784
BATCH_SIZE = 128
EPOCHS = 30
LATENT_DIM = 2 # For 2D visualization
HIDDEN_DIM_ENCODER = 400 # Example hidden dimension for encoder MLP
HIDDEN_DIM_DECODER = 400 # Example hidden dimension for decoder MLP
LEARNING_RATE = 1e-3

# Visualization Params
SCATTER_SAMPLE_SIZE = 5000
N_GRID_POINTS_VIZ = 15 # For manifold plot (n in user's function)
VIZ_SCALE = 2.5 # Scale for manifold plot grid

# Create a directory for VAE outputs
VAE_SAVE_DIR = 'vae_outputs_pytorch'
if not os.path.exists(VAE_SAVE_DIR):
    os.makedirs(VAE_SAVE_DIR)

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Loading and Preprocessing ---
transform = transforms.Compose([
    transforms.ToTensor(), # Converts to [C, H, W] and scales to [0, 1]
    # transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std, optional
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- 2. VAE Model Definition ---
class VAE(nn.Module):
    def __init__(self, input_size, hidden_dim_encoder, hidden_dim_decoder, latent_dim):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Encoder (MLP for simplicity, can be CNN)
        self.fc_enc1 = nn.Linear(input_size, hidden_dim_encoder)
        self.fc_enc2 = nn.Linear(hidden_dim_encoder, hidden_dim_encoder // 2)
        self.fc_mean = nn.Linear(hidden_dim_encoder // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim_encoder // 2, latent_dim)

        # Decoder (MLP for simplicity)
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim_decoder // 2)
        self.fc_dec2 = nn.Linear(hidden_dim_decoder // 2, hidden_dim_decoder)
        self.fc_dec_out = nn.Linear(hidden_dim_decoder, input_size)

    def encode(self, x):
        x = x.view(-1, self.input_size) # Flatten image
        h = F.relu(self.fc_enc1(x))
        h = F.relu(self.fc_enc2(h))
        return self.fc_mean(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # Sample from N(0, I)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec1(z))
        h = F.relu(self.fc_dec2(h))
        # Sigmoid to output pixel values in [0, 1]
        return torch.sigmoid(self.fc_dec_out(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

# Loss function
def vae_loss_function(x_reconstructed, x, mu, log_var):
    # Reconstruction loss (Binary Cross Entropy for sigmoid output)
    # Sum over pixels, mean over batch
    recon_loss = F.binary_cross_entropy(x_reconstructed, x.view(-1, INPUT_SIZE), reduction='sum') / x.size(0)

    # KL divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Mean over batch
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return recon_loss + kld_loss, recon_loss, kld_loss


model = VAE(INPUT_SIZE, HIDDEN_DIM_ENCODER, HIDDEN_DIM_DECODER, LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)

# --- 3. Model Training ---
print("\n--- Training VAE (PyTorch) ---")
train_losses, train_recon_losses, train_kld_losses = [], [], []
val_losses, val_recon_losses, val_kld_losses = [], [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss, epoch_recon_loss, epoch_kld_loss = 0, 0, 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss, recon, kld = vae_loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_recon_loss += recon.item()
        epoch_kld_loss += kld.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_recon_loss = epoch_recon_loss / len(train_loader)
    avg_epoch_kld_loss = epoch_kld_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    train_recon_losses.append(avg_epoch_recon_loss)
    train_kld_losses.append(avg_epoch_kld_loss)

    # Validation
    model.eval()
    val_loss, val_recon, val_kld = 0, 0, 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            v_loss, v_recon, v_kld = vae_loss_function(recon_batch, data, mu, log_var)
            val_loss += v_loss.item()
            val_recon += v_recon.item()
            val_kld += v_kld.item()
            
    avg_val_loss = val_loss / len(test_loader)
    avg_val_recon_loss = val_recon / len(test_loader)
    avg_val_kld_loss = val_kld / len(test_loader)
    val_losses.append(avg_val_loss)
    val_recon_losses.append(avg_val_recon_loss)
    val_kld_losses.append(avg_val_kld_loss)

    print(f'Epoch {epoch}/{EPOCHS} \t'
          f'Train Loss: {avg_epoch_loss:.4f} (Recon: {avg_epoch_recon_loss:.4f}, KLD: {avg_epoch_kld_loss:.4f}) \t'
          f'Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon_loss:.4f}, KLD: {avg_val_kld_loss:.4f})')


# --- 4. Plot Loss per Epoch ---
plt.figure(figsize=(12, 8))
plt.plot(train_losses, label='Training Total Loss')
plt.plot(val_losses, label='Validation Total Loss')
plt.plot(train_recon_losses, label='Training Reconstruction Loss', linestyle='--')
plt.plot(val_recon_losses, label='Validation Reconstruction Loss', linestyle='--')
plt.plot(train_kld_losses, label='Training KL Loss', linestyle=':')
plt.plot(val_kld_losses, label='Validation KL Loss', linestyle=':')
plt.title('VAE Loss per Epoch (PyTorch)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(VAE_SAVE_DIR, 'vae_loss_plot_pytorch.png'))
print(f"\nLoss plot saved as {os.path.join(VAE_SAVE_DIR, 'vae_loss_plot_pytorch.png')}")
plt.close()

# --- 5. Save Model State Dictionary ---
torch.save(model.state_dict(), os.path.join(VAE_SAVE_DIR, 'vae_model_pytorch.pth'))
print(f"VAE model state_dict saved in '{VAE_SAVE_DIR}'.")

# --- Part 2: VAE Latent Space Visualization (Integrated PyTorch) ---
print("\n--- Visualizing VAE Latent Space (PyTorch) ---")
model.eval() # Ensure model is in evaluation mode

# Reconstruct some test images
data_iter = iter(test_loader)
sample_data, _ = next(data_iter)
sample_data = sample_data.to(device)
with torch.no_grad():
    reconstructed_sample, _, _ = model(sample_data)

reconstructed_sample_cpu = reconstructed_sample.cpu().view(-1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
original_sample_cpu = sample_data.cpu()

plt.figure(figsize=(10, 3))
n_display = min(10, BATCH_SIZE)
for i in range(n_display):
    # Original
    ax = plt.subplot(2, n_display, i + 1)
    plt.imshow(original_sample_cpu[i].permute(1, 2, 0).squeeze().numpy(), cmap='gray')
    plt.title("Orig")
    plt.axis('off')
    # Reconstructed
    ax = plt.subplot(2, n_display, i + 1 + n_display)
    plt.imshow(reconstructed_sample_cpu[i].permute(1, 2, 0).squeeze().numpy(), cmap='gray')
    plt.title("Recon")
    plt.axis('off')
plt.suptitle('Original vs. Reconstructed Images (VAE PyTorch)')
plt.savefig(os.path.join(VAE_SAVE_DIR, 'vae_reconstructions_pytorch.png'))
print(f"VAE reconstructions plot saved in '{VAE_SAVE_DIR}'.")
plt.close()


if LATENT_DIM == 2:
    # 1. Scatter plot of encoded test images
    print("Generating VAE 2D latent space scatter plot (PyTorch)...")
    all_z_means = []
    all_labels = []
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            if len(all_z_means) * BATCH_SIZE >= SCATTER_SAMPLE_SIZE:
                break
            data = data.to(device)
            mu, _ = model.encode(data)
            all_z_means.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_z_means = np.concatenate(all_z_means, axis=0)[:SCATTER_SAMPLE_SIZE]
    all_labels = np.concatenate(all_labels, axis=0)[:SCATTER_SAMPLE_SIZE]

    plt.figure(figsize=(12, 10))
    plt.scatter(all_z_means[:, 0], all_z_means[:, 1], c=all_labels, cmap='viridis', s=5)
    plt.colorbar(label='Digit Label')
    plt.xlabel("Latent Dimension 1 (z_mean)")
    plt.ylabel("Latent Dimension 2 (z_mean)")
    plt.title(f'VAE 2D Latent Space of MNIST Test Samples (PyTorch, N={len(all_z_means)})')
    plt.grid(True)
    plt.savefig(os.path.join(VAE_SAVE_DIR, 'vae_latent_space_scatter_pytorch.png'))
    print(f"VAE latent space scatter plot saved.")
    plt.close()

    # 2. Manifold plot (adapted from user's function)
    print("Generating VAE 2D latent space manifold (PyTorch)...")
    
    def plot_latent_space_pytorch(vae_model, scale=VIZ_SCALE, n=N_GRID_POINTS_VIZ, digit_size=IMG_HEIGHT, figsize=10):
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1] # y-axis reversed for image display

        vae_model.eval() # Ensure model is in eval mode
        with torch.no_grad():
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
                    x_decoded_flat = vae_model.decode(z_sample) # Decoder outputs flattened image
                    # Reshape to [C, H, W] then permute to [H, W, C] for imshow if C=1 or C=3
                    # For MNIST (C=1), reshape directly to [H,W]
                    digit = x_decoded_flat.cpu().reshape(digit_size, digit_size).numpy()
                    figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size] = digit
        
        plt.figure(figsize=(figsize, figsize))
        plt.title('VAE Latent Space Manifold (PyTorch)')
        
        # Ticks and labels (as per user's function)
        start_range = digit_size // 2
        end_range = n * digit_size + start_range - (digit_size if n > 1 else 0) # Adjust end_range for pixel_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        if len(pixel_range) > n : pixel_range = pixel_range[:n] # Ensure correct number of ticks

        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1) # grid_y is already reversed

        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y) # Use reversed grid_y for labels to match visual
        
        plt.xlabel("Latent Dim 1 (formerly mean, z[0])") # Adapted labels
        plt.ylabel("Latent Dim 2 (formerly var, z[1])")  # Adapted labels
        plt.imshow(figure, cmap="Greys_r") # User requested Greys_r
        plt.savefig(os.path.join(VAE_SAVE_DIR, f'vae_latent_manifold_pytorch_{LATENT_DIM}d.png'))
        print(f"VAE latent manifold (PyTorch) saved.")
        plt.close()

    plot_latent_space_pytorch(model)

else: # LATENT_DIM > 2
    print(f"VAE Latent Dimension is {LATENT_DIM}. For >2D scatter plot, dimensionality reduction (e.g., t-SNE) is typically used.")
    print("Generating random samples from prior N(0,1) instead of manifold for >2D.")
    with torch.no_grad():
        random_latent_vectors = torch.randn(N_GRID_POINTS_VIZ * N_GRID_POINTS_VIZ, LATENT_DIM).to(device)
        generated_images_flat = model.decode(random_latent_vectors)
        generated_images_from_prior = generated_images_flat.cpu().reshape(-1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    plt.figure(figsize=(10, 10))
    for i in range(min(N_GRID_POINTS_VIZ**2, 225)):
        ax = plt.subplot(N_GRID_POINTS_VIZ, N_GRID_POINTS_VIZ, i + 1)
        plt.imshow(generated_images_from_prior[i].permute(1,2,0).squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'VAE Generated Samples (from N(0,1), Latent Dim={LATENT_DIM}, PyTorch)')
    plt.savefig(os.path.join(VAE_SAVE_DIR, f'vae_generated_samples_prior_pytorch_{LATENT_DIM}d.png'))
    print(f"VAE generated samples from prior (PyTorch) saved.")
    plt.close()

print("\nVAE script (PyTorch with integrated visualization) finished.")
