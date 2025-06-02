# gan_mnist_pytorch.py
# Part 1: Generative Adversarial Network (GAN) Training and Analysis (PyTorch)
# Part 2: GAN Latent Space Visualization (PyTorch)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import os

# --- Parameters ---
IMG_WIDTH, IMG_HEIGHT = 28, 28
IMG_CHANNELS = 1
INPUT_SHAPE_DISC = (IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH) # PyTorch uses (C, H, W)
BATCH_SIZE = 128
EPOCHS = 50 # GANs often need more epochs
LATENT_DIM_GAN = 100 # Size of the noise vector (nz)
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA1 = 0.5 # Adam optimizer beta1
BETA2 = 0.999 # Adam optimizer beta2

# Visualization Params
N_GRID_POINTS_VIZ_GAN = 20 # For manifold plot (e.g., 20x20 grid = 400 images)
VIZ_FIXED_NOISE_SAMPLES = N_GRID_POINTS_VIZ_GAN * N_GRID_POINTS_VIZ_GAN

# Create a directory for GAN outputs
GAN_SAVE_DIR = 'gan_outputs_pytorch'
if not os.path.exists(GAN_SAVE_DIR):
    os.makedirs(GAN_SAVE_DIR)
if not os.path.exists(os.path.join(GAN_SAVE_DIR, 'training_progress')):
    os.makedirs(os.path.join(GAN_SAVE_DIR, 'training_progress'))


# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Loading and Preprocessing ---
transform_gan = transforms.Compose([
    transforms.ToTensor(), # Converts to [C, H, W] and scales to [0, 1]
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] for tanh output
])

train_dataset_gan = datasets.MNIST('./data', train=True, download=True, transform=transform_gan)
train_loader_gan = DataLoader(train_dataset_gan, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# --- 2. GAN Model Definition (DCGAN-like architecture) ---

# Generator
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        # nz: size of latent vector z
        # ngf: size of feature maps in generator
        # nc: number of channels in the output image
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False), # (nz,1,1) -> (ngf*4, 4, 4)
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False), # (ngf*4,4,4) -> (ngf*2, 7,7) with 3,2,1 for 28x28
                                                                    # For 28x28 output, need to adjust kernel/stride/padding
                                                                    # Let's adjust to get to 7x7 first
            # For MNIST 28x28, a common DCGAN structure starts from a dense layer or reshapes noise
            # Simpler approach for MNIST:
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0, bias=False), # (nz,1,1) -> (ngf*4, 7, 7)
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 7 x 7
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # (ngf*4,7,7) -> (ngf*2, 14, 14)
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 14 x 14
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False), # (ngf*2,14,14) -> (nc, 28, 28)
            nn.Tanh() # Output range [-1, 1]
            # state size. (nc) x 28 x 28
        )

    def forward(self, input_noise):
        # Input noise shape: (batch_size, nz, 1, 1)
        return self.main(input_noise)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # nc: number of channels in the input image
        # ndf: size of feature maps in discriminator
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # (nc,28,28) -> (ndf, 14,14)
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # (ndf,14,14) -> (ndf*2, 7,7)
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False), # (ndf*2,7,7) -> (1,1,1)
            # No sigmoid here, using BCEWithLogitsLoss
        )

    def forward(self, input_image):
        # Output is a single scalar logit (before sigmoid)
        return self.main(input_image).view(-1, 1) # Flatten to (batch_size, 1)


# Initialize models
# ngf: relates to the depth of feature maps propagated through the generator
# ndf: sets the depth of feature maps processed by the discriminator
ngf = 64 
ndf = 64
netG = Generator(LATENT_DIM_GAN, ngf, IMG_CHANNELS).to(device)
netD = Discriminator(IMG_CHANNELS, ndf).to(device)

# Custom weights initialization (from DCGAN paper)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

print(netG)
print(netD)

# --- Loss Function and Optimizers ---
criterion = nn.BCEWithLogitsLoss() # Combines Sigmoid layer and BCELoss

# Fixed noise for consistent visualization of G's progress
fixed_noise_for_progress = torch.randn(64, LATENT_DIM_GAN, 1, 1, device=device) # 64 samples for 8x8 grid

# Real and fake labels
real_label_val = 1.
fake_label_val = 0.

optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))

# --- 3. GAN Training Loop ---
print("\n--- Training GAN (PyTorch) ---")
G_losses = []
D_losses = []
img_list = [] # To store generated images for animation or display
iters = 0

for epoch in range(EPOCHS):
    for i, data in enumerate(train_loader_gan, 0):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        
        # Real batch
        label_real = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)
        output_real = netD(real_cpu).view(-1) # Flatten output
        errD_real = criterion(output_real, label_real)
        errD_real.backward()
        D_x = output_real.mean().item() # Average D(x) over the batch

        # Fake batch
        noise = torch.randn(b_size, LATENT_DIM_GAN, 1, 1, device=device)
        fake = netG(noise)
        label_fake = torch.full((b_size,), fake_label_val, dtype=torch.float, device=device)
        output_fake = netD(fake.detach()).view(-1) # Use .detach() to avoid backprop through G
        errD_fake = criterion(output_fake, label_fake)
        errD_fake.backward()
        D_G_z1 = output_fake.mean().item() # Average D(G(z)) before D update
        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        # For G, we want D to output "real" for fake images
        label_g = torch.full((b_size,), real_label_val, dtype=torch.float, device=device) 
        output_g = netD(fake).view(-1) # Re-run fake batch through D (now D is updated)
        errG = criterion(output_g, label_g)
        errG.backward()
        D_G_z2 = output_g.mean().item() # Average D(G(z)) after G update
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 100 == 0) or ((epoch == EPOCHS-1) and (i == len(train_loader_gan)-1)):
            print(f'[{epoch+1}/{EPOCHS}][{i}/{len(train_loader_gan)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == EPOCHS-1) and (i == len(train_loader_gan)-1)):
            with torch.no_grad():
                fake_progress = netG(fixed_noise_for_progress).detach().cpu()
            img_grid = make_grid(fake_progress, padding=2, normalize=True)
            save_image(img_grid, os.path.join(GAN_SAVE_DIR, 'training_progress', f'progress_epoch_{epoch+1}_iter_{iters}.png'))
            if iters == 0 : img_list.append(img_grid) # Store first one for later display if needed

        iters += 1

print(f"\nGenerated sample images during training saved in '{os.path.join(GAN_SAVE_DIR, 'training_progress')}'.")

# --- 4. Plot Loss per Epoch ---
# Note: G_losses and D_losses are per iteration. For per-epoch plot, average them.
# For simplicity, plotting per iteration here.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training (PyTorch)")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GAN_SAVE_DIR, 'gan_loss_plot_pytorch.png'))
print(f"Loss plot saved to {os.path.join(GAN_SAVE_DIR, 'gan_loss_plot_pytorch.png')}")
plt.close()

# --- 5. Save Model State Dictionaries ---
torch.save(netG.state_dict(), os.path.join(GAN_SAVE_DIR, 'gan_generator_pytorch.pth'))
torch.save(netD.state_dict(), os.path.join(GAN_SAVE_DIR, 'gan_discriminator_pytorch.pth'))
print(f"GAN generator and discriminator state_dicts saved in '{GAN_SAVE_DIR}'.")

# --- Part 2: GAN Latent Space Visualization (Integrated PyTorch Slice Manifold) ---
print("\n--- Visualizing GAN Latent Space (2D Slice of Noise Input - PyTorch) ---")
netG.eval() # Set generator to evaluation mode

# Your visualization snippet adapted:
# It seems your snippet intended to use a 2D grid of points (x from data_loader_myspace)
# to modify the first two dimensions of a 100D noise vector.
# Let's create that 2D grid of points manually.

grid_size = N_GRID_POINTS_VIZ_GAN # e.g., 20 for a 20x20 grid
total_images_manifold = grid_size * grid_size

# Create a 2D grid of values for the first two latent dimensions
# These will range from -scale to +scale, e.g., -1.5 to 1.5
latent_val_range = torch.linspace(-1.5, 1.5, grid_size, device=device)
grid_x, grid_y = torch.meshgrid(latent_val_range, latent_val_range, indexing='ij') # Create grid

# Prepare the noise vectors
# Base noise: random for dimensions 2 to LATENT_DIM_GAN-1, zeros for dim 0 and 1 initially
base_noise_manifold = torch.randn(total_images_manifold, LATENT_DIM_GAN, 1, 1, device=device)

# Populate the first two dimensions from the grid
idx = 0
for r in range(grid_size):
    for c in range(grid_size):
        base_noise_manifold[idx, 0, 0, 0] = grid_x[r, c] # Vary 1st dim
        base_noise_manifold[idx, 1, 0, 0] = grid_y[r, c] # Vary 2nd dim
        idx += 1
        
with torch.no_grad():
    fake_manifold_images = netG(base_noise_manifold).detach().cpu()

# Plotting the manifold
fig_manifold = plt.figure(figsize=(12, 12)) # Adjusted for better display
plt.title(f"GAN Latent Space Slice (Varying first 2 of {LATENT_DIM_GAN} dims)")
img_grid_manifold = make_grid(fake_manifold_images, nrow=grid_size, padding=2, normalize=True)
plt.imshow(np.transpose(img_grid_manifold,(1,2,0))) # Transpose (C,H,W) to (H,W,C) for imshow
plt.axis('off')

# Add ticks similar to VAE manifold if desired, mapping pixel positions to latent_val_range
tick_positions = np.linspace(0, img_grid_manifold.shape[2] - (img_grid_manifold.shape[2]/grid_size), grid_size) + (img_grid_manifold.shape[2]/grid_size)/2
tick_labels = [f"{val:.1f}" for val in latent_val_range.cpu().numpy()]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels[::-1]) # Y-axis often inverted in image display
plt.xlabel("Latent Dim 0 Value")
plt.ylabel("Latent Dim 1 Value")

plt.savefig(os.path.join(GAN_SAVE_DIR, f'gan_latent_slice_manifold_pytorch.png'))
print(f"GAN latent slice manifold saved to {os.path.join(GAN_SAVE_DIR, f'gan_latent_slice_manifold_pytorch.png')}")
plt.close(fig_manifold)


print("\nGAN script (PyTorch with integrated visualization) finished.")
