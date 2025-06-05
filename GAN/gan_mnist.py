import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
matplotlib.style.use('ggplot')

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True) # For storing dataset

# Learning parameters
batch_size = 256 
epochs = 100     
sample_size = 64 # Fixed sample size for generated images during training
nz = 100         # Latent vector size 
k = 1            # Number of steps to apply to the discriminator per generator step
lr = 0.0002      # Learning rate
beta1 = 0.5      # Beta1 for Adam optimizer

# Setup device (auto-detects CUDA if available, compatible with ROCm via PyTorch's CUDA interface)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)), # Normalize to [-1, 1] for Tanh activation
])

# MNIST dataset
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Generator Network
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 784), # 28*28 for MNIST
            nn.Tanh() # Output images in [-1, 1]
        )
        print("Generator initialized")

    def forward(self, x):
        # x is (batch_size, nz)
        output = self.main(x)
        return output.view(-1, 1, 28, 28) # Reshape to image format

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784 # 28*28 for MNIST
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid() 
        )
        print("Discriminator initialized")

    def forward(self, x):
        # x is (batch_size, 1, 28, 28)
        x_flat = x.view(-1, self.n_input) # Flatten image
        return self.main(x_flat)

# Initialize models
netG = Generator(nz).to(device)
netD = Discriminator().to(device)

print('\n--- GENERATOR ---')
print(netG)

print('\n--- DISCRIMINATOR ---')
print(netD)

# Optimizers
optim_G = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optim_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Fixed noise for consistent visualization during training
fixed_noise = torch.randn(sample_size, nz, device=device)

# Lists to store losses and images
losses_G = []
losses_D = []
images = []

# Helper functions for labels
def label_real(size):
    return torch.ones(size, 1).to(device)

def label_fake(size):
    return torch.zeros(size, 1).to(device)

# Training loop
print("\n--- Starting Training Loop ---")
for epoch in range(epochs):
    epoch_loss_g = 0.0
    epoch_loss_d = 0.0
    
    # Use tqdm for a progress bar
    pbar = tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size))
    for i, data in pbar:
        real_images, _ = data
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)

        # --- Train Discriminator ---
        netD.zero_grad()
        
        # Real images
        output_real = netD(real_images)
        loss_d_real = criterion(output_real, label_real(current_batch_size))
        
        # Fake images
        noise_vec = torch.randn(current_batch_size, nz, device=device)
        fake_images = netG(noise_vec).detach() # Detach to avoid backprop through G
        output_fake = netD(fake_images)
        loss_d_fake = criterion(output_fake, label_fake(current_batch_size))
        
        # Total discriminator loss
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optim_D.step()
        
        epoch_loss_d += loss_d.item()

        # --- Train Generator ---
        # run discriminator k steps before one generator step, here k=1)
        netG.zero_grad()
        
        # Generate fresh fake images (as D has been updated)
        noise_vec_g = torch.randn(current_batch_size, nz, device=device)
        fake_images_g = netG(noise_vec_g)
        output_g = netD(fake_images_g) # Try to fool discriminator
        loss_g = criterion(output_g, label_real(current_batch_size)) # Generator wants D to think they are real
        
        loss_g.backward()
        optim_G.step()
        
        epoch_loss_g += loss_g.item()

        pbar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

    # Average losses for the epoch
    avg_loss_g = epoch_loss_g / len(train_loader)
    avg_loss_d = epoch_loss_d / len(train_loader)
    losses_G.append(avg_loss_g)
    losses_D.append(avg_loss_d)

    print(f"Epoch {epoch+1}/{epochs} completed. Avg Generator Loss: {avg_loss_g:.4f}, Avg Discriminator Loss: {avg_loss_d:.4f}")

    # Generate and save sample images using fixed noise
    with torch.no_grad():
        generated_sample_img = netG(fixed_noise).cpu().detach()
        img_grid = make_grid(generated_sample_img, padding=2, normalize=True)
        images.append(img_grid) # For GIF

print('\n--- DONE TRAINING ---')

# Save models
torch.save(netG.state_dict(), 'outputs/generator_mnist.pth')
torch.save(netD.state_dict(), 'outputs/discriminator_mnist.pth')
print("Models saved to 'outputs/' directory.")

# Save the generated images as GIF
if imageio:
    # Convert tensors to numpy arrays suitable for imageio
    imgs_for_gif = [np.array(transforms.ToPILImage()(img)) for img in images]
    imageio.mimsave('outputs/generator_images_mnist.gif', imgs_for_gif, fps=5)
    print("Generated images saved as GIF in 'outputs/' directory.")
else:
    print("imageio library not found, skipping GIF creation.")


# Plot and save the generator and discriminator loss
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(losses_G, label="Generator Loss")
plt.plot(losses_D, label="Discriminator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('outputs/loss_plot_mnist.png')
print("Loss plot saved as 'outputs/loss_plot_mnist.png'.")

# Visualize the latent space 
print("\n--- Visualizing latent space ---")
netG.eval() # Set generator to evaluation mode

vis_grid_size = 20 # Create a 20x20 grid of images
num_vis_images = vis_grid_size * vis_grid_size
vis_batch_size = 100 # Batch size for generating visualization images

# Create 2D points for latent space visualization
x_vals = np.linspace(-3, 3, vis_grid_size)  # Range for the first latent dimension
y_vals = np.linspace(-3, 3, vis_grid_size)  # Range for the second latent dimension

myspace_points = []
for y_val in y_vals:
    for x_val in x_vals:
        myspace_points.append([x_val, y_val])
myspace = torch.tensor(myspace_points, dtype=torch.float32).to(device) # Shape: (400, 2)

generated_imgs_latent_space = []

with torch.no_grad():
    for i in range(0, len(myspace), vis_batch_size):
        current_points_batch = myspace[i:i+vis_batch_size]
        current_actual_batch_size = current_points_batch.size(0)
        
        # Create latent vectors: z has shape (current_actual_batch_size, nz)
        # Initialize with zeros for dimensions not controlled by myspace
        # This ensures that changes are only due to the first two dimensions
        z = torch.zeros(current_actual_batch_size, nz, device=device)
        
        # Assign the 2D points from myspace to the first two dimensions of z
        z[:, :2] = current_points_batch
                
        batch_generated_imgs = netG(z)
        generated_imgs_latent_space.append(batch_generated_imgs.cpu())

# Concatenate all generated image batches
generated_imgs_latent_space = torch.cat(generated_imgs_latent_space)

# Plot the grid of images from the latent space
fig_latent = plt.figure(figsize=(15, 15))
plt.title(f"Latent Space Visualization", fontsize=16)
plt.subplots_adjust(wspace=0, hspace=0) # Reduce spacing between subplots

for i, img_tensor in enumerate(generated_imgs_latent_space):
    if i >= num_vis_images:
        break
    ax = fig_latent.add_subplot(vis_grid_size, vis_grid_size, i + 1)
    # img_tensor is (1, 28, 28), imshow expects (28,28) or (28,28,1) for grayscale
    ax.imshow(img_tensor.squeeze().numpy(), cmap='gray')
    ax.set_axis_off()

plt.savefig('outputs/latent_space_visualization_mnist.png')
print("Latent space visualization saved as 'outputs/latent_space_visualization_mnist.png'.")

print("\n--- Script finished ---")

