# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid # <-- make_grid is essential for the fix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars in console
import os # For path expansion

# Create output directory if it doesn't exist
os.makedirs("outputs_gan", exist_ok=True)
os.makedirs("outputs_gan/weights", exist_ok=True)

# --- Utility Functions ---
def weights_init(m):
    """
    Custom weights initialization called on netG and netD.
    From the DCGAN paper, all model weights shall be randomly initialized 
    from a Normal distribution with mean=0, stdev=0.02.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- GAN Architecture ---
class Generator(nn.Module):
    """
    Generator Network (G)
    Takes a latent vector (z) as input and generates an image.
    Uses a series of ConvTranspose2d layers to upsample the vector to a 28x28 image.
    """
    def __init__(self, latent_dim=100, num_channels=1, gen_features=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, gen_features * 4, kernel_size=3, stride=2, padding=0, bias=False), # 1x1 -> 3x3
            nn.BatchNorm2d(gen_features * 4),
            nn.ReLU(True),
            # state size. (gen_features*4) x 3 x 3
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, kernel_size=4, stride=2, padding=1, bias=False), # 3x3 -> 6x6
            nn.BatchNorm2d(gen_features * 2),
            nn.ReLU(True),
            # state size. (gen_features*2) x 6 x 6
            nn.ConvTranspose2d(gen_features * 2, gen_features, kernel_size=4, stride=2, padding=1, bias=False), # 6x6 -> 12x12
            nn.BatchNorm2d(gen_features),
            nn.ReLU(True),
            # state size. (gen_features) x 12 x 12
            nn.ConvTranspose2d(gen_features, num_channels, kernel_size=3, stride=3, padding=4, bias=False), # 12x12 -> 28x28
            nn.Tanh() # Tanh output to match normalized input image range [-1, 1]
            # state size. (num_channels) x 28 x 28
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """
    Discriminator Network (D)
    Takes an image as input and outputs a single value representing the
    probability that the image is real (as opposed to fake).
    """
    def __init__(self, num_channels=1, disc_features=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (num_channels) x 28 x 28
            nn.Conv2d(num_channels, disc_features, kernel_size=4, stride=2, padding=1, bias=False), # 28x28 -> 14x14
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features) x 14 x 14
            nn.Conv2d(disc_features, disc_features * 2, kernel_size=4, stride=2, padding=1, bias=False), # 14x14 -> 7x7
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_features*2) x 7 x 7
            nn.Conv2d(disc_features * 2, 1, kernel_size=7, stride=1, padding=0, bias=False), # 7x7 -> 1x1
            nn.Sigmoid()
        )

    def forward(self, input):
        # Flatten the output to a 1D tensor of probabilities
        return self.main(input).view(-1)

# --- Train model ---
def train_gan(netG, netD, optimizerG, optimizerD, train_loader, num_epochs, current_device, latent_dim):
    """Main training loop for the GAN."""
    netG.train()
    netD.train()
    
    criterion = nn.BCELoss()
    history = {'G_loss': [], 'D_loss': []}
    
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    print("\n--- Starting GAN Training ---")
    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0, 0
        num_batches = len(train_loader)

        for i, (real_images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            
            # --- (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z))) ---
            netD.zero_grad()
            
            # Train with all-real batch
            real_images = real_images.to(current_device)
            batch_size = real_images.size(0)
            labels = torch.full((batch_size,), real_label, dtype=torch.float, device=current_device)
            
            output = netD(real_images)
            errD_real = criterion(output, labels)
            errD_real.backward()
            
            # Train with all-fake batch
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=current_device)
            fake_images = netG(noise)
            labels.fill_(fake_label)
            
            output = netD(fake_images.detach())
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # --- (2) Update Generator network: maximize log(D(G(z))) ---
            netG.zero_grad()
            labels.fill_(real_label)
            output = netD(fake_images)
            errG = criterion(output, labels)
            errG.backward()
            optimizerG.step()
            
            g_loss_epoch += errG.item()
            d_loss_epoch += errD.item()

        avg_g_loss = g_loss_epoch / num_batches
        avg_d_loss = d_loss_epoch / num_batches
        history['G_loss'].append(avg_g_loss)
        history['D_loss'].append(avg_d_loss)
        print(f"\tEpoch {epoch + 1}/{num_epochs} \tGenerator Loss: {avg_g_loss:.4f} \tDiscriminator Loss: {avg_d_loss:.4f}")

    torch.save(netG.state_dict(), f'outputs_gan/weights/gan_generator_final.pth')
    torch.save(netD.state_dict(), f'outputs_gan/weights/gan_discriminator_final.pth')
    print("Final model weights saved.")
    
    return history

# --- PLOTTING FUNCTION 1: Loss Curve ---
def plot_loss(history):
    print("\n--- Plotting Training Loss Curve ---")
    plt.figure(figsize=(10, 6))
    plt.plot(history['G_loss'], label='Generator Loss')
    plt.plot(history['D_loss'], label='Discriminator Loss')
    plt.title('GAN Training Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig('outputs_gan/gan_loss_curve.png')
    plt.close()
    print("Loss curve plot saved as 'outputs_gan/gan_loss_curve.png'")

# --- PLOTTING FUNCTION 2: Generated Image Grid ---
def plot_generated_images(netG, current_device, latent_dim, num_images=64):
    print("\n--- Generating and Plotting a Grid of Fake Images ---")
    netG.eval()
    fixed_noise = torch.randn(num_images, latent_dim, 1, 1, device=current_device)
    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()

    grid = make_grid(fake_images, padding=2, normalize=True)
    save_image(grid, 'outputs_gan/gan_generated_image_grid.png')
    np_grid = grid.permute(1, 2, 0).numpy()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off"); plt.title("Generated Images"); plt.imshow(np_grid)
    plt.savefig('outputs_gan/gan_generated_images_plot.png'); plt.close()
    
    print("Grid of generated images saved to 'outputs_gan/gan_generated_image_grid.png'")

# ##############################################################################
# ### THIS IS THE CORRECTED FUNCTION TO PRODUCE THE TIGHTLY TILED PLOT ###
# ##############################################################################
def plot_latent_space_manifold(netG, current_device, latent_dim, grid_size=20, val_range=2.0):
    if latent_dim != 2:
        print("\n--- Latent space manifold plot skipped (requires latent_dim=2) ---")
        return
    print("\n--- Visualizing the 2D Latent Space Manifold (Tightly Tiled) ---")
    netG.eval()
    
    # Create a 2D grid of latent vectors
    x_vals = np.linspace(-val_range, val_range, grid_size)
    y_vals = np.linspace(-val_range, val_range, grid_size)
    latent_grid_vectors = []
    # The order of loops is important for the final grid layout
    for y in y_vals:
        for x in x_vals:
            latent_grid_vectors.append([x, y])
    
    latent_grid_tensor = torch.tensor(latent_grid_vectors, dtype=torch.float32, device=current_device).view(-1, latent_dim, 1, 1)

    with torch.no_grad():
        # Generate all images in one batch
        generated_images = netG(latent_grid_tensor).detach().cpu()

    # Use make_grid to stitch all images into a single image tensor.
    # Set padding=0 to remove the background/gutters between images.
    grid = make_grid(generated_images, nrow=grid_size, padding=0, normalize=True)
    
    # Now, plot this single, pre-stitched grid image.
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid.permute(1, 2, 0)) # Reorder dimensions for matplotlib
    ax.set_title('2D Latent Space Manifold', fontsize=16)
    ax.axis('off') # Turn off the axes and ticks

    plt.tight_layout()
    plt.savefig('outputs_gan/gan_2d_latent_manifold_tiled.png')
    plt.close()
    print("Tightly tiled 2D Latent space manifold plot saved as 'outputs_gan/gan_2d_latent_manifold_tiled.png'")

# --- MAIN EXECUTION ---
def main():
    print("--- GAN MNIST Training Script ---")
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    # Set latent_dim to 2 to generate the 2D manifold plot
    latent_dim = 2
    num_channels = 1
    epochs = 25 
    lr = 0.0002
    beta1 = 0.5
    batch_size = 128
    
    # --- Initialize Models ---
    netG = Generator(latent_dim=latent_dim, num_channels=num_channels).to(device)
    netD = Discriminator(num_channels=num_channels).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # --- Setup Optimizers ---
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # --- Loading dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    path = os.path.expanduser('~/datasets')
    train_dataset = datasets.MNIST(root=path, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # --- Train the Model ---
    history = train_gan(netG, netD, optimizerG, optimizerD, train_loader, epochs, device, latent_dim)
    
    # --- Generate All Visualizations ---
    plot_loss(history)
    plot_generated_images(netG, device, latent_dim)
    # This will now create the plot you want
    plot_latent_space_manifold(netG, device, latent_dim)
    
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()