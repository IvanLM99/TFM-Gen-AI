import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
nz = 100  # Latent vector size (must match the trained generator)
lr_inversion = 0.01  # Learning rate for optimizing z
num_inversion_steps = 5000 # Number of optimization steps for z
generator_model_path = 'outputs/generator_mnist.pth' # Path to your trained generator
output_dir_inversion = "outputs_inversion"
os.makedirs(output_dir_inversion, exist_ok=True)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# --- Generator Network Definition (same as in your training script) ---
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
            nn.Linear(1024, 784),  # 28*28 for MNIST
            nn.Tanh()  # Output images in [-1, 1]
        )
        print("Generator class defined.")

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1, 28, 28) # Reshape to image format

def denormalize_image(tensor_image):
    """Denormalizes an image tensor from [-1, 1] to [0, 1]."""
    return (tensor_image + 1.0) / 2.0

def main_inversion():
    # --- 1. Load Pre-trained Generator ---
    print(f"Loading pre-trained generator from: {generator_model_path}")
    if not os.path.exists(generator_model_path):
        print(f"Error: Generator model not found at {generator_model_path}.")
        print("Please make sure you have trained the GAN first and the model is saved correctly.")
        return

    netG = Generator(nz).to(device)
    netG.load_state_dict(torch.load(generator_model_path, map_location=device))
    netG.eval() # Set generator to evaluation mode
    print("Generator loaded successfully.")

    # --- 2. Get a Target Image ---
    # use an image from the MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # Normalize to [-1, 1]
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    test_data = datasets.MNIST(
        root='./data',
        train=False, # Use test set
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True) # Shuffle to get different images each run

    # Get a single target image
    try:
        target_image, target_label = next(iter(test_loader))
    except StopIteration:
        print("Error: Could not load an image from the test set.")
        return
        
    target_image = target_image.to(device)
    print(f"Target image shape: {target_image.shape}, Label: {target_label.item()}")

    # --- 3. Initialize Latent Vector z ---
    # Initialize z randomly or with zeros. Requires gradients for optimization.
    z_optimize = torch.randn(1, nz, device=device, requires_grad=True)
    # initialize with zeros:
    # z_optimize = torch.zeros(1, nz, device=device, requires_grad=True)
    
    # Generate an image from the initial z to see the starting point
    with torch.no_grad():
        initial_generated_image = netG(z_optimize)

    # --- 4. Define Loss Function ---
    # Mean Squared Error (L2 loss) is common for image reconstruction
    reconstruction_loss_fn = nn.MSELoss()
    # also try L1 loss:
    # reconstruction_loss_fn = nn.L1Loss()

    # --- 5. Optimizer for z ---
    # optimize only the latent vector z
    optimizer_z = optim.Adam([z_optimize], lr=lr_inversion)

    print("\n--- Starting GAN Inversion Optimization ---")
    losses = []
    for step in range(num_inversion_steps):
        optimizer_z.zero_grad()

        # Generate image from current z_optimize
        generated_image = netG(z_optimize)

        # Calculate reconstruction loss
        loss = reconstruction_loss_fn(generated_image, target_image)
        
        # (Optional) Add a regularization term for z if needed (e.g., to keep z close to origin)
        # z_regularization_loss = torch.norm(z_optimize, p=2) * 0.001 # L2 norm
        # total_loss = loss + z_regularization_loss
        # total_loss.backward()

        loss.backward()
        optimizer_z.step()

        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(f"Step [{step+1}/{num_inversion_steps}], Loss: {loss.item():.6f}")

    print("--- GAN Inversion Optimization Finished ---")

    # --- 6. Visualize Results ---
    # Get the final reconstructed image
    with torch.no_grad():
        reconstructed_image = netG(z_optimize)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("GAN Inversion: Latent Vector Optimization Loss")
    plt.xlabel("Optimization Step")
    plt.ylabel("Reconstruction Loss (MSE)")
    loss_plot_path = os.path.join(output_dir_inversion, 'inversion_loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"\nInversion loss plot saved to {loss_plot_path}")

    # Display side-by-side (Target, Initial, Reconstructed)
    target_img_np = denormalize_image(target_image.cpu().squeeze()).numpy()
    initial_img_np = denormalize_image(initial_generated_image.cpu().squeeze()).numpy()
    reconstructed_img_np = denormalize_image(reconstructed_image.cpu().squeeze()).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"GAN Inversion Results (Label: {target_label.item()})", fontsize=16)
    
    axes[0].imshow(target_img_np, cmap='gray')
    axes[0].set_title("Target Image")
    axes[0].axis('off')
    
    axes[1].imshow(initial_img_np, cmap='gray')
    axes[1].set_title("Initial from Random z")
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_img_np, cmap='gray')
    axes[2].set_title(f"Reconstructed (Loss: {losses[-1]:.4f})")
    axes[2].axis('off')
    
    comparison_plot_path = os.path.join(output_dir_inversion, 'inversion_comparison.png')
    plt.savefig(comparison_plot_path)
    print(f"Comparison plot saved to {comparison_plot_path}")

    print(f"\nFinal optimized z values (first 10): {z_optimize.detach().cpu().numpy()[0, :10]}")
    print(f"Norm of optimized z: {torch.norm(z_optimize).item():.4f}")

if __name__ == '__main__':
    main_inversion()
