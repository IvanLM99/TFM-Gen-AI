import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- Configuration ---
# Model Paths
vae_model_path = 'outputs_vae/vae_mnist.pth'
generator_model_path = 'outputs_gan/generator_mnist.pth'
output_dir = "outputs_projection"
os.makedirs(output_dir, exist_ok=True)

# Latent space dimensions (match trained models)
vae_latent_dim = 50
gan_latent_dim = 100

# --- PROJECTION PARAMETERS ---
num_projection_samples = 500
lr_inversion = 0.01
num_inversion_steps = 3000

# --- Setup Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Model Definitions ---
# VAE Definition
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim_internal=200, latent_dim_output=50):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim_internal),
            nn.LeakyReLU(0.2)
        )
        self.mean_layer = nn.Linear(latent_dim_internal, latent_dim_output)
        self.logvar_layer = nn.Linear(latent_dim_internal, latent_dim_output)
        self.decoder_nn = nn.Sequential(
            nn.Linear(latent_dim_output, latent_dim_internal),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim_internal, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # VAE uses Sigmoid for [0, 1] output
        )
    
    def encode(self, x):
        x = self.encoder(x.view(-1, 784))
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def decode(self, z):
        x_hat = self.decoder_nn(z)
        return x_hat.view(-1, 1, 28, 28)

# Generator Definition
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh() # GAN uses Tanh for [-1, 1] output
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1, 28, 28)


# --- Helper Functions ---
def denormalize_to_0_1(tensor_image):
    """Denormalizes a tensor from [-1, 1] (GAN) or [0, 1] (VAE) to [0, 1] for fair comparison."""
    return (tensor_image.clamp(-1, 1) + 1.0) / 2.0

def invert_gan_to_find_z(target_image, gan_generator, gan_latent_dim, num_steps, lr):
    """
    Performs GAN inversion via optimization to find the latent vector z
    that best reconstructs the target_image.
    
    Args:
        target_image (Tensor): The image to reconstruct. Assumed to be in [-1, 1] range.
        gan_generator (nn.Module): The pre-trained GAN generator.
        gan_latent_dim (int): The dimension of the GAN's latent space.
        num_steps (int): Number of optimization steps.
        lr (float): Learning rate for the optimization.
    Returns:
        Tensor: The optimized latent vector z_gan.
    """
    # Start with a random latent vector
    z_optimize = torch.randn(1, gan_latent_dim, device=device, requires_grad=True)
    optimizer_z = optim.Adam([z_optimize], lr=lr)
    reconstruction_loss_fn = nn.MSELoss()
    for _ in range(num_steps):
        optimizer_z.zero_grad()
        generated_image = gan_generator(z_optimize)
        loss = reconstruction_loss_fn(generated_image, target_image)
        loss.backward()
        optimizer_z.step()
    return z_optimize.detach()

def main():
    # --- 1. Load Models ---
    print("\n--- Loading pre-trained VAE and GAN Generator ---")
    if not (os.path.exists(vae_model_path) and os.path.exists(generator_model_path)):
        print("ERROR: One or more model files not found. Please run training scripts first.")
        return

    vae_model = VAE(latent_dim_output=vae_latent_dim).to(device)
    vae_model.load_state_dict(torch.load(vae_model_path, map_location=device))
    vae_model.eval()
    print("VAE model loaded successfully.")

    gan_generator = Generator(gan_latent_dim).to(device)
    gan_generator.load_state_dict(torch.load(generator_model_path, map_location=device))
    gan_generator.eval()
    print("GAN Generator model loaded successfully.")

    # --- 2. Load Data ---
    # VAE expects input in [0, 1] range.
    transform_vae = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_vae)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # --- 3. Collect Latent Space Pairs (z_vae, z_gan) ---
    print(f"\n--- Collecting {num_projection_samples} latent vector pairs ---")
    z_vae_list, z_gan_list = [], []
    for i, (image_original, _) in enumerate(tqdm(data_loader, desc="Projecting samples", total=num_projection_samples)):
        if i >= num_projection_samples: break
        image_original = image_original.to(device)
        with torch.no_grad():
            z_vae, _ = vae_model.encode(image_original)
        target_image_for_gan = (image_original * 2.0) - 1.0
        z_gan_found = invert_gan_to_find_z(target_image_for_gan, gan_generator, gan_latent_dim, num_inversion_steps, lr_inversion)
        z_vae_list.append(z_vae)
        z_gan_list.append(z_gan_found)

    Z_vae = torch.cat(z_vae_list, dim=0)
    Z_gan = torch.cat(z_gan_list, dim=0)

    # --- 4. Compute and Save the Linear Transformation ---
    print("\n--- Computing and Saving Linear Transformation T ---")
    solution = torch.linalg.lstsq(Z_vae, Z_gan)
    transformation_matrix = solution.solution
    matrix_path = os.path.join(output_dir, "vae_to_gan_matrix.pth")
    torch.save(transformation_matrix, matrix_path)
    print(f"Transformation matrix T of shape {transformation_matrix.shape} saved to '{matrix_path}'")
    
    # --- 5. Test and Visualize the Transformation ---
    print("\n--- Testing the transformation on a new sample ---")
    data_iterator = iter(data_loader)
    for _ in range(num_projection_samples + 25):
        test_image_original, test_label = next(data_iterator)
    test_image_original = test_image_original.to(device)

    # a) Get z_vae and the "ground truth" z_gan via inversion
    with torch.no_grad():
        z_vae_true, _ = vae_model.encode(test_image_original)
    test_image_for_gan = (test_image_original * 2.0) - 1.0
    z_gan_optimized = invert_gan_to_find_z(test_image_for_gan, gan_generator, gan_latent_dim, num_inversion_steps, lr_inversion)

    # b) Project z_vae to get the predicted z_gan
    with torch.no_grad():
        z_gan_projected = z_vae_true @ transformation_matrix

    # c) Generate images from all key latent vectors
    with torch.no_grad():
        img_from_vae_recon = vae_model.decode(z_vae_true)
        img_from_gan_optimized = gan_generator(z_gan_optimized)
        img_from_gan_projected = gan_generator(z_gan_projected)

    # --- 6. Calculate Metrics and Save Results ---
    print("\n--- Results for a Test Sample ---")
    print(f"Image Label: {test_label.item()}")
    
    # --- Latent Space Metrics ---
    l2_diff = torch.norm(z_gan_optimized - z_gan_projected).item()
    cosine_sim = F.cosine_similarity(z_gan_optimized, z_gan_projected).item()
    l_mse = F.mse_loss(z_gan_projected, z_gan_optimized).item()
    
    # --- Image Space Metrics (as in the paper) ---
    # all images are in the same [0, 1] range for fair comparison
    img_orig_norm = denormalize_to_0_1(test_image_original)
    img_vae_norm = denormalize_to_0_1(img_from_vae_recon)
    img_gan_proj_norm = denormalize_to_0_1(img_from_gan_projected)
    
    r_mse = F.mse_loss(img_vae_norm, img_orig_norm).item()
    m_mse = F.mse_loss(img_gan_proj_norm, img_orig_norm).item()

    print("\n--- Latent Vector Comparison ---")
    print(f"L-MSE (Latent Vector MSE):         {l_mse:.4f}")
    print(f"L2 Distance (Latent Vectors):      {l2_diff:.4f}")
    print(f"Cosine Similarity (Latent Vectors): {cosine_sim:.4f}")

    print("\n--- Image Space Error (Paper's Metrics) ---")
    print(f"R-MSE (VAE Reconstruction Error):  {r_mse:.4f}")
    print(f"M-MSE (Mapped GAN Error):          {m_mse:.4f}")

    # --- Print Latent Vectors and Matrix ---
    np.set_printoptions(precision=4, suppress=True)
    print("\n--- Latent Vector Snippets ---")
    print(f"Original z_vae (first 10):      {z_vae_true.cpu().numpy().flatten()[:10]}")
    print(f"'Ground Truth' z_gan (first 10): {z_gan_optimized.cpu().numpy().flatten()[:10]}")
    print(f"Projected z_gan (first 10):     {z_gan_projected.cpu().numpy().flatten()[:10]}")
    
    print("\n--- Linear Transformation Matrix (T) ---")
    print("This matrix T maps z_vae to z_gan (z_gan = z_vae @ T)")
    print(transformation_matrix.cpu().numpy()[:5, :5]) # Print top-left 5x5 corner
    print("...")

    # --- Final Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'VAE to GAN Projection (Label: {test_label.item()})', fontsize=16)
    
    images_to_plot = [
        denormalize_to_0_1(test_image_original.cpu().squeeze()),
        denormalize_to_0_1(img_from_vae_recon.cpu().squeeze()),
        denormalize_to_0_1(img_from_gan_optimized.cpu().squeeze()),
        denormalize_to_0_1(img_from_gan_projected.cpu().squeeze())
    ]
    titles = ['Original Image', 'VAE Reconstruction', 'GAN (Inverted z)', 'GAN (Projected z)']

    for ax, img, title in zip(axes, images_to_plot, titles):
        ax.imshow(img, cmap='gray'); ax.set_title(title); ax.axis('off')
        
    plt.figtext(0.5, 0.01, f"M-MSE: {m_mse:.4f} | L-MSE: {l_mse:.4f} | Cosine Sim: {cosine_sim:.4f}", ha="center", fontsize=12)
    plot_path = os.path.join(output_dir, "projection_comparison_plot.png")
    plt.savefig(plot_path)
    print(f"\nSaved final comparison plot to '{plot_path}'")

if __name__ == '__main__':
    if not os.path.exists(vae_model_path) or not os.path.exists(generator_model_path):
        print(f"ERROR: Missing model files.\n  - VAE: {vae_model_path}\n  - GAN: {generator_model_path}")
    else:
        main()