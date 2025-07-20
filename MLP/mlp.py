# ==============================================================================
#
#       VAE <--> GAN Latent Space Projections
#
#
# --- Project Overview ---
# This script implements a complete, symmetrical pipeline for projecting images
# between the latent spaces of a Variational Autoencoder (VAE) and a
# Generative Adversarial Network (GAN) trained on the MNIST dataset.
#
# The core of the projection is a pair of MLP Regressors that act as
# bidirectional "bridges" between the two latent spaces.
#
#
# --- Pipeline Details ---
#
# Direction 1: VAE -> GAN Projection
#   1. A real image is selected from the MNIST test set.
#   2. The pre-trained VAE encodes this image into a 2D latent vector (z_vae).
#   3. An MLP bridge (trained for VAE->GAN) takes z_vae as input and predicts
#      the corresponding latent vector in the GAN's space (predicted_z_gan).
#   4. The pre-trained GAN Generator uses predicted_z_gan to synthesize a
#      new image, completing the projection.
#
# Direction 2: GAN -> VAE Projection
#   1. A real image is selected from the MNIST test set.
#   2. GAN Inversion is used to find the 2D latent vector (z_gan) 
#      that allows the GAN to best reconstruct the real image.
#   3. A second MLP bridge (trained for GAN->VAE) takes z_gan as input and
#      predicts the corresponding vector in the VAE's space (predicted_z_vae).
#   4. The pre-trained VAE Decoder uses predicted_z_vae to reconstruct a
#      final image, completing the projection.
#
# ==============================================================================

# ##############################################################################
# ### 1. IMPORTS ###
# ##############################################################################
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# ##############################################################################
# ### 2. MODEL DEFINITIONS ###
# ##############################################################################

class VAE(nn.Module):
    """
    A Variational Autoencoder for MNIST, mapping images to a 2D latent space.
    It consists of an encoder that maps an image to a 2D latent distribution
    (mean and log-variance) and a decoder that reconstructs an image from a
    point sampled from this latent space.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim_internal=200, latent_dim_output=2):
        super(VAE, self).__init__()
        # Encoder maps the 784-pixel image down to an intermediate representation.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim_internal), nn.LeakyReLU(0.2)
        )
        # Two separate layers predict the mean and log-variance of the latent distribution.
        self.mean_layer = nn.Linear(latent_dim_internal, latent_dim_output)
        self.logvar_layer = nn.Linear(latent_dim_internal, latent_dim_output)
        # Decoder maps a 2D latent vector back to a 784-pixel image.
        self.decoder_nn = nn.Sequential(
            nn.Linear(latent_dim_output, latent_dim_internal), nn.LeakyReLU(0.2),
            nn.Linear(latent_dim_internal, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # Sigmoid ensures output pixel values are between 0 and 1.
        )
        
    """Passes an image through the encoder to get the latent distribution parameters."""
    def encode(self, x): 
        h = self.encoder(x.view(-1, 784))
        return self.mean_layer(h), self.logvar_layer(h)
    """Passes a latent vector through the decoder to reconstruct an image."""
    def decode(self, z): 
        return self.decoder_nn(z)

class Generator(nn.Module):
    """
    A GAN Generator for MNIST.
    It uses a series of transposed convolutions to upscale a 2D latent
    vector into a 28x28 grayscale image.
    """
    def __init__(self, latent_dim=2, num_channels=1, gen_features=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, gen_features * 4, 3, 2, 0, bias=False), nn.BatchNorm2d(gen_features * 4), nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(gen_features * 2), nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 2, gen_features, 4, 2, 1, bias=False), nn.BatchNorm2d(gen_features), nn.ReLU(True),
            nn.ConvTranspose2d(gen_features, num_channels, 3, 3, 4, bias=False), 
            nn.Tanh() # Tanh ensures output pixel values are between -1 and 1.
        )
    """Generates an image from a latent vector z."""
    def forward(self, z): return self.main(z)


# ##############################################################################
# ### 3. SETUP, TRAINING, AND DATA PREPARATION ###
# ##############################################################################

def load_models(device):
    """
    Loads the pre-trained VAE and GAN Generator models from disk and sets them
    to evaluation mode.
    
    Args:
        device (torch.device): The device (CPU or CUDA) to load the models onto.
    
    Returns:
        tuple: A tuple containing the loaded (vae_model, gan_generator).
    """
    print("--- Loading pre-trained VAE and GAN models ---")
    try:
        vae_model = VAE().to(device); vae_model.load_state_dict(torch.load('outputs_vae/vae_model_2d.pth', map_location=device)); vae_model.eval()
        gan_generator = Generator().to(device); gan_generator.load_state_dict(torch.load('outputs_gan/weights/gan_generator_final.pth', map_location=device)); gan_generator.eval()
    except FileNotFoundError as e:
        print(f"Error: Could not find a model file. Make sure pre-trained models exist.\nDetails: {e}"); exit()
    return vae_model, gan_generator

def create_paired_latent_dataset(vae_model, gan_generator, device, num_samples=75000):
    """
    Generates a paired (z_vae, z_gan) dataset. This is the foundation for
    training the MLP networks. It works by creating an image from a known GAN
    vector and then finding where the VAE places that image in its own space.
    
    Args:
        vae_model (nn.Module): The pre-trained VAE.
        gan_generator (nn.Module): The pre-trained GAN Generator.
        device (torch.device): The device to run computations on.
        num_samples (int): The number of vector pairs to generate.
        
    Returns:
        tuple: A tuple of numpy arrays (paired_vae_vectors, paired_gan_vectors).
    """
    print("\n--- Creating Paired Latent Dataset for Training Bridges ---")
    paired_vae_vectors_list, paired_gan_vectors_list = [], []
    batch_size = 512
    with torch.no_grad():
        for _ in tqdm(range(int(np.ceil(num_samples / batch_size))), desc="Generating paired (VAE, GAN) vectors"):
            # 1. Create a batch of random latent vectors for the GAN.
            z_gan = torch.randn(batch_size, 2, 1, 1, device=device)
            # 2. Use the GAN to generate images from these vectors.
            gan_images = gan_generator(z_gan)
            # 3. Normalize GAN images from [-1, 1] to [0, 1] for the VAE.
            gan_images_normalized = (gan_images + 1) / 2.0
            # 4. Use the VAE to encode the GAN's images, getting the corresponding z_vae.
            z_vae, _ = vae_model.encode(gan_images_normalized)
            # 5. Store the original z_gan and the resulting z_vae as a pair.
            paired_gan_vectors_list.append(z_gan.squeeze().cpu().numpy())
            paired_vae_vectors_list.append(z_vae.cpu().numpy())
    return np.concatenate(paired_vae_vectors_list, axis=0), np.concatenate(paired_gan_vectors_list, axis=0)

def train_bridge(X_train, Y_train, direction):
    """
    Trains an MLP Regressor to learn the mapping from one latent space to another.
    
    Args:
        X_train (np.array): The input vectors (source space).
        Y_train (np.array): The target vectors (destination space).
        direction (str): A string describing the projection direction for logging.
        
    Returns:
        MLPRegressor: The trained scikit-learn MLP regressor model.
    """
    print(f"\n--- Training MLP Bridge for {direction} ---")
    regressor = MLPRegressor(hidden_layer_sizes=(2, 4, 2), activation='identity', solver='adam', max_iter=1000, random_state=42, early_stopping=True, verbose=False)
    print("Fitting regressor... (This may take a minute)")
    regressor.fit(X_train, Y_train)
    mse = mean_squared_error(Y_train, regressor.predict(X_train))
    print(f"MLP Regressor for {direction} trained. Final Training MSE on vectors: {mse:.4f}")
    return regressor

def gan_inversion(gan_generator, target_image, device, steps=2500, lr=0.01):
    """
    Finds the GAN latent vector 'z' that best reconstructs the 'target_image'
    using an iterative optimization process (gradient descent).
    
    Args:
        gan_generator (nn.Module): The pre-trained GAN Generator.
        target_image (torch.Tensor): The real image to be reconstructed.
        device (torch.device): The device to run optimization on.
        steps (int): The number of optimization steps.
        lr (float): The learning rate for the optimizer.
        
    Returns:
        torch.Tensor: The optimized latent vector z.
    """
    # Ensure target image has a batch dimension [1, C, H, W] and is in the GAN's [-1, 1] range.
    target_image_gan_range = (target_image.to(device) * 2 - 1).unsqueeze(0)
    # Initialize a random latent vector that we will optimize.
    z = torch.randn(1, 2, 1, 1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)
    loss_fn = nn.MSELoss()
    # Optimization loop
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(gan_generator(z), target_image_gan_range) # Compare generated vs. target
        loss.backward() # Calculate gradients of the loss with respect to z
        optimizer.step() # Update z to minimize the loss
    return z.detach()

def create_visualization_datasets(vae_model, gan_generator, device):
    """
    Generates labeled latent vectors for VAE and GAN spaces, used only for
    plotting the background scatter of the spaces in visualizations.
    
    Returns:
        tuple: A tuple containing ((vae_vectors, vae_labels), (gan_vectors, gan_labels)).
    """
    print("\n--- Creating latent space datasets for visualization ---")
    mnist_train = datasets.MNIST(root='~/datasets', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=512, shuffle=False)
    # VAE space: Encode all real training images.
    with torch.no_grad():
        all_vae_vectors = np.concatenate([vae_model.encode(img.to(device))[0].cpu().numpy() for img, _ in tqdm(train_loader, desc="1/2: Visualizing VAE space")], axis=0)
    all_vae_labels = mnist_train.targets.numpy()
    # GAN space: Generate images and label them by finding the closest VAE digit centroid.
    vae_centroids = np.array([all_vae_vectors[all_vae_labels == i].mean(axis=0) for i in range(10)])
    all_gan_vectors_list, all_gan_labels_list = [], []
    with torch.no_grad():
        for _ in tqdm(range(200), desc="2/2: Visualizing GAN space"):
            random_gan_vectors = torch.randn(256, 2, 1, 1, device=device)
            gan_images = gan_generator(random_gan_vectors)
            gan_images_normalized = (gan_images + 1) / 2.0
            corresponding_vae_vectors, _ = vae_model.encode(gan_images_normalized)
            distances = torch.cdist(corresponding_vae_vectors, torch.tensor(vae_centroids, device=device, dtype=torch.float))
            predicted_labels = torch.argmin(distances, dim=1)
            all_gan_vectors_list.append(random_gan_vectors.squeeze().cpu().numpy())
            all_gan_labels_list.append(predicted_labels.cpu().numpy())
    all_gan_vectors = np.concatenate(all_gan_vectors_list, axis=0)
    all_gan_labels = np.concatenate(all_gan_labels_list, axis=0)
    return (all_vae_vectors, all_vae_labels), (all_gan_vectors, all_gan_labels)


# ##############################################################################
# ### 4. VISUALIZATION FUNCTIONS ###
# ##############################################################################

def _create_plot(fig_title, plot_data):
    """
    A master helper function to generate the 2x2 projection plot, showing
    the start/end images and the corresponding latent space points.
    
    Args:
        fig_title (str): The main title for the entire figure.
        plot_data (dict): A dictionary containing all necessary data for plotting.
        
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    fig = plt.figure(figsize=(15, 14)); fig.suptitle(fig_title, fontsize=18, y=0.98)
    
    # Create a 2x2 main grid. The top row is for images, bottom row for plots.
    gs_main = GridSpec(2, 2, figure=fig, height_ratios=[0.8, 1])
    ax_img1 = fig.add_subplot(gs_main[0, 0])
    ax_img2 = fig.add_subplot(gs_main[0, 1])
    
    # Create a nested subgrid in the bottom row to make space for the color bar.
    # This creates 3 columns: plot, plot, and a narrow one for the color bar.
    gs_bottom = gs_main[1, :].subgridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    ax_ls1 = fig.add_subplot(gs_bottom[0, 0])
    ax_ls2 = fig.add_subplot(gs_bottom[0, 1])
    colorbar_axis = fig.add_subplot(gs_bottom[0, 2]) # This is the dedicated axis for the color bar.
    
    # Plot images in the top row.
    ax_img1.imshow(plot_data['img1'], cmap='gray'); ax_img1.set_title(plot_data['title1'], fontsize=12); ax_img1.axis('off')
    ax_img2.imshow(plot_data['img2'], cmap='gray'); ax_img2.set_title(plot_data['title2'], fontsize=12); ax_img2.axis('off')
    
    # Plot latent spaces in the bottom row.
    cmap = plt.get_cmap('tab10', 10); alpha, s = 0.5, 15
    ax_ls1.scatter(plot_data['X1'][:, 0], plot_data['X1'][:, 1], c=plot_data['y1'], cmap=cmap, alpha=alpha, s=s)
    ax_ls1.scatter(plot_data['vec1'][:, 0], plot_data['vec1'][:, 1], c='k', marker='*', s=350, edgecolors='w', label=plot_data['label1'])
    ax_ls1.set_title(plot_data['title_ls1'], fontsize=12); ax_ls1.set_xlabel("Latent Dim 1"); ax_ls1.set_ylabel("Latent Dim 2"); ax_ls1.legend(loc='upper right')
    
    # The second scatter plot's artist is captured to generate the color bar.
    scatter_artist = ax_ls2.scatter(plot_data['X2'][:, 0], plot_data['X2'][:, 1], c=plot_data['y2'], cmap=cmap, alpha=alpha, s=s)
    ax_ls2.scatter(plot_data['vec2'][:, 0], plot_data['vec2'][:, 1], c='k', marker='*', s=350, edgecolors='w', label=plot_data['label2'])
    ax_ls2.set_title(plot_data['title_ls2'], fontsize=12); ax_ls2.set_xlabel("Latent Dim 1"); ax_ls2.set_ylabel("Latent Dim 2"); ax_ls2.legend(loc='upper right')
    
    # Add the color bar to its dedicated axis.
    fig.colorbar(scatter_artist, cax=colorbar_axis, label='Digit Class', ticks=np.arange(10))
    
    fig.tight_layout(rect=[0, 0, 1, 0.94]); return fig

def visualize_vae_to_gan(vae_model, gan_generator, regressor, device, vis_data, test_dataset, seed, num):
    """Visualizes the full VAE -> GAN projection pipeline for one image."""
    (X_vae, y_vae), (X_gan, y_gan) = vis_data
    print(f"\n--- Running VAE -> GAN projection (Experiment #{num}) ---")
    np.random.seed(seed); random_idx = np.random.randint(0, len(test_dataset))
    img, label = test_dataset[random_idx]
    # Step 1: Encode real image with VAE to get the initial vector.
    with torch.no_grad(): z_vae, _ = vae_model.encode(img.to(device))
    z_vae_np = z_vae.cpu().numpy()
    # Step 2: Use MLP bridge to predict the corresponding GAN vector.
    pred_gan_vec_np = regressor.predict(z_vae_np)
    # Step 3: Use GAN to generate the final image from the predicted vector.
    with torch.no_grad(): final_img = gan_generator(torch.tensor(pred_gan_vec_np, device=device, dtype=torch.float).view(1, 2, 1, 1)).squeeze().cpu().numpy()
    # Step 4: Calculate MSE for evaluation.
    mse = np.mean(((img.cpu().squeeze().numpy() * 2 - 1) - final_img)**2)
    # Prepare data for plotting.
    plot_data = {
        'img1': img.squeeze(), 'img2': final_img,
        'title1': f"1. Real image (Label: {label})\nEncoded to VAE Vector: ({z_vae_np[0,0]:.2f}, {z_vae_np[0,1]:.2f})",
        'title2': f"4. Final image from predicted GAN Vector\nVector: ({pred_gan_vec_np[0,0]:.2f}, {pred_gan_vec_np[0,1]:.2f})",
        'X1': X_vae, 'y1': y_vae, 'vec1': z_vae_np, 'label1': 'Initial VAE Vector',
        'X2': X_gan, 'y2': y_gan, 'vec2': pred_gan_vec_np, 'label2': 'Predicted GAN Vector',
        'title_ls1': "2. VAE encodes image to get z_vae.", 'title_ls2': "3. MLP predicts z_gan from z_vae."
    }
    fig = _create_plot(f"Direction 1 (Run {num}): VAE -> GAN projection\nImage projection MSE = {mse:.4f}", plot_data)
    fig.savefig(f"outputs_mlp/dir1_vae_to_gan_run_{num}.png"); plt.close(fig)
    print(f"Saved VAE->GAN plot to 'outputs_mlp/dir1_vae_to_gan_run_{num}.png'")

def visualize_gan_to_vae(vae_model, gan_generator, regressor, device, vis_data, test_dataset, seed, num):
    """Visualizes the full GAN -> VAE projection pipeline for one image."""
    (X_vae, y_vae), (X_gan, y_gan) = vis_data
    print(f"\n--- Running GAN -> VAE projection (Experiment #{num}) ---")
    np.random.seed(seed); random_idx = np.random.randint(0, len(test_dataset))
    img, label = test_dataset[random_idx]
    # Step 1: Use GAN Inversion to find the initial GAN vector for the real image.
    print("Performing GAN Inversion... (This is slow, as it's an optimization process)")
    z_gan_inv = gan_inversion(gan_generator, img, device)
    z_gan_inv_np = z_gan_inv.squeeze().cpu().numpy().reshape(1, 2)
    # Step 2: Use MLP bridge to predict the corresponding VAE vector.
    pred_vae_vec_np = regressor.predict(z_gan_inv_np)
    # Step 3: Use VAE decoder to generate the final image from the predicted vector.
    with torch.no_grad(): final_img = vae_model.decode(torch.tensor(pred_vae_vec_np, device=device, dtype=torch.float)).squeeze().view(28,28).cpu().numpy()
    # Step 4: Calculate MSE for evaluation.
    mse = mean_squared_error(img.squeeze().numpy(), final_img)
    # Prepare data for plotting.
    plot_data = {
        'img1': img.squeeze(), 'img2': final_img,
        'title1': f"1. Real image (Label: {label})\nInverted to GAN Vector: ({z_gan_inv_np[0,0]:.2f}, {z_gan_inv_np[0,1]:.2f})",
        'title2': f"4. Final image from predicted VAE Vector\nVector: ({pred_vae_vec_np[0,0]:.2f}, {pred_vae_vec_np[0,1]:.2f})",
        'X1': X_gan, 'y1': y_gan, 'vec1': z_gan_inv_np, 'label1': 'Initial GAN Vector',
        'X2': X_vae, 'y2': y_vae, 'vec2': pred_vae_vec_np, 'label2': 'Predicted VAE Vector',
        'title_ls1': "2. GAN Inversion finds z_gan.", 'title_ls2': "3. MLP predicts z_vae from z_gan."
    }
    fig = _create_plot(f"Direction 2 (Run {num}): GAN -> VAE projection\nImage projection MSE = {mse:.4f}", plot_data)
    fig.savefig(f"outputs_mlp/dir2_gan_to_vae_run_{num}.png"); plt.close(fig)
    print(f"Saved GAN->VAE plot to 'outputs_mlp/dir2_gan_to_vae_run_{num}.png'")

def plot_latent_space(model, model_type, device, output_path, n_images=20):
    """
    Generates a plot visualizing the latent space of a given model.
    """
    print(f"\n--- Generating Latent Space plot for {model_type.upper()} ---")
    grid_range = 3.5; x_points = np.linspace(-grid_range, grid_range, n_images); y_points = np.linspace(-grid_range, grid_range, n_images)
    canvas = np.empty((28 * n_images, 28 * n_images))
    with torch.no_grad():
        for i, y_coord in enumerate(tqdm(y_points, desc=f"Plotting {model_type.upper()} space")):
            for j, x_coord in enumerate(x_points):
                z = torch.tensor([[x_coord, y_coord]], device=device, dtype=torch.float)
                if model_type == 'vae': image = model.decode(z).view(28, 28).cpu().numpy()
                else: image = (model(z.view(1, 2, 1, 1)).squeeze() + 1) / 2.0; image = image.cpu().numpy()
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = image
    plt.figure(figsize=(12, 12)); plt.imshow(canvas, origin="upper", cmap="gray", extent=[-grid_range, grid_range, -grid_range, grid_range])
    plt.title(f"{model_type.upper()} Latent Space", fontsize=16); plt.xlabel("Latent Dimension 1"); plt.ylabel("Latent Dimension 2")
    plt.savefig(output_path); plt.close()
    print(f"Saved {model_type.upper()} latent space plot to '{output_path}'")


# ##############################################################################
# ### 5. MAIN EXECUTION SCRIPT ###
# ##############################################################################

def main():
    """Main function to orchestrate the entire analysis pipeline."""
    os.makedirs("outputs_mlp", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_experiments = 5

    # --- Step 1: Load Models ---
    vae_model, gan_generator = load_models(device)

    # --- Step 2: Create Paired Dataset & Export ---
    paired_vae_vectors, paired_gan_vectors = create_paired_latent_dataset(vae_model, gan_generator, device)
    print("\n--- Exporting paired latent vectors to text files ---")
    np.savetxt("outputs_mlp/paired_vae_vectors.txt", paired_vae_vectors, delimiter=",")
    np.savetxt("outputs_mlp/paired_gan_vectors.txt", paired_gan_vectors, delimiter=",")
    print("Vectors saved to 'outputs_mlp/paired_vae_vectors.txt' and 'outputs_mlp/paired_gan_vectors.txt'")

    # --- Step 3: Train Both MLP Bridges ---
    regressor_vae_to_gan = train_bridge(paired_vae_vectors, paired_gan_vectors, "VAE -> GAN")
    regressor_gan_to_vae = train_bridge(paired_gan_vectors, paired_vae_vectors, "GAN -> VAE")

    # --- Step 4: Prepare Data for Visualizations & Run Experiments ---
    vis_data = create_visualization_datasets(vae_model, gan_generator, device)
    test_dataset = datasets.MNIST(root='~/datasets', train=False, download=True, transform=transforms.ToTensor())
    
    for i in range(1, num_experiments + 1):
        seed = np.random.randint(0, 10000)
        # Run Direction 1: VAE -> GAN
        visualize_vae_to_gan(vae_model, gan_generator, regressor_vae_to_gan, device, vis_data, test_dataset, seed, i)
        # Run Direction 2: GAN -> VAE
        visualize_gan_to_vae(vae_model, gan_generator, regressor_gan_to_vae, device, vis_data, test_dataset, seed, i)

    # --- Step 5: Generate Final Latent Space Plots ---
    plot_latent_space(vae_model, 'vae', device, output_path="outputs_mlp/vae_latent_space.png")
    plot_latent_space(gan_generator, 'gan', device, output_path="outputs_mlp/gan_latent_space.png")

    print(f"\n\nAnalysis complete. All outputs saved in 'outputs_mlp/' directory.")

if __name__ == "__main__":
    main()