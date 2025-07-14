# ==============================================================================
#
#           VAE-GAN Latent Space Bridging and Analysis
#
# This script demonstrates a method to "translate" an image from the latent
# space of a Variational Autoencoder (VAE) to the latent space of a
# Generative Adversarial Network (GAN), and vice-versa. It includes detailed,
# presentation-ready visualizations of the process and the learned latent spaces.
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
from matplotlib.gridspec import GridSpec # For advanced, professional plot layouts
import matplotlib.cm as cm             # For creating solid-color colorbars

# Scikit-learn imports for a simple classifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# ##############################################################################
# ### 2. MODEL DEFINITIONS ###
# ##############################################################################

class VAE(nn.Module):
    """
    A simple Variational Autoencoder (VAE) for the MNIST dataset.
    It learns to compress images into a 2D latent space and then reconstruct them.
    This structure allows for encoding real images into a meaningful, low-dimensional space.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim_internal=200, latent_dim_output=2):
        super(VAE, self).__init__()
        # The Encoder network maps a high-dimensional input image to a lower-dimensional representation.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim_internal), nn.LeakyReLU(0.2)
        )
        # These layers produce the parameters (mean and log-variance) of the learned latent distribution.
        self.mean_layer = nn.Linear(latent_dim_internal, latent_dim_output)
        self.logvar_layer = nn.Linear(latent_dim_internal, latent_dim_output)
        
        # The Decoder network maps a point from the latent space back to a full, reconstructed image.
        self.decoder_nn = nn.Sequential(
            nn.Linear(latent_dim_output, latent_dim_internal), nn.LeakyReLU(0.2),
            nn.Linear(latent_dim_internal, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # Sigmoid activation constrains output pixels to the range [0, 1].
        )
    
    def encode(self, image_tensor):
        """Encodes an input image into its latent representation (mean and log-variance)."""
        hidden_representation = self.encoder(image_tensor.view(-1, 784))
        return self.mean_layer(hidden_representation), self.logvar_layer(hidden_representation)

    def decode(self, latent_vector):
        """Decodes a latent vector from the 2D space back into a reconstructed image."""
        return self.decoder_nn(latent_vector)

class Generator(nn.Module):
    """
    A DCGAN-style Generator model for the MNIST dataset.
    It learns to generate realistic images from a random 2D latent vector.
    """
    def __init__(self, latent_dim=2, num_channels=1, gen_features=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # A series of transposed convolutions progressively upsample the latent vector into a 28x28 image.
            nn.ConvTranspose2d(latent_dim, gen_features * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(gen_features * 4), nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 2), nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 2, gen_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features), nn.ReLU(True),
            nn.ConvTranspose2d(gen_features, num_channels, 3, 3, 4, bias=False),
            nn.Tanh() # Tanh activation constrains output pixels to the range [-1, 1].
        )
    def forward(self, latent_vector):
        """Generates an image from a given latent vector."""
        return self.main(latent_vector)


# ##############################################################################
# ### 3. DATA PREPARATION AND HELPER FUNCTIONS ###
# ##############################################################################

def load_models(device):
    """Loads pre-trained VAE and GAN Generator models from disk and sets them to evaluation mode."""
    print("--- Loading pre-trained VAE and GAN models ---")
    vae_model = VAE().to(device)
    vae_model.load_state_dict(torch.load('outputs_vae/vae_model_2d.pth', map_location=device))
    vae_model.eval()
    
    gan_generator = Generator().to(device)
    gan_generator.load_state_dict(torch.load('outputs_gan/weights/gan_generator_final.pth', map_location=device))
    gan_generator.eval()
    return vae_model, gan_generator

def create_latent_datasets(vae_model, gan_generator, device):
    """
    Generates latent vector datasets for both VAE and GAN spaces.
    The vae_centroids are essential for labeling the GAN's latent space.
    """
    print("\n--- Creating latent space datasets and centroids ---")
    
    # --- VAE Latent Space Creation ---
    mnist_train = datasets.MNIST(root='~/datasets', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=512, shuffle=False)
    
    with torch.no_grad():
        # Encode all training images to get their corresponding latent vectors (means)
        all_vae_vectors = np.concatenate([vae_model.encode(img.to(device))[0].cpu().numpy() for img, _ in tqdm(train_loader, desc="1/2: Generating VAE latent data")], axis=0)
    all_vae_labels = mnist_train.targets.numpy()
    
    # Calculate the average vector (centroid) for each digit class in the VAE space
    vae_centroids = np.array([all_vae_vectors[all_vae_labels == i].mean(axis=0) for i in range(10)])
    print("VAE latent dataset and centroids created.")

    # --- GAN Latent Space Creation (and Labeling) ---
    all_gan_vectors_list, all_gan_labels_list = [], []
    with torch.no_grad():
        for _ in tqdm(range(200), desc="2/2: Generating and labeling GAN latent data"):
            # Generate random latent vectors, which are the native input for the GAN
            random_gan_vectors = torch.randn(256, 2, 1, 1, device=device)
            # Use the GAN to generate images from these random vectors
            gan_images = gan_generator(random_gan_vectors)
            # Normalize GAN output from [-1, 1] to [0, 1] to be compatible with the VAE encoder
            gan_images_normalized = (gan_images + 1) / 2.0
            
            # Use the VAE to encode the GAN's generated images
            corresponding_vae_vectors, _ = vae_model.encode(gan_images_normalized)
            # Find the closest VAE centroid to each encoded vector
            distances = torch.cdist(corresponding_vae_vectors, torch.tensor(vae_centroids, device=device, dtype=torch.float))
            # The label for the GAN's random vector is the class of the closest VAE centroid
            predicted_labels = torch.argmin(distances, dim=1)
            
            all_gan_vectors_list.append(random_gan_vectors.squeeze().cpu().numpy())
            all_gan_labels_list.append(predicted_labels.cpu().numpy())
    
    all_gan_vectors = np.concatenate(all_gan_vectors_list, axis=0)
    all_gan_labels = np.concatenate(all_gan_labels_list, axis=0)
    # Calculate centroids for the newly labeled GAN space
    gan_centroids = np.array([all_gan_vectors[all_gan_labels == i].mean(axis=0) for i in range(10)])
    print("GAN latent dataset and centroids created.")
    
    return (all_vae_vectors, all_vae_labels, vae_centroids), (all_gan_vectors, all_gan_labels, gan_centroids)

def train_classifier(latent_vectors, labels, model_name="Model"):
    """Trains a simple MLP classifier to predict digit class from a 2D latent vector."""
    print(f"\n--- Training MLP Classifier for {model_name} Space ---")
    classifier = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', max_iter=500, random_state=42, early_stopping=True)
    classifier.fit(latent_vectors, labels)
    print(f"{model_name} Classifier trained. Final Accuracy: {accuracy_score(labels, classifier.predict(latent_vectors)):.2f}")
    return classifier


# ##############################################################################
# ### 4. VISUALIZATION FUNCTIONS ###
# ##############################################################################

def _get_discrete_cmap():
    """Helper function to create a discrete colormap for digits 0-9."""
    # Note: A MatplotlibDeprecationWarning for get_cmap is expected and can be ignored.
    return plt.cm.get_cmap('tab10', 10)

def _create_bridge_plot(fig_title, plot_data):
    """
    A master helper function to generate the 2x2 bridge plot using GridSpec for a professional layout.
    This function handles the complex plotting logic to keep the main visualization functions clean.
    """
    fig = plt.figure(figsize=(15, 14))
    fig.suptitle(fig_title, fontsize=18)

    # Use GridSpec for advanced layout control
    gs_main = GridSpec(2, 2, figure=fig, height_ratios=[0.8, 1])
    ax_img1 = fig.add_subplot(gs_main[0, 0])
    ax_img2 = fig.add_subplot(gs_main[0, 1])
    
    # Create a nested GridSpec for the bottom row to include a dedicated column for the colorbar
    gs_bottom = gs_main[1, :].subgridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    ax_ls1 = fig.add_subplot(gs_bottom[0, 0])
    ax_ls2 = fig.add_subplot(gs_bottom[0, 1])
    colorbar_axis = fig.add_subplot(gs_bottom[0, 2])

    ax_img1.imshow(plot_data['img1'], cmap='gray'); ax_img1.set_title(plot_data['title1'], fontsize=12); ax_img1.axis('off')
    ax_img2.imshow(plot_data['img2'], cmap='gray'); ax_img2.set_title(plot_data['title2'], fontsize=12); ax_img2.axis('off')
    
    cmap = _get_discrete_cmap()
    # alpha_val=1 makes the scatter plot points fully opaque.
    # A value < 1 would make them semi-transparent to show density.
    alpha_val = 1
    point_size = 15
    
    # Plot first latent space (bottom-left)
    ax_ls1.scatter(plot_data['X1'][:, 0], plot_data['X1'][:, 1], c=plot_data['y1'], cmap=cmap, alpha=alpha_val, s=point_size)
    ax_ls1.scatter(plot_data['vec1'][:, 0], plot_data['vec1'][:, 1], c='black', marker='*', s=350, edgecolor='white', label=plot_data['label1'])
    ax_ls1.set_title(plot_data['title_ls1'], fontsize=12)
    ax_ls1.set_xlabel("Latent Dim 1"); ax_ls1.set_ylabel("Latent Dim 2")
    ax_ls1.legend(loc='upper right')

    # Plot second latent space (bottom-right)
    scatter_plot_artist = ax_ls2.scatter(plot_data['X2'][:, 0], plot_data['X2'][:, 1], c=plot_data['y2'], cmap=cmap, alpha=alpha_val, s=point_size)
    ax_ls2.scatter(plot_data['vec2'][:, 0], plot_data['vec2'][:, 1], c='black', marker='*', s=350, edgecolor='white', label=plot_data['label2'])
    ax_ls2.set_title(plot_data['title_ls2'], fontsize=12)
    ax_ls2.set_xlabel("Latent Dim 1"); ax_ls2.set_ylabel("Latent Dim 2")
    ax_ls2.legend(loc='upper right')
    
    # The colorbar is created directly from the scatter plot object.
    # As alpha_val=1, the colorbar will be fully opaque.
    fig.colorbar(scatter_plot_artist, cax=colorbar_axis, label='Digit Class', ticks=np.arange(10))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def visualize_vae_to_gan(vae_model, gan_generator, vae_classifier, device, data_tuple, test_dataset, random_seed, experiment_num):
    """Performs the VAE->GAN translation for a RANDOM image and prepares data for plotting."""
    X_vae, y_vae, _, X_gan, y_gan, _ = data_tuple
    print(f"\n--- Running VAE to GAN (Experiment #{experiment_num}) ---")

    np.random.seed(random_seed)
    random_idx = np.random.randint(0, len(test_dataset))
    image_tensor, true_label = test_dataset[random_idx] 
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        z_vae_original, _ = vae_model.encode(image_tensor)
        z_vae_original_np = z_vae_original.cpu().numpy()

    predicted_label = vae_classifier.predict(z_vae_original_np)[0]
    gan_vectors_for_label = X_gan[y_gan == predicted_label]
    gan_start_vector_np = gan_vectors_for_label[np.random.randint(0, len(gan_vectors_for_label))]
    
    with torch.no_grad():
        final_gan_image = gan_generator(torch.tensor(gan_start_vector_np, device=device, dtype=torch.float).view(1, 2, 1, 1)).squeeze().cpu().numpy()

    mse = np.mean(((image_tensor.cpu().squeeze().numpy()*2-1) - final_gan_image)**2)
    
    plot_data = {
        'img1': image_tensor.cpu().squeeze(), 'img2': final_gan_image,
        'title1': f"1. A real image is passed to the VAE.\nVector: ({z_vae_original_np[0,0]:.2f}, {z_vae_original_np[0,1]:.2f})",
        'title2': f"4. GAN generates a new image from its vector.\nVector: ({gan_start_vector_np[0]:.2f}, {gan_start_vector_np[1]:.2f})",
        'X1': X_vae, 'y1': y_vae, 'vec1': z_vae_original_np, 'label1': f'Start Vector (Classifier says: {predicted_label})',
        'X2': X_gan, 'y2': y_gan, 'vec2': gan_start_vector_np.reshape(1, 2), 'label2': 'End Vector',
        'title_ls1': "2. VAE maps the image to a point in its space.",
        'title_ls2': f"3. A random point for digit '{predicted_label}' is chosen in GAN space."
    }
    
    fig_title = f"Experiment 1 (Run {experiment_num}): VAE -> GAN Bridge\nGoal: Project a real image (True Label: {true_label}) | Result: Translation MSE={mse:.4f}"
    fig = _create_bridge_plot(fig_title, plot_data)
    
    output_filename = f"outputs_mlp/vae_to_gan_run_{experiment_num}.png"
    fig.savefig(output_filename)
    plt.close(fig)
    print(f"Saved VAE->GAN bridge plot to '{output_filename}'")


def visualize_gan_to_vae(vae_model, gan_generator, device, data_tuple, random_seed, experiment_num):
    """Performs the GAN->VAE translation, starting from a RANDOM point in the GAN's latent space."""
    X_vae, y_vae, _, X_gan, y_gan, gan_centroids = data_tuple # Use gan_centroids only for the random selection
    print(f"\n--- Running GAN to VAE (Experiment #{experiment_num}) ---")

    np.random.seed(random_seed)
    # Instead of using a centroid, pick a random class and then a random vector for that class.
    digit_to_generate = np.random.randint(0, 10)
    gan_vectors_for_label = X_gan[y_gan == digit_to_generate]
    gan_start_vector_np = gan_vectors_for_label[np.random.randint(0, len(gan_vectors_for_label))]
    
    with torch.no_grad():
        initial_gan_image = gan_generator(torch.tensor(gan_start_vector_np, device=device, dtype=torch.float).view(1, 2, 1, 1))
        gan_image_for_vae = (initial_gan_image + 1) / 2.0
        z_vae_end_vector, _ = vae_model.encode(gan_image_for_vae)
        final_vae_image_tensor = vae_model.decode(z_vae_end_vector)

    mse = torch.mean((gan_image_for_vae.squeeze() - final_vae_image_tensor.squeeze().view(1, 28, 28))**2).item()
    z_vae_end_vector_np = z_vae_end_vector.cpu().numpy()

    plot_data = {
        'img1': initial_gan_image.cpu().squeeze(),
        'img2': final_vae_image_tensor.cpu().squeeze().view(28, 28).numpy(),
        'title1': f"1. A GAN image is generated from a random point.\nVector: ({gan_start_vector_np[0]:.2f}, {gan_start_vector_np[1]:.2f})",
        'title2': f"4. VAE reconstructs the GAN image.\nVector: ({z_vae_end_vector_np[0,0]:.2f}, {z_vae_end_vector_np[0,1]:.2f})",
        'X1': X_gan, 'y1': y_gan, 'vec1': gan_start_vector_np.reshape(1, 2), 'label1': 'Start Vector',
        'X2': X_vae, 'y2': y_vae, 'vec2': z_vae_end_vector_np, 'label2': 'End Vector',
        'title_ls1': f"2. Start from a random point for digit '{digit_to_generate}' in GAN space.",
        'title_ls2': "3. VAE maps the GAN image to a point in its space."
    }
    fig_title = f"Experiment 2 (Run {experiment_num}): GAN -> VAE Bridge\nGoal: See how VAE reconstructs a random GAN digit '{digit_to_generate}' | Result: Reconstruction MSE={mse:.4f}"
    fig = _create_bridge_plot(fig_title, plot_data)

    output_filename = f"outputs_mlp/gan_to_vae_run_{experiment_num}.png"
    fig.savefig(output_filename)
    plt.close(fig)
    print(f"Saved GAN->VAE bridge plot to '{output_filename}'")


def plot_latent_space(model, model_type, device, output_path, n_images=20):
    """Creates a latent space plot by sampling a grid of points to visualize what the model has learned."""
    print(f"\n--- Generating Latent Space plot for {model_type.upper()} ---")
    grid_range = 3.5
    x_points = np.linspace(-grid_range, grid_range, n_images)
    y_points = np.linspace(-grid_range, grid_range, n_images)
    canvas = np.empty((28 * n_images, 28 * n_images))
    with torch.no_grad():
        for i, y_coord in enumerate(tqdm(y_points, desc=f"Plotting {model_type.upper()} latent space")):
            for j, x_coord in enumerate(x_points):
                z = torch.tensor([[x_coord, y_coord]], device=device, dtype=torch.float)
                if model_type == 'vae':
                    image = model.decode(z).view(28, 28).cpu().numpy()
                elif model_type == 'gan':
                    image = (model(z.view(1, 2, 1, 1)).squeeze() + 1) / 2.0
                    image = image.cpu().numpy()
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = image
    plt.figure(figsize=(12, 12))
    plt.imshow(canvas, origin="upper", cmap="gray", extent=[-grid_range, grid_range, -grid_range, grid_range])
    plt.title(f"{model_type.upper()} Latent Space", fontsize=16)
    plt.xlabel("Latent Dimension 1"); plt.ylabel("Latent Dimension 2")
    plt.savefig(output_path)
    print(f"Saved {model_type.upper()} latent space plot to '{output_path}'")
    plt.close()


# ##############################################################################
# ### 5. MAIN EXECUTION SCRIPT ###
# ##############################################################################

def main():
    """Main function to run the entire analysis pipeline."""
    # --- Setup ---
    os.makedirs("outputs_mlp", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Configuration ---
    num_experiments = 3 # Set how many examples to generate for each bridge
    
    # --- Step 1: Load Models ---
    vae_model, gan_generator = load_models(device)
    
    # --- Step 2: Create Latent Datasets ---
    (X_vae, y_vae, vae_centroids), (X_gan, y_gan, gan_centroids) = create_latent_datasets(vae_model, gan_generator, device)
    
    # --- Step 3: Train Classifier for Bridging ---
    vae_classifier = train_classifier(X_vae, y_vae, model_name="VAE")
    
    # --- Step 4: Run Experiments and Visualize ---
    test_dataset = datasets.MNIST(root='~/datasets', train=False, download=True, transform=transforms.ToTensor())
    data_tuple = (X_vae, y_vae, vae_centroids, X_gan, y_gan, gan_centroids)
    
    # Generate multiple, unique examples for each experiment
    for i in range(1, num_experiments + 1):
        # Generate new random seeds for each run to ensure different results
        seed1 = np.random.randint(0, 10000)
        seed2 = np.random.randint(0, 10000)
        
        visualize_vae_to_gan(vae_model, gan_generator, vae_classifier, device, data_tuple, test_dataset, random_seed=seed1, experiment_num=i)
        visualize_gan_to_vae(vae_model, gan_generator, device, data_tuple, random_seed=seed2, experiment_num=i)
    
    # --- Step 5: Generate Latent Space Plots for Deeper Insight ---
    plot_latent_space(vae_model, 'vae', device, output_path="outputs_mlp/vae_latent_space.png")
    plot_latent_space(gan_generator, 'gan', device, output_path="outputs_mlp/gan_latent_space.png")

    print(f"\n\nAnalysis complete. {num_experiments} examples for each experiment saved in 'outputs_mlp/' directory.")


if __name__ == "__main__":
    main()