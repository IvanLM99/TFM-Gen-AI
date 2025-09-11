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
#  - CNN in Visuals: Plots show the CNN's predicted label for projected images.
#
# ==============================================================================


# ##############################################################################
# ### 1. IMPORTS ###
# ##############################################################################
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime


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

    def encode(self, x):
        """Passes an image through the encoder to get the latent distribution parameters."""
        h = self.encoder(x.view(-1, 784))
        return self.mean_layer(h), self.logvar_layer(h)

    def decode(self, z):
        """Passes a latent vector through the decoder to reconstruct an image."""
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
    def forward(self, z):
        """Generates an image from a latent vector z."""
        return self.main(z)

class CNNClassifier(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST classification.
    Used to evaluate the classifiability of generated/projected images.
    """
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Normalize input to have a mean of 0 and std of 1, as expected by many classifiers
        x = transforms.functional.normalize(x, (0.1307,), (0.3081,))
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


# ##############################################################################
# ### 3. SETUP, TRAINING, AND DATA PREPARATION ###
# ##############################################################################

def setup_logging(log_dir):
    """Configures logging to print to console and save to a timestamped file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f'experiment_log_{timestamp}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return log_file

def filter_mnist_by_digit(dataset, digit):
    """
    Filters a PyTorch MNIST dataset to include only a specific digit.
    
    Args:
        dataset (torch.utils.data.Dataset): The original MNIST dataset.
        digit (int): The digit (0-9) to keep.
        
    Returns:
        torch.utils.data.TensorDataset: A new dataset containing only the specified digit.
    """
    # Find indices of all samples with the target digit
    idx = (dataset.targets == digit)
    
    # Filter the data and targets using the indices
    filtered_data = dataset.data[idx]
    filtered_targets = dataset.targets[idx]
    
    # The VAE expects data in the range [0, 1] as float tensors.
    # The original dataset.data is uint8 [0, 255].
    filtered_data = filtered_data.float() / 255.0

    return TensorDataset(filtered_data.unsqueeze(1), filtered_targets) # Add channel dim

def load_models(device):
    """
    Loads the pre-trained VAE and GAN Generator models from disk.
    Args:
        device (torch.device): The device (CPU or CUDA) to load the models onto.
    Returns:
        A tuple containing the loaded (vae_model, gan_generator).
    """
    logging.info("--- Loading pre-trained VAE and GAN models ---")
    try:
        vae_model = VAE().to(device); vae_model.load_state_dict(torch.load('outputs_vae/vae_model_2d.pth', map_location=device)); vae_model.eval()
        gan_generator = Generator().to(device); gan_generator.load_state_dict(torch.load('outputs_gan/weights/gan_generator_final.pth', map_location=device)); gan_generator.eval()
    except FileNotFoundError as e:
        logging.error(f"FATAL: Could not find a model file. Make sure pre-trained models exist.\nDetails: {e}"); exit()
    return vae_model, gan_generator

def create_paired_latent_dataset(vae_model, gan_generator, device, num_samples=75000):
    """
    Generates a paired (z_vae, z_gan) dataset for training the MLP bridges.
    Args:
        vae_model (nn.Module): The pre-trained VAE.
        gan_generator (nn.Module): The pre-trained GAN Generator.
        device (torch.device): The device to run computations on.
        num_samples (int): The number of vector pairs to generate.
    Returns:
        A tuple with train/test splits: ((train_vae, train_gan), (test_vae, test_gan)).
    """
    logging.info("\n--- Creating Paired Latent Dataset for Training Bridges ---")
    paired_vae_vectors, paired_gan_vectors = [], []
    batch_size = 512
    with torch.no_grad():
        for _ in tqdm(range(int(np.ceil(num_samples / batch_size))), desc="Generating paired (VAE, GAN) vectors"):
            z_gan = torch.randn(batch_size, 2, 1, 1, device=device)
            gan_images = gan_generator(z_gan)
            gan_images_normalized = (gan_images + 1) / 2.0
            z_vae, _ = vae_model.encode(gan_images_normalized)
            paired_gan_vectors.append(z_gan.squeeze().cpu().numpy())
            paired_vae_vectors.append(z_vae.cpu().numpy())

    all_vae = np.concatenate(paired_vae_vectors, axis=0)[:num_samples]
    all_gan = np.concatenate(paired_gan_vectors, axis=0)[:num_samples]
    
    X_vae_train, X_vae_test, X_gan_train, X_gan_test = train_test_split(
        all_vae, all_gan, test_size=0.2, random_state=42
    )
    logging.info(f"Dataset split into {len(X_vae_train)} training and {len(X_vae_test)} test samples.")
    return (X_vae_train, X_gan_train), (X_vae_test, X_gan_test)

def create_paired_latent_dataset_for_digit(vae_model, gan_generator, classifier, device, digit, num_samples=10000):
    """
    Generates a paired (z_vae, z_gan) dataset for a single digit.
    
    This is done by generating images from the GAN, classifying them, and
    only keeping the latent vectors of the images that match the target digit.
    """
    logging.info(f"\n--- Creating Paired Latent Dataset for ONLY Digit '{digit}' ---")
    paired_vae_vectors, paired_gan_vectors = [], []
    batch_size = 512
    pbar = tqdm(total=num_samples, desc=f"Finding GAN vectors for digit '{digit}'")

    with torch.no_grad():
        while len(paired_vae_vectors) < num_samples:
            z_gan = torch.randn(batch_size, 2, 1, 1, device=device)
            gan_images = gan_generator(z_gan) # Range [-1, 1]
            
            # Use the CNN to classify the generated images
            # Normalize GAN images from [-1, 1] to [0, 1] for VAE and CNN
            gan_images_normalized = (gan_images + 1) / 2.0
            predictions = classifier(gan_images_normalized).argmax(dim=1)
            
            # Find the indices where the prediction matches our target digit
            mask = (predictions == digit)
            
            if mask.sum() > 0:
                # Select the z_gan vectors that produced the correct digit
                gan_vectors_for_digit = z_gan[mask]
                
                # Select the corresponding images to encode with the VAE
                images_for_digit = gan_images_normalized[mask]
                
                # Get the corresponding z_vae vectors
                z_vae, _ = vae_model.encode(images_for_digit)
                
                # Append the numpy arrays to our lists
                paired_gan_vectors.extend(gan_vectors_for_digit.squeeze().cpu().numpy())
                paired_vae_vectors.extend(z_vae.cpu().numpy())
                
                pbar.update(len(gan_vectors_for_digit))

    pbar.close()

    # Trim to the exact number of samples
    all_vae = np.array(paired_vae_vectors[:num_samples])
    all_gan = np.array(paired_gan_vectors[:num_samples])
    
    X_vae_train, X_vae_test, X_gan_train, X_gan_test = train_test_split(
        all_vae, all_gan, test_size=0.2, random_state=42
    )
    logging.info(f"Dataset for digit '{digit}' split into {len(X_vae_train)} training and {len(X_vae_test)} test samples.")
    return (X_vae_train, X_gan_train), (X_vae_test, X_gan_test)

def train_bridge_regressor(X_train, Y_train, direction, hidden_layer_sizes, activation, max_iter=1000):
    """
    Trains an MLP Regressor with a specific architecture and activation.
    Args:
        X_train (np.array): The input vectors (source space).
        Y_train (np.array): The target vectors (destination space).
        direction (str): Description of the projection direction for logging.
        hidden_layer_sizes (tuple): Defines the size of the hidden layers.
        activation (str): 'identity', 'relu', etc.
    Returns:
        The trained scikit-learn MLPRegressor model.
    """
    logging.info(f"--- Training MLP Bridge for {direction} ---")
    logging.info(f"    Architecture: hidden_layer_sizes={hidden_layer_sizes}, activation='{activation}'")
    
    regressor = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver='adam',
        max_iter=max_iter, random_state=42, early_stopping=True, verbose=False,
        n_iter_no_change=15
    )
    
    logging.info("    Fitting regressor... (This may take a minute)")
    regressor.fit(X_train, Y_train)
    mse = mean_squared_error(Y_train, regressor.predict(X_train))
    logging.info(f"    MLP Regressor for {direction} trained. Final Training MSE on vectors: {mse:.6f}")
    return regressor

def gan_inversion(gan_generator, target_image, device, steps=2500, lr=0.01):
    """Finds the GAN latent vector 'z' that best reconstructs a target image."""
    target_image_gan_range = (target_image.to(device) * 2 - 1).unsqueeze(0)
    z = torch.randn(1, 2, 1, 1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(gan_generator(z), target_image_gan_range)
        loss.backward()
        optimizer.step()
    return z.detach()

def create_visualization_datasets(vae_model, gan_generator, device):
    """Generates labeled latent vectors for plotting the background of the spaces."""
    logging.info("\n--- Creating latent space datasets for visualization plots ---")
    mnist_train = datasets.MNIST(root='~/datasets', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=512, shuffle=False)
    with torch.no_grad():
        all_vae_vectors = np.concatenate([vae_model.encode(img.to(device))[0].cpu().numpy() for img, _ in tqdm(train_loader, desc="1/2: Visualizing VAE space")], axis=0)
    all_vae_labels = mnist_train.targets.numpy()
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

def train_cnn_classifier(device, epochs=5, batch_size=64, lr=0.001):
    """Trains the CNNClassifier on MNIST or loads it if already trained."""
    classifier_path = 'outputs_mlp/cnn_classifier.pth'
    if os.path.exists(classifier_path):
        logging.info("\n--- Loading pre-trained CNN Classifier ---")
        model = CNNClassifier().to(device); model.load_state_dict(torch.load(classifier_path, map_location=device)); model.eval()
        logging.info("    CNN Classifier loaded successfully.")
        return model
    logging.info("\n--- Training new CNN Classifier for Evaluation ---")
    model = CNNClassifier().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss_fn = nn.NLLLoss(); test_loss_fn = nn.NLLLoss(reduction='sum')
    train_loader = DataLoader(datasets.MNIST('~/datasets', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('~/datasets', train=False, download=True, transform=transforms.ToTensor()), batch_size=1000, shuffle=False)
    for epoch in range(1, epochs + 1):
        model.train()
        for _, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(); output = model(data); loss = train_loss_fn(output, target); loss.backward(); optimizer.step()
        model.eval(); test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device); output = model(data)
                test_loss += test_loss_fn(output, target).item(); pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset); accuracy = 100. * correct / len(test_loader.dataset)
        logging.info(f'    Epoch {epoch} Test Set: Avg loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    torch.save(model.state_dict(), classifier_path)
    logging.info(f"    CNN Classifier trained and saved to '{classifier_path}'")
    model.eval()
    return model

def evaluate_projection_quality(direction, vae_model, gan_generator, regressor, classifier, device, test_dataset, paired_test_data, num_eval_samples):
    """Evaluates projection quality via classification accuracy and reconstruction error."""
    logging.info(f"--- Evaluating Quantitative Quality for {direction} ---")
    mses, correct, total = [], 0, 0
    with torch.no_grad():
        if direction == 'VAE->GAN':
            loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            iterator = tqdm(loader, total=num_eval_samples, desc=f"Evaluating {direction}")
            for i, (img, label) in enumerate(iterator):
                if i >= num_eval_samples: break
                img, label = img.to(device), label.to(device)
                z_vae, _ = vae_model.encode(img)
                pred_gan_vec = torch.tensor(regressor.predict(z_vae.cpu().numpy()), device=device, dtype=torch.float).view(1, 2, 1, 1)
                final_img = (gan_generator(pred_gan_vec) + 1) / 2.0
                if classifier(final_img).argmax().item() == label.item(): correct += 1
                mses.append(mean_squared_error(img.cpu().numpy().flatten(), final_img.cpu().numpy().flatten()))
                total += 1
        else: # GAN->VAE
            _, X_gan_test = paired_test_data; num_eval_samples = min(num_eval_samples, len(X_gan_test))
            iterator = tqdm(range(num_eval_samples), desc=f"Evaluating {direction}")
            for i in iterator:
                z_gan_np = X_gan_test[i].reshape(1, 2)
                z_gan_tensor = torch.tensor(z_gan_np, device=device, dtype=torch.float).view(1, 2, 1, 1)
                orig_img = (gan_generator(z_gan_tensor) + 1) / 2.0
                orig_label = classifier(orig_img).argmax().item()
                pred_vae_vec = torch.tensor(regressor.predict(z_gan_np), device=device, dtype=torch.float)
                final_img = vae_model.decode(pred_vae_vec).view(1, 1, 28, 28)
                if classifier(final_img).argmax().item() == orig_label: correct += 1
                mses.append(mean_squared_error(orig_img.cpu().numpy().flatten(), final_img.cpu().numpy().flatten()))
                total += 1
    
    accuracy = 100. * correct / total
    logging.info(f"--- Quality Report for {direction} (Config: {regressor.activation}, Layers: {regressor.hidden_layer_sizes}) ---")
    logging.info(f"    - Evaluated on {total} samples.")
    logging.info(f"    - Projected Image Classification Accuracy: {accuracy:.2f}%")
    logging.info(f"    - Image Reconstruction MSE (Median): {np.median(mses):.6f} (+/- {np.std(mses):.6f})")

# ##############################################################################
# ### 4. VISUALIZATION FUNCTIONS ###
# ##############################################################################

def _create_plot(fig_title, plot_data, num_top_images=2):
    """
    Helper to generate the projection plot, supporting 2 or 3 top images.
    A 3-image layout shows the intermediate reconstruction from the source model.
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(fig_title, fontsize=18, y=0.98)

    if num_top_images == 3:
        # Layout for 3 images on top (Original -> Reconstruction -> Final Projection)
        gs_main = GridSpec(2, 3, figure=fig, height_ratios=[0.7, 1])
        ax_img1 = fig.add_subplot(gs_main[0, 0])
        ax_img_middle = fig.add_subplot(gs_main[0, 1]) # Middle image for reconstruction
        ax_img2 = fig.add_subplot(gs_main[0, 2])
        
        ax_img1.imshow(plot_data['img1'], cmap='gray'); ax_img1.set_title(plot_data['title1'], fontsize=12); ax_img1.axis('off')
        ax_img_middle.imshow(plot_data['img_middle'], cmap='gray'); ax_img_middle.set_title(plot_data['title_middle'], fontsize=12); ax_img_middle.axis('off')
        ax_img2.imshow(plot_data['img2'], cmap='gray'); ax_img2.set_title(plot_data['title2'], fontsize=12); ax_img2.axis('off')
        
        gs_bottom_span = gs_main[1, :]
    else: # Fallback to original 2-image layout if needed
        gs_main = GridSpec(2, 2, figure=fig, height_ratios=[0.8, 1])
        ax_img1 = fig.add_subplot(gs_main[0, 0])
        ax_img2 = fig.add_subplot(gs_main[0, 1])

        ax_img1.imshow(plot_data['img1'], cmap='gray'); ax_img1.set_title(plot_data['title1'], fontsize=12); ax_img1.axis('off')
        ax_img2.imshow(plot_data['img2'], cmap='gray'); ax_img2.set_title(plot_data['title2'], fontsize=12); ax_img2.axis('off')
        
        gs_bottom_span = gs_main[1, :]

    # Common code for bottom row (latent spaces and colorbar)
    gs_bottom = gs_bottom_span.subgridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    ax_ls1 = fig.add_subplot(gs_bottom[0, 0])
    ax_ls2 = fig.add_subplot(gs_bottom[0, 1])
    colorbar_axis = fig.add_subplot(gs_bottom[0, 2])

    cmap = plt.get_cmap('tab10', 10); alpha, s = 0.5, 15
    ax_ls1.scatter(plot_data['X1'][:, 0], plot_data['X1'][:, 1], c=plot_data['y1'], cmap=cmap, alpha=alpha, s=s)
    ax_ls1.scatter(plot_data['vec1'][:, 0], plot_data['vec1'][:, 1], c='k', marker='*', s=350, edgecolors='w', label=plot_data['label1'])
    ax_ls1.set_title(plot_data['title_ls1'], fontsize=12); ax_ls1.set_xlabel("Latent Dim 1"); ax_ls1.set_ylabel("Latent Dim 2"); ax_ls1.legend(loc='upper right')

    scatter_artist = ax_ls2.scatter(plot_data['X2'][:, 0], plot_data['X2'][:, 1], c=plot_data['y2'], cmap=cmap, alpha=alpha, s=s)
    ax_ls2.scatter(plot_data['vec2'][:, 0], plot_data['vec2'][:, 1], c='k', marker='*', s=350, edgecolors='w', label=plot_data['label2'])
    ax_ls2.set_title(plot_data['title_ls2'], fontsize=12); ax_ls2.set_xlabel("Latent Dim 1"); ax_ls2.set_ylabel("Latent Dim 2"); ax_ls2.legend(loc='upper right')

    fig.colorbar(scatter_artist, cax=colorbar_axis, label='Digit Class', ticks=np.arange(10))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def visualize_vae_to_gan(vae_model, gan_generator, regressor, classifier, device, vis_data, test_dataset, seed, num, config_tag, output_dir):
    """Visualizes the VAE -> GAN pipeline, INCLUDING the VAE's own reconstruction."""
    (X_vae, y_vae), (X_gan, y_gan) = vis_data
    np.random.seed(seed); random_idx = np.random.randint(0, len(test_dataset))
    img, label = test_dataset[random_idx]

    with torch.no_grad():
        z_vae, _ = vae_model.encode(img.to(device)); z_vae_np = z_vae.cpu().numpy()
        
        # --- NEW: Generate the VAE's reconstruction of the original image ---
        vae_reconstructed_img_tensor = vae_model.decode(z_vae).view(1, 1, 28, 28)
        vae_reconstructed_img_np = vae_reconstructed_img_tensor.squeeze().cpu().numpy()
        
        # Project to GAN space
        pred_gan_vec_np = regressor.predict(z_vae_np)
        pred_gan_tensor = torch.tensor(pred_gan_vec_np, device=device, dtype=torch.float).view(1, 2, 1, 1)
        final_img_tensor = (gan_generator(pred_gan_tensor) + 1) / 2.0
        final_img_np = final_img_tensor.squeeze().cpu().numpy()
        pred_label = classifier(final_img_tensor).argmax(dim=1).item()

    # Calculate two MSEs for better analysis
    mse_reconstruction = mean_squared_error(img.squeeze().numpy(), vae_reconstructed_img_np)
    mse_projection = mean_squared_error(vae_reconstructed_img_np, final_img_np)

    plot_data = {
        'img1': img.squeeze(),
        'img_middle': vae_reconstructed_img_np, # The VAE's reconstruction
        'img2': final_img_np,
        
        'title1': f"1. Original Real Image\n(True Class: {label})",
        'title_middle': f"2. VAE Reconstruction\n(MSE vs Original: {mse_reconstruction:.4f})",
        'title2': f"5. Final Projected Image\n(Predicted Class: {pred_label})",
        
        'X1': X_vae, 'y1': y_vae, 'vec1': z_vae_np, 'label1': 'Initial VAE Vector',
        'X2': X_gan, 'y2': y_gan, 'vec2': pred_gan_vec_np, 'label2': 'Predicted GAN Vector',
        
        'title_ls1': "3. VAE encodes image to get z_vae",
        'title_ls2': "4. MLP predicts z_gan from z_vae"
    }
    
    fig_title = (f"VAE -> GAN Projection (Config: {config_tag})\n"
                 f"Projection MSE (VAE Recon -> GAN Final) = {mse_projection:.4f}")

    # Call the plot helper with num_top_images=3
    fig = _create_plot(fig_title, plot_data, num_top_images=3)
    
    save_path = os.path.join(output_dir, f"vis_vae_to_gan_run_{num}.png")
    fig.savefig(save_path); plt.close(fig)


def visualize_gan_to_vae(vae_model, gan_generator, regressor, classifier, device, vis_data, test_dataset, seed, num, config_tag, output_dir):
    """Visualizes the GAN -> VAE pipeline, INCLUDING the GAN inversion result."""
    (X_vae, y_vae), (X_gan, y_gan) = vis_data
    np.random.seed(seed); random_idx = np.random.randint(0, len(test_dataset))
    img, label = test_dataset[random_idx]

    logging.info(f"    Performing GAN Inversion for visualization #{num}...")
    z_gan_inv = gan_inversion(gan_generator, img, device); z_gan_inv_np = z_gan_inv.squeeze().cpu().numpy().reshape(1, 2)

    with torch.no_grad():
        # --- NEW: Generate the reconstructed image from the inverted vector ---
        reconstructed_img_tensor = (gan_generator(z_gan_inv) + 1) / 2.0
        reconstructed_img_np = reconstructed_img_tensor.squeeze().cpu().numpy()
        
        # Project to VAE space
        pred_vae_vec_np = regressor.predict(z_gan_inv_np)
        pred_vae_tensor = torch.tensor(pred_vae_vec_np, device=device, dtype=torch.float)
        final_img_tensor = vae_model.decode(pred_vae_tensor).view(1, 1, 28, 28)
        final_img_np = final_img_tensor.squeeze().cpu().numpy()
        pred_label = classifier(final_img_tensor).argmax(dim=1).item()

    # Calculate two MSEs for better analysis
    mse_inversion = mean_squared_error(img.squeeze().numpy(), reconstructed_img_np)
    mse_projection = mean_squared_error(reconstructed_img_np, final_img_np)

    plot_data = {
        'img1': img.squeeze(),
        'img_middle': reconstructed_img_np, # The GAN's reconstruction
        'img2': final_img_np,
        
        'title1': f"1. Original Real Image\n(True Class: {label})",
        'title_middle': f"2. GAN Inversion Result\n(MSE vs Original: {mse_inversion:.4f})",
        'title2': f"5. Final Projected Image\n(Predicted Class: {pred_label})",
        
        'X1': X_gan, 'y1': y_gan, 'vec1': z_gan_inv_np, 'label1': 'Inverted GAN Vector',
        'X2': X_vae, 'y2': y_vae, 'vec2': pred_vae_vec_np, 'label2': 'Predicted VAE Vector',
        
        'title_ls1': "3. GAN Inversion finds z_gan",
        'title_ls2': "4. MLP predicts z_vae from z_gan"
    }
    
    fig_title = (f"GAN -> VAE Projection (Config: {config_tag})\n"
                 f"Projection MSE (GAN Recon -> VAE Final) = {mse_projection:.4f}")
    
    # Call the plot helper with num_top_images=3
    fig = _create_plot(fig_title, plot_data, num_top_images=3) 
    
    save_path = os.path.join(output_dir, f"vis_gan_to_vae_run_{num}.png")
    fig.savefig(save_path); plt.close(fig)
def plot_latent_space(model, model_type, device, output_path, n_images=20):
    """Generates a plot visualizing the latent space of a given model."""
    logging.info(f"\n--- Generating Latent Space plot for {model_type.upper()} ---")
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
    logging.info(f"    Saved {model_type.upper()} latent space plot to '{output_path}'")


# ##############################################################################
# ### 5. EXPERIMENT ORCHESTRATION ###
# ##############################################################################

def run_experiment(exp_name, config_tag, output_dir, bridge_params, common_data):
    """
    A helper function to run a full experimental configuration.
    This includes training bridges, creating visualizations, and evaluating quality.
    """
    logging.info("\n\n" + "="*80)
    logging.info(f"### {exp_name.upper()} ###")
    logging.info("="*80 + "\n")
    
    # Unpack common data
    vae_model, gan_generator, cnn_classifier, device = common_data['models']
    paired_train_data, paired_test_data = common_data['paired_data']
    vis_data, test_dataset = common_data['vis_data']
    NUM_VIS_EXP, NUM_EVAL_SAMPLES = common_data['constants']

    # Create dedicated output directory for this experiment
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Outputs for this experiment will be saved in: '{output_dir}'")

    # Train the bridges for this configuration
    reg_v2g = train_bridge_regressor(
        paired_train_data[0], paired_train_data[1], f"VAE -> GAN ({config_tag})", **bridge_params
    )
    reg_g2v = train_bridge_regressor(
        paired_train_data[1], paired_train_data[0], f"GAN -> VAE ({config_tag})", **bridge_params
    )

    # Run qualitative visualizations
    logging.info("\n--- Running Qualitative Visualizations ---")
    for i in range(1, NUM_VIS_EXP + 1):
        seed = np.random.randint(0, 10000)
        visualize_vae_to_gan(vae_model, gan_generator, reg_v2g, cnn_classifier, device, vis_data, test_dataset, seed, i, config_tag, output_dir)
        visualize_gan_to_vae(vae_model, gan_generator, reg_g2v, cnn_classifier, device, vis_data, test_dataset, seed, i, config_tag, output_dir)
    logging.info(f"    {NUM_VIS_EXP} visualization pairs created.")

    # Run quantitative evaluation
    evaluate_projection_quality('VAE->GAN', vae_model, gan_generator, reg_v2g, cnn_classifier, device, test_dataset, paired_test_data, NUM_EVAL_SAMPLES)
    evaluate_projection_quality('GAN->VAE', vae_model, gan_generator, reg_g2v, cnn_classifier, device, test_dataset, paired_test_data, NUM_EVAL_SAMPLES)

def run_single_digit_experiment(digit, base_output_dir, common_data_all_digits, experiments):
    """
    Orchestrates the setup and execution of experiments for a single MNIST digit.
    This version RETRAINS the MLP bridges on data exclusively from the target digit.
    """
    logging.info("\n\n" + "#"*80)
    logging.info(f"### STARTING SPECIALIZED SINGLE-DIGIT EXPERIMENT FOR DIGIT: '{digit}' ###")
    logging.info("### (Bridges will be retrained on single-digit data) ###")
    logging.info("#"*80)

    # --- 1. Prepare Single-Digit Data ---
    # Unpack models and device from the common data
    vae_model, gan_generator, cnn_classifier, device = common_data_all_digits['models']

    # Create the specialized paired latent dataset for training the bridges
    (paired_vae_train_digit, paired_gan_train_digit), paired_test_data_digit = \
        create_paired_latent_dataset_for_digit(
            vae_model, gan_generator, cnn_classifier, device, digit=digit
        )

    # Create the filtered test set for evaluation and visualization
    full_test_dataset = datasets.MNIST(root='~/datasets', train=False, download=True, transform=transforms.ToTensor())
    test_dataset_single_digit = filter_mnist_by_digit(full_test_dataset, digit)
    logging.info(f"Created a filtered test set with {len(test_dataset_single_digit)} samples of digit '{digit}' for final evaluation.")

    # --- 2. Create a new common_data package for this experiment ---
    # We replace BOTH the paired training data AND the test data
    common_data_single_digit = common_data_all_digits.copy()
    
    # Use the new specialized data for training and testing the bridges
    common_data_single_digit['paired_data'] = ((paired_vae_train_digit, paired_gan_train_digit), paired_test_data_digit)
    
    # The visualization background remains all digits, but the test images will be the single digit
    vis_scatter_data, _ = common_data_all_digits['vis_data']
    common_data_single_digit['vis_data'] = (vis_scatter_data, test_dataset_single_digit)

    # --- 3. Run all experiment configurations with the specialized data ---
    for exp in experiments:
        # Modify the experiment name and output directory for clarity
        single_digit_exp_name = f"Specialized Bridge for Digit '{digit}': {exp['name']}"
        single_digit_output_dir = os.path.join(base_output_dir, f"digit_{digit}_specialized", exp['tag'])
        
        run_experiment(
            exp_name=single_digit_exp_name,
            config_tag=exp['tag'],
            output_dir=single_digit_output_dir,
            bridge_params=exp['params'],
            common_data=common_data_single_digit
        )

def main():
    """Main function to orchestrate the entire analysis pipeline."""
    BASE_OUTPUT_DIR = "outputs_mlp"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    log_file_path = setup_logging(BASE_OUTPUT_DIR)

    # --- Initial Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting VAE-GAN Projection Analysis on device: {device}")
    logging.info(f"Full log will be saved to: {log_file_path}")
    
    # --- Load Models & Data (once for all experiments) ---
    vae_model, gan_generator = load_models(device)
    cnn_classifier = train_cnn_classifier(device)
    (paired_vae_train, paired_gan_train), paired_test_data = create_paired_latent_dataset(vae_model, gan_generator, device)
    vis_data = create_visualization_datasets(vae_model, gan_generator, device)
    test_dataset = datasets.MNIST(root='~/datasets', train=False, download=True, transform=transforms.ToTensor())
    
    # --- Define Experiment Configurations ---
    experiments = [
        {
            "name": "Baseline: Small MLP with Identity Activation",
            "tag": "baseline_small_identity",
            "params": {"hidden_layer_sizes": (2, 4, 2), "activation": 'identity'}
        },
        {
            "name": "Experiment 1: Large MLP with Identity Activation",
            "tag": "exp1_large_identity",
            "params": {"hidden_layer_sizes": (2, 8, 2), "activation": 'identity'}
        },
        {
            "name": "Experiment 2: Small MLP with ReLU Activation",
            "tag": "exp2_small_relu",
            "params": {"hidden_layer_sizes": (2, 4, 2), "activation": 'relu'}
        },
        {
            "name": "Experiment 3: Large MLP with ReLU Activation",
            "tag": "exp3_large_relu",
            "params": {"hidden_layer_sizes": (2, 8, 2), "activation": 'relu'}
        }
    ]

    # --- Package common data to pass to the runner ---
    common_data = {
        "models": (vae_model, gan_generator, cnn_classifier, device),
        "paired_data": ((paired_vae_train, paired_gan_train), paired_test_data),
        "vis_data": (vis_data, test_dataset), # This contains the FULL test set
        "constants": (3, 1000) # (NUM_VIS_EXP, NUM_EVAL_SAMPLES)
    }

    # --- Run All Multi-Class Experiments ---
    multi_class_output_dir = os.path.join(BASE_OUTPUT_DIR, "multi_class")
    for exp in experiments:
        run_experiment(
            exp_name=f"Multi-Class: {exp['name']}",
            config_tag=exp['tag'],
            output_dir=os.path.join(multi_class_output_dir, exp['tag']),
            bridge_params=exp['params'],
            common_data=common_data
        )

    # --- Run Single-Digit Experiment for Digit '2' with Retrained Bridges ---
    run_single_digit_experiment(
        digit=2,
        base_output_dir=BASE_OUTPUT_DIR,
        common_data_all_digits=common_data,
        experiments=experiments
    )

    # --- Final Steps ---
    plot_latent_space(vae_model, 'vae', device, output_path=os.path.join(BASE_OUTPUT_DIR, "final_vae_latent_space.png"))
    plot_latent_space(gan_generator, 'gan', device, output_path=os.path.join(BASE_OUTPUT_DIR, "final_gan_latent_space.png"))

    logging.info(f"\n\nAnalysis complete. All outputs saved in '{BASE_OUTPUT_DIR}/' subdirectories.")
    logging.info(f"A detailed log has been saved to '{log_file_path}'")

if __name__ == "__main__":
    main()