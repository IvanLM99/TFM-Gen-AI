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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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

# ==============================================================================
# === NEW MODEL FOR EXPERIMENT 2 ===
# ==============================================================================
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
        tuple: Contains train and test splits: ((train_vae, train_gan), (test_vae, test_gan)).
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

    paired_vae_vectors = np.concatenate(paired_vae_vectors_list, axis=0)[:num_samples]
    paired_gan_vectors = np.concatenate(paired_gan_vectors_list, axis=0)[:num_samples]
    
    # Split the dataset for training and for quality evaluation later
    X_vae_train, X_vae_test, X_gan_train, X_gan_test = train_test_split(
        paired_vae_vectors, paired_gan_vectors, test_size=0.2, random_state=42
    )
    print(f"Dataset split into {len(X_vae_train)} training samples and {len(X_vae_test)} test samples.")
    return (X_vae_train, X_gan_train), (X_vae_test, X_gan_test)

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

# ==============================================================================
# === NEW FUNCTIONS FOR EXPERIMENTS ===
# ==============================================================================
def train_bridge_experimental(X_train, Y_train, direction, hidden_layer_sizes, activation, max_iter=1000):
    """
    Trains an MLP Regressor with specified architecture and activation function.
    
    Args:
        X_train (np.array): The input vectors (source space).
        Y_train (np.array): The target vectors (destination space).
        direction (str): A string describing the projection direction for logging.
        hidden_layer_sizes (tuple): Defines the size of the hidden layers.
        activation (str): 'identity', 'relu', etc.
        max_iter (int): Maximum iterations for the solver.
        
    Returns:
        MLPRegressor: The trained scikit-learn MLP regressor model.
    """
    print(f"\n--- Training Experimental MLP Bridge for {direction} ---")
    print(f"Architecture: hidden_layer_sizes={hidden_layer_sizes}, activation='{activation}'")
    
    regressor = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, 
        activation=activation, 
        solver='adam', 
        max_iter=max_iter, 
        random_state=42, 
        early_stopping=True, 
        verbose=False,
        n_iter_no_change=15
    )
    
    print("Fitting regressor... (This may take a minute or more for larger models)")
    regressor.fit(X_train, Y_train)
    mse = mean_squared_error(Y_train, regressor.predict(X_train))
    print(f"MLP Regressor for {direction} trained. Final Training MSE on vectors: {mse:.6f}")
    return regressor

def train_cnn_classifier(device, epochs=5, batch_size=64, lr=0.001):
    """
    Trains the CNNClassifier on the MNIST dataset or loads it if already trained.
    
    Returns:
        CNNClassifier: The trained (or loaded) model.
    """
    classifier_path = 'outputs_mlp/cnn_classifier.pth'
    os.makedirs(os.path.dirname(classifier_path), exist_ok=True)

    if os.path.exists(classifier_path):
        print("\n--- Loading pre-trained CNN Classifier ---")
        model = CNNClassifier().to(device)
        model.load_state_dict(torch.load(classifier_path, map_location=device))
        model.eval()
        print("CNN Classifier loaded.")
        return model

    print("\n--- Training new CNN Classifier for Evaluation ---")
    model = CNNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # The default 'mean' is fine for the training loop's backpropagation.
    train_loss_fn = nn.NLLLoss() 
    # Create a separate loss function for testing to get the sum ---
    test_loss_fn = nn.NLLLoss(reduction='sum')

    train_loader = DataLoader(
        datasets.MNIST('~/datasets', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(
        datasets.MNIST('~/datasets', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # Use the training loss function
            loss = train_loss_fn(output, target)
            loss.backward()
            optimizer.step()
        
        # Test after each epoch
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # Use the test loss function and remove the 'reduction' argument 
                test_loss += test_loss_fn(output, target).item() 
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Epoch {epoch} Test Set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

    torch.save(model.state_dict(), classifier_path)
    print(f"CNN Classifier trained and saved to '{classifier_path}'")
    model.eval()
    return model

def evaluate_projection_quality(
    direction, vae_model, gan_generator, regressor, classifier, 
    device, test_dataset, paired_test_data, num_eval_samples=1000):
    """
    Evaluates the quality of a projection pipeline by measuring the
    classification accuracy and reconstruction error of projected images.
    """
    print(f"\n--- Evaluating Projection Quality for {direction} ---")
    
    mses, correct_predictions, total_samples = [], 0, 0
    
    with torch.no_grad():
        if direction == 'VAE->GAN':
            eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            iterator = tqdm(eval_loader, total=num_eval_samples, desc=f"Evaluating {direction}")
            for i, (original_img_tensor, original_label) in enumerate(iterator):
                if i >= num_eval_samples: break
                
                original_img_tensor, original_label = original_img_tensor.to(device), original_label.to(device)
                
                z_vae, _ = vae_model.encode(original_img_tensor)
                pred_gan_vec = torch.tensor(regressor.predict(z_vae.cpu().numpy()), device=device, dtype=torch.float).view(1, 2, 1, 1)
                final_img_tensor = (gan_generator(pred_gan_vec) + 1) / 2.0 # Project and normalize to [0,1]
                
                output = classifier(final_img_tensor)
                pred = output.argmax(dim=1, keepdim=True)
                if pred.item() == original_label.item(): correct_predictions += 1
                
                mse = mean_squared_error(original_img_tensor.cpu().numpy().flatten(), final_img_tensor.cpu().numpy().flatten())
                mses.append(mse)
                total_samples += 1

        elif direction == 'GAN->VAE':
            # Use the paired test set to avoid slow GAN inversion for large-scale evaluation
            X_vae_test, X_gan_test = paired_test_data
            num_eval_samples = min(num_eval_samples, len(X_gan_test))
            iterator = tqdm(range(num_eval_samples), desc=f"Evaluating {direction}")
            for i in iterator:
                z_gan_np = X_gan_test[i].reshape(1, 2)
                z_gan_tensor = torch.tensor(z_gan_np, device=device, dtype=torch.float).view(1, 2, 1, 1)

                original_img_tensor = (gan_generator(z_gan_tensor) + 1) / 2.0
                original_label = classifier(original_img_tensor).argmax(dim=1, keepdim=True)

                pred_vae_vec = torch.tensor(regressor.predict(z_gan_np), device=device, dtype=torch.float)
                final_img_tensor = vae_model.decode(pred_vae_vec).view(1, 1, 28, 28)

                output = classifier(final_img_tensor)
                pred = output.argmax(dim=1, keepdim=True)
                if pred.item() == original_label.item(): correct_predictions += 1

                mse = mean_squared_error(original_img_tensor.cpu().numpy().flatten(), final_img_tensor.cpu().numpy().flatten())
                mses.append(mse)
                total_samples += 1
    
    mses = np.array(mses)
    median_mse = np.median(mses)
    std_dev_mse = np.std(mses)
    accuracy = 100. * correct_predictions / total_samples

    print(f"\n--- Quality Report for {direction} (Regressor: {regressor.activation}, Layers: {regressor.hidden_layer_sizes}) ---")
    print(f"Evaluated on {total_samples} samples.")
    print(f"Projected Image Classification Accuracy: {accuracy:.2f}%")
    print(f"Image Reconstruction MSE:")
    print(f"  - Median: {median_mse:.6f}")
    print(f"  - Standard Deviation: {std_dev_mse:.6f}")

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

def visualize_vae_to_gan(vae_model, gan_generator, regressor, device, vis_data, test_dataset, seed, num, tag=""):
    """Visualizes the full VAE -> GAN projection pipeline for one image."""
    (X_vae, y_vae), (X_gan, y_gan) = vis_data
    print(f"\n--- Running VAE -> GAN projection (Experiment #{num}, Tag: {tag or 'original'}) ---")
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
    filename_tag = f"_{tag}" if tag else ""
    fig = _create_plot(f"Direction 1 (Run {num}): VAE -> GAN projection\nImage projection MSE = {mse:.4f}", plot_data)
    fig.savefig(f"outputs_mlp/dir1_vae_to_gan_run_{num}{filename_tag}.png"); plt.close(fig)
    print(f"Saved VAE->GAN plot to 'outputs_mlp/dir1_vae_to_gan_run_{num}{filename_tag}.png'")

def visualize_gan_to_vae(vae_model, gan_generator, regressor, device, vis_data, test_dataset, seed, num, tag=""):
    """Visualizes the full GAN -> VAE projection pipeline for one image."""
    (X_vae, y_vae), (X_gan, y_gan) = vis_data
    print(f"\n--- Running GAN -> VAE projection (Experiment #{num}, Tag: {tag or 'original'}) ---")
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
    filename_tag = f"_{tag}" if tag else ""
    fig = _create_plot(f"Direction 2 (Run {num}): GAN -> VAE projection\nImage projection MSE = {mse:.4f}", plot_data)
    fig.savefig(f"outputs_mlp/dir2_gan_to_vae_run_{num}{filename_tag}.png"); plt.close(fig)
    print(f"Saved GAN->VAE plot to 'outputs_mlp/dir2_gan_to_vae_run_{num}{filename_tag}.png'")

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
    num_visualization_experiments = 3 # Reduced for brevity of new experiments
    num_quality_eval_samples = 1000

    # --- Step 1: Load Models ---
    vae_model, gan_generator = load_models(device)

    # --- Step 2: Create Paired Dataset & Export ---
    (paired_vae_train, paired_gan_train), paired_test_data = create_paired_latent_dataset(vae_model, gan_generator, device)
    print("\n--- Exporting paired latent vectors to text files ---")
    np.savetxt("outputs_mlp/paired_vae_vectors_train.txt", paired_vae_train, delimiter=",")
    np.savetxt("outputs_mlp/paired_gan_vectors_train.txt", paired_gan_train, delimiter=",")
    print("Training vectors saved to 'outputs_mlp/'")

    # --- Step 3: Train Original MLP Bridges & Run Visualizations ---
    regressor_vae_to_gan = train_bridge(paired_vae_train, paired_gan_train, "VAE -> GAN")
    regressor_gan_to_vae = train_bridge(paired_gan_train, paired_vae_train, "GAN -> VAE")

    vis_data = create_visualization_datasets(vae_model, gan_generator, device)
    test_dataset = datasets.MNIST(root='~/datasets', train=False, download=True, transform=transforms.ToTensor())
    
    for i in range(1, num_visualization_experiments + 1):
        seed = np.random.randint(0, 10000)
        visualize_vae_to_gan(vae_model, gan_generator, regressor_vae_to_gan, device, vis_data, test_dataset, seed, i)
        visualize_gan_to_vae(vae_model, gan_generator, regressor_gan_to_vae, device, vis_data, test_dataset, seed, i)

    # ##############################################################################
    # ### NEW EXPERIMENTS START HERE ###
    # ##############################################################################
    print("\n\n" + "="*80)
    print("### STARTING NEW EXPERIMENTS ###")
    print("="*80 + "\n")
    
    # --- EXPERIMENT 1: MLP with more neurons (Identity Activation) ---
    print("\n\n--- EXPERIMENT 1: MLP with more neurons (Identity Activation) ---\n")
    large_mlp_layers = (2, 8, 2)
    regressor_v2g_large = train_bridge_experimental(paired_vae_train, paired_gan_train, "VAE -> GAN (Large)", large_mlp_layers, 'identity')
    regressor_g2v_large = train_bridge_experimental(paired_gan_train, paired_vae_train, "GAN -> VAE (Large)", large_mlp_layers, 'identity')
    
    for i in range(1, num_visualization_experiments + 1):
        seed = np.random.randint(0, 10000)
        visualize_vae_to_gan(vae_model, gan_generator, regressor_v2g_large, device, vis_data, test_dataset, seed, i, tag="large_mlp")
        visualize_gan_to_vae(vae_model, gan_generator, regressor_g2v_large, device, vis_data, test_dataset, seed, i, tag="large_mlp")

    # --- EXPERIMENT 2: CNN Classifier and Quality Evaluation (for original MLPs) ---
    print("\n\n--- EXPERIMENT 2: CNN Classifier and Quality Evaluation ---\n")
    cnn_classifier = train_cnn_classifier(device)
    
    print("\n--- Evaluating quality of ORIGINAL (small, identity) MLPs ---")
    evaluate_projection_quality('VAE->GAN', vae_model, gan_generator, regressor_vae_to_gan, cnn_classifier, device, test_dataset, paired_test_data, num_quality_eval_samples)
    evaluate_projection_quality('GAN->VAE', vae_model, gan_generator, regressor_gan_to_vae, cnn_classifier, device, test_dataset, paired_test_data, num_quality_eval_samples)
    
    # --- EXPERIMENT 3: All experiments but with RELU activation ---
    print("\n\n--- EXPERIMENT 3: All experiments with ReLU activation ---\n")

    # Part A: Small MLP with ReLU
    print("\n--- Part 3A: Small MLP with ReLU Activation ---\n")
    small_mlp_layers = (2, 4, 2)
    regressor_v2g_relu_small = train_bridge_experimental(paired_vae_train, paired_gan_train, "VAE -> GAN (Small, ReLU)", small_mlp_layers, 'relu')
    regressor_g2v_relu_small = train_bridge_experimental(paired_gan_train, paired_vae_train, "GAN -> VAE (Small, ReLU)", small_mlp_layers, 'relu')
    
    for i in range(1, num_visualization_experiments + 1):
        seed = np.random.randint(0, 10000)
        visualize_vae_to_gan(vae_model, gan_generator, regressor_v2g_relu_small, device, vis_data, test_dataset, seed, i, tag="relu_small")
        visualize_gan_to_vae(vae_model, gan_generator, regressor_g2v_relu_small, device, vis_data, test_dataset, seed, i, tag="relu_small")
    
    print("\n--- Evaluating quality of SMALL ReLU MLPs ---")
    evaluate_projection_quality('VAE->GAN', vae_model, gan_generator, regressor_v2g_relu_small, cnn_classifier, device, test_dataset, paired_test_data, num_quality_eval_samples)
    evaluate_projection_quality('GAN->VAE', vae_model, gan_generator, regressor_g2v_relu_small, cnn_classifier, device, test_dataset, paired_test_data, num_quality_eval_samples)

    # Part B: Large MLP with ReLU
    print("\n--- Part 3B: Large MLP with ReLU Activation ---\n")
    regressor_v2g_relu_large = train_bridge_experimental(paired_vae_train, paired_gan_train, "VAE -> GAN (Large, ReLU)", large_mlp_layers, 'relu')
    regressor_g2v_relu_large = train_bridge_experimental(paired_gan_train, paired_vae_train, "GAN -> VAE (Large, ReLU)", large_mlp_layers, 'relu')

    for i in range(1, num_visualization_experiments + 1):
        seed = np.random.randint(0, 10000)
        visualize_vae_to_gan(vae_model, gan_generator, regressor_v2g_relu_large, device, vis_data, test_dataset, seed, i, tag="relu_large")
        visualize_gan_to_vae(vae_model, gan_generator, regressor_g2v_relu_large, device, vis_data, test_dataset, seed, i, tag="relu_large")

    print("\n--- Evaluating quality of LARGE ReLU MLPs ---")
    evaluate_projection_quality('VAE->GAN', vae_model, gan_generator, regressor_v2g_relu_large, cnn_classifier, device, test_dataset, paired_test_data, num_quality_eval_samples)
    evaluate_projection_quality('GAN->VAE', vae_model, gan_generator, regressor_g2v_relu_large, cnn_classifier, device, test_dataset, paired_test_data, num_quality_eval_samples)
    
    # --- Step 5: Generate Final Latent Space Plots ---
    plot_latent_space(vae_model, 'vae', device, output_path="outputs_mlp/vae_latent_space.png")
    plot_latent_space(gan_generator, 'gan', device, output_path="outputs_mlp/gan_latent_space.png")

    print(f"\n\nAnalysis complete. All outputs saved in 'outputs_mlp/' directory.")

if __name__ == "__main__":
    main()
