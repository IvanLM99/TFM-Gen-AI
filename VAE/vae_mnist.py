# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars in console
import os # For path expansion

def main():
    print("--- VAE MNIST Training Script ---")

    # Device configuration
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA (or ROCm via HIP) available: True")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA (or ROCm via HIP) available: False, using CPU.")
    print(f"Using device: {device}")

    # Hyperparameters
    input_dim = 784
    hidden_dim = 400
    latent_dim_vae = 200 
    latent_dim_plot = 2
    batch_size = 100 
    epochs = 50
    lr = 0.001 

    print(f"\nHyperparameters:")
    print(f"  Input Dimension: {input_dim}")
    print(f"  Hidden Dimension: {hidden_dim}")
    print(f"  VAE Latent Dimension (internal): {latent_dim_vae}")
    print(f"  VAE Latent Dimension (output z for plots): {latent_dim_plot}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")

    # --- Loading dataset ---
    print("\n--- Loading MNIST dataset ---")
    transform = transforms.Compose([transforms.ToTensor()])
    path = os.path.expanduser('~/datasets')
    print(f"Dataset path: {path}")
    try:
        train_dataset = MNIST(path, transform=transform, download=True, train=True)
        test_dataset  = MNIST(path, transform=transform, download=True, train=False)
    except Exception as e:
        print(f"Error downloading/loading dataset: {e}")
        return

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    # --- VAE architecture ---
    class VAE(nn.Module):
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim_internal=200, latent_dim_output=2):
            super(VAE, self).__init__()
            self.input_dim = input_dim
            # Encoder network
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, latent_dim_internal),
                nn.LeakyReLU(0.2)
            )
            # Mean and log variance layers
            self.mean_layer = nn.Linear(latent_dim_internal, latent_dim_output)
            self.logvar_layer = nn.Linear(latent_dim_internal, latent_dim_output)
            # Decoder network
            self.decoder_nn = nn.Sequential(
                nn.Linear(latent_dim_output, latent_dim_internal),
                nn.LeakyReLU(0.2),
                nn.Linear(latent_dim_internal, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )
        
        def encode(self, x):
            x = self.encoder(x)
            mean, logvar = self.mean_layer(x), self.logvar_layer(x)
            return mean, logvar

        def reparameterization(self, mean, log_var, current_device):
            epsilon = torch.randn_like(log_var).to(current_device)    
            z = mean + torch.exp(0.5 * log_var) * epsilon 
            return z

        def decode(self, x):
            return self.decoder_nn(x)

        def forward(self, x, current_device): 
            x = x.view(-1, self.input_dim)
            mean, log_var = self.encode(x) 
            z = self.reparameterization(mean, log_var, current_device)
            x_hat = self.decode(z)
            return x_hat, mean, log_var


    # --- Initialize VAE ---
    print("\n--- Initializing VAE model and optimizer ---")
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim_internal=latent_dim_vae, latent_dim_output=latent_dim_plot).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Model and optimizer initialized.")

    # --- Loss function ---
    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = F.binary_cross_entropy(x_hat, x.view(-1, input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - torch.exp(log_var))
        return reproduction_loss + KLD

    # --- Train model ---
    print("\n--- Starting VAE training ---")
    def train_vae(model_instance, optimizer_instance, num_epochs, current_device, x_dim=784):
        model_instance.train()
        history = {'loss': []} # Initialize dictionary to store loss values
        for epoch in range(num_epochs):
            overall_loss = 0
            for batch_idx, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                x = x.view(batch_size, x_dim).to(current_device) 
                optimizer_instance.zero_grad()
                x_hat, mean, log_var = model_instance(x, current_device)
                loss = loss_function(x, x_hat, mean, log_var)
                overall_loss += loss.item()
                loss.backward()
                optimizer_instance.step()
            
            avg_loss_per_sample = overall_loss / (batch_idx*batch_size)
            history['loss'].append(avg_loss_per_sample)
            print(f"\tEpoch {epoch + 1}/{num_epochs} \tAverage Loss: {avg_loss_per_sample:.6f}")
        return history

    vae_history = train_vae(model, optimizer, epochs, device, x_dim=input_dim) # Pass correct epochs
    print("--- VAE Training Finished ---")

    # --- VAE loss curves ---
    print("\n--- Plotting VAE training loss ---")
    def plot_loss(history, title):
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='VAE Loss')
        plt.title(f'{title} Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{title.lower().replace(" ", "_")}_loss_curve.png')
        print(f"Loss curve plot saved as '{title.lower().replace(' ', '_')}_loss_curve.png'")

    plot_loss(vae_history, 'VAE')

    # --- Latent space scatter plot visualization ---
    print("\n--- Visualizing Latent Space (Scatter Plot) ---")
    def plot_latent_space_scatter(model_instance, data_loader, current_device, num_batches=100):
        model_instance.eval()
        latents = []
        labels_list = []
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(data_loader, desc="Scatter plot data extraction")):
                if i >= num_batches:
                    break
                x = x.view(x.size(0), -1).to(current_device)
                z_mean, _ = model_instance.encode(x) # Use z_mean for scatter
                latents.append(z_mean.cpu().numpy())
                labels_list.append(y.numpy())
        
        latents = np.concatenate(latents, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        if latents.shape[1] != 2:
            print(f"Latent space dimension for scatter plot is {latents.shape[1]}, not 2. Skipping direct 2D scatter plot.")
            return

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels_list, cmap='tab10', s=5)
        plt.colorbar(scatter, ticks=range(10))
        plt.xlabel("Latent Dim 1 (mean)")
        plt.ylabel("Latent Dim 2 (mean)")
        plt.title("VAE Latent Space Scatter Plot")
        plt.grid(True)
        plt.savefig('vae_latent_space_scatter.png')
        print("Latent space scatter plot saved as 'vae_latent_space_scatter.png'")

    plot_latent_space_scatter(model, test_loader, device, num_batches=50)

    # --- Latent space visualization ---
    print("\n--- Visualizing Latent Space ---")
    def plot_latent_space(model_instance, current_device, scale=1.5, n=25, digit_size=28, figsize=15):
        # latent_dim_plot is 2 for this visualization
        if model_instance.mean_layer.out_features != 2:
            print(f"Latent dimension for plot is {model_instance.mean_layer.out_features}, needs to be 2. Skipping plot.")
            return

        model_instance.eval() # Set model to evaluation mode
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        with torch.no_grad(): # No gradients needed for generation
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(current_device)
                    x_decoded = model_instance.decode(z_sample)
                    digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
                    figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size] = digit.numpy()

        plt.figure(figsize=(figsize, figsize))
        plt.title(f'VAE Latent Space Visualization)')
        
        # Adjust ticks to be at the center of each digit
        start_range = digit_size / 2.0
        end_range = n * digit_size - digit_size + start_range
        pixel_range = np.linspace(start_range, end_range, n)
        
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        
        plt.xlabel("mean, z [0]")
        plt.ylabel("var, z [1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.savefig(f'vae_latent_space.png')
        print(f"VAE Latent Space Visualization plot saved as 'vae_latent_space.png'")

    plot_latent_space(model, device, scale=1.5, n=25, digit_size=28, figsize=15)


    # --- t-SNE visualization of latent space ---
    print("\n--- Visualizing Latent Space with t-SNE ---")
    def plot_tsne(model_instance, data_loader, current_device, num_batches=100, perplexity_val=30):
        from sklearn.manifold import TSNE 
        model_instance.eval()
        latents_mu = []
        labels_list = []
        print("Extracting latent means for t-SNE...")
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(data_loader, desc="t-SNE data extraction")):
                if i >= num_batches:
                    break
                x = x.view(x.size(0), -1).to(current_device)
                mu, _ = model_instance.encode(x) 
                latents_mu.append(mu.cpu().numpy())
                labels_list.append(y.numpy())
        
        latents_mu = np.concatenate(latents_mu, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        print(f"Running t-SNE on {latents_mu.shape[0]} samples with latent dimension {latents_mu.shape[1]}...")
        tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity_val, n_iter=300, random_state=42)
        tsne_results = tsne.fit_transform(latents_mu)
        
        print("Plotting t-SNE results...")
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels_list, cmap="tab10", s=5)
        plt.colorbar(scatter, ticks=range(10))
        plt.xlabel("t-SNE component 1")
        plt.ylabel("t-SNE component 2")
        plt.title(f"t-SNE visualization of VAE Latent Space)")
        plt.grid(True)
        plt.savefig(f'vae_tsne_latent_space.png')
        print(f"t-SNE plot saved as 'vae_tsne_latent_space.png'")

    plot_tsne(model, test_loader, device, num_batches=50, perplexity_val=30)

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
