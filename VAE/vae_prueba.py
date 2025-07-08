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
from mpl_toolkits.mplot3d import Axes3D # For 3D scatter plotting
from tqdm import tqdm # For progress bars in console
import os # For path expansion

# Create output directory if it doesn't exist
os.makedirs("outputs_vae", exist_ok=True)

# --- VAE architecture ---
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim_internal=200, latent_dim_output=3):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, latent_dim_internal), nn.LeakyReLU(0.2))
        self.mean_layer = nn.Linear(latent_dim_internal, latent_dim_output)
        self.logvar_layer = nn.Linear(latent_dim_internal, latent_dim_output)
        self.decoder_nn = nn.Sequential(nn.Linear(latent_dim_output, latent_dim_internal), nn.LeakyReLU(0.2), nn.Linear(latent_dim_internal, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
    def encode(self, x):
        return self.mean_layer(self.encoder(x)), self.logvar_layer(self.encoder(x))
    def reparameterization(self, mean, log_var, current_device):
        epsilon = torch.randn_like(log_var).to(current_device)    
        return mean + torch.exp(0.5 * log_var) * epsilon 
    def decode(self, x):
        return self.decoder_nn(x)
    def forward(self, x, current_device): 
        x = x.view(-1, self.input_dim)
        mean, log_var = self.encode(x) 
        z = self.reparameterization(mean, log_var, current_device)
        return self.decode(z), mean, log_var

# --- Loss function ---
def loss_function(x, x_hat, mean, log_var, input_dim=784):
    reproduction_loss = F.binary_cross_entropy(x_hat, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - torch.exp(log_var))
    return reproduction_loss + KLD

# --- Train model ---
def train_vae(model_instance, optimizer_instance, train_loader, num_epochs, current_device, x_dim=784):
    model_instance.train()
    history = {'loss': []}
    for epoch in range(num_epochs):
        overall_loss, num_samples = 0, 0
        for batch_idx, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            x = x.view(-1, x_dim).to(current_device) 
            optimizer_instance.zero_grad()
            x_hat, mean, log_var = model_instance(x, current_device)
            loss = loss_function(x, x_hat, mean, log_var, input_dim=x_dim)
            overall_loss += loss.item()
            num_samples += x.size(0)
            loss.backward()
            optimizer_instance.step()
        avg_loss_per_sample = overall_loss / num_samples
        history['loss'].append(avg_loss_per_sample)
        print(f"\tEpoch {epoch + 1}/{num_epochs} \tAverage Loss: {avg_loss_per_sample:.6f}")
    return history

# --- PLOTTING FUNCTION 1: Loss Curve ---
def plot_loss(history):
    print("\n--- Plotting Training Loss Curve ---")
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='VAE Loss')
    plt.title('VAE Training Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig('outputs_vae/vae_loss_curve.png')
    print("Loss curve plot saved as 'vae_loss_curve.png'")

# --- PLOTTING FUNCTION 2: 3D Scatter Plot ---
def plot_3d_latent_scatter(model_instance, data_loader, current_device, max_batches=50):
    if model_instance.mean_layer.out_features != 3: return
    print("\n--- Visualizing Latent Space (Scatter Plot) ---")
    model_instance.eval()
    all_latents, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(data_loader, desc="Extracting points for 3D scatter")):
            if batch_idx >= max_batches: break
            z_mean, _ = model_instance.encode(x.view(x.size(0), -1).to(current_device))
            all_latents.append(z_mean.cpu().numpy()); all_labels.append(y.numpy())
    latents = np.concatenate(all_latents, axis=0); labels = np.concatenate(all_labels, axis=0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(latents[:, 0], latents[:, 1], latents[:, 2], c=labels, cmap='tab10', s=5)
    ax.set_title('Latent Space plot'); ax.set_xlabel('z[0]'); ax.set_ylabel('z[1]'); ax.set_zlabel('z[2]')
    legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
    ax.add_artist(legend1)
    plt.savefig('outputs_vae/vae_3d_latent_scatter.png')
    print("3D scatter plot saved as 'vae_3d_latent_scatter.png'")

# --- PLOTTING FUNCTION 3: t-SNE Plot ---
def plot_tsne(model_instance, data_loader, current_device):
    from sklearn.manifold import TSNE 
    print("\n--- Visualizing Latent Space with t-SNE ---")
    model_instance.eval()
    latents_mu, labels_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(data_loader, desc="t-SNE data extraction")):
            if i >= 50: break # Use a subset for speed
            mu, _ = model_instance.encode(x.view(x.size(0), -1).to(current_device)) 
            latents_mu.append(mu.cpu().numpy()); labels_list.append(y.numpy())
    latents_mu = np.concatenate(latents_mu, axis=0); labels_list = np.concatenate(labels_list, axis=0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(latents_mu)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels_list, cmap="tab10", s=10)
    plt.colorbar(scatter, ticks=range(10)); plt.title('t-SNE Visualization of Latent Space')
    plt.savefig('outputs_vae/vae_tsne_plot.png'); print("t-SNE plot saved as 'vae_tsne_plot.png'")

# --- PLOTTING FUNCTION 4: 8-Corner Manifold Plot ---
def plot_3d_latent_space_corners(model_instance, current_device, scale=1.5, n=7, zoom=0.2, digit_size=28, figsize=(20, 10)):
    if model_instance.mean_layer.out_features != 3: return
    print("\n--- Visualizing 8 Corner Regions of the 3D Latent Space Manifold ---")
    model_instance.eval()
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle('Latent Space Corners', fontsize=24)
    axes = axes.flatten()
    corners = [
        (-scale,-scale,-scale),(scale,-scale,-scale),(-scale,scale,-scale),(scale,scale,-scale),
        (-scale,-scale,scale),(scale,-scale,scale),(-scale,scale,scale),(scale,scale,scale),
    ]
    # Cube corner names for clear orientation (Convention: z=+ Front, y=+ Top, x=+ Right)
    corner_names = [
        "Bottom-Left-Back", "Bottom-Right-Back", "Top-Left-Back", "Top-Right-Back",
        "Bottom-Left-Front", "Bottom-Right-Front", "Top-Left-Front", "Top-Right-Front",
    ]
    for i, (corner, name) in enumerate(zip(corners, corner_names)):
        ax, (cx, cy, cz) = axes[i], corner
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x, grid_y = np.linspace(cx-zoom, cx+zoom, n), np.linspace(cy-zoom, cy+zoom, n)[::-1]
        with torch.no_grad():
            for row_idx, val_y in enumerate(grid_y):
                for col_idx, val_x in enumerate(grid_x):
                    z_sample = torch.tensor([[val_x, val_y, cz]], dtype=torch.float)
                    digit = model_instance.decode(z_sample.to(current_device))[0].detach().cpu().reshape(digit_size, digit_size)
                    figure[row_idx*digit_size:(row_idx+1)*digit_size, col_idx*digit_size:(col_idx+1)*digit_size] = digit.numpy()
        ax.imshow(figure, cmap="Greys_r")
        ax.set_title(f'{name}\n({cx:.1f}, {cy:.1f}, {cz:.1f})', fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig('outputs_vae/vae_3d_latent_space_corners.png')
    print("3D Latent space corner plots saved as 'outputs_vae/vae_3d_latent_space_corners.png'")

# --- MAIN EXECUTION ---
def main():
    print("--- VAE MNIST Training Script (Comprehensive 3D Analysis) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    latent_dim_plot = 3 # MUST be 3 for this analysis
    epochs = 50 # A good number of epochs for decent results
    model = VAE(latent_dim_output=latent_dim_plot).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Loading dataset ---
    transform = transforms.Compose([transforms.ToTensor()])
    path = os.path.expanduser('~/datasets')
    train_dataset = MNIST(path, transform=transform, download=True, train=True)
    test_dataset  = MNIST(path, transform=transform, download=True, train=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    
    # --- Train the Model ---
    history = train_vae(model, optimizer, train_loader, epochs, device)
    
    # --- Generate All Visualizations ---
    plot_loss(history)
    plot_3d_latent_scatter(model, test_loader, device)
    plot_tsne(model, test_loader, device)
    plot_3d_latent_space_corners(model, device, scale=1.8, n=7, zoom=0.3)
    
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()