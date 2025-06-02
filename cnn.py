# cnn_classification_pytorch.py
# Part 3: Image Classification with Generated Images (PyTorch)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os

# --- Parameters ---
IMG_WIDTH, IMG_HEIGHT = 28, 28
IMG_CHANNELS = 1
INPUT_SHAPE_CNN = (IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH) # PyTorch (C, H, W)
NUM_CLASSES = 10
BATCH_SIZE_CNN = 128
EPOCHS_CNN = 10 # Epochs for CNN training

# GAN Params (should match gan_mnist_pytorch.py)
GAN_LATENT_DIM = 100
GAN_NGF = 64 # Generator feature maps
GAN_SAVE_DIR = 'gan_outputs_pytorch' # Directory where PyTorch GAN weights are saved

# Number of GAN images to generate for training the second CNN
NUM_GAN_IMAGES_TO_GENERATE = 50000 # e.g., similar to MNIST train size

# Create a directory for CNN outputs
CNN_SAVE_DIR = 'cnn_outputs_pytorch'
if not os.path.exists(CNN_SAVE_DIR):
    os.makedirs(CNN_SAVE_DIR)

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load and Preprocess Original MNIST Data ---
# Transform for CNN: ToTensor normalizes to [0,1]
transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,)) # Optional: MNIST mean/std
])

train_dataset_real = datasets.MNIST('./data', train=True, download=True, transform=transform_cnn)
test_dataset_real = datasets.MNIST('./data', train=False, download=True, transform=transform_cnn)

train_loader_real = DataLoader(train_dataset_real, batch_size=BATCH_SIZE_CNN, shuffle=True, num_workers=4, pin_memory=True)
test_loader_real = DataLoader(test_dataset_real, batch_size=BATCH_SIZE_CNN, shuffle=False, num_workers=4, pin_memory=True)

# --- 2. Define CNN Classifier Model (PyTorch) ---
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 32, kernel_size=3, padding=1) # 28x28x1 -> 28x28x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28x32 -> 14x14x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14x32 -> 14x14x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14x64 -> 7x7x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # 7x7x64 -> 7x7x64
        
        # Calculate flattened size after conv layers
        # For 28x28 input: 7x7x64 = 3136
        self.flattened_size = 64 * 7 * 7 
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x)) # No pooling after last conv
        x = x.view(-1, self.flattened_size) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output raw logits, CrossEntropyLoss will apply softmax
        return x

# Training function for CNN
def train_cnn_model(model, train_loader, optimizer, criterion, epochs, val_loader=None, model_name="CNN"):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = 100. * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        
        log_str = f'Epoch {epoch+1}/{epochs} [{model_name}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%'

        if val_loader:
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs_val, targets_val in val_loader:
                    inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, targets_val)
                    val_loss += loss_val.item()
                    _, predicted_val = outputs_val.max(1)
                    total_val += targets_val.size(0)
                    correct_val += predicted_val.eq(targets_val).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = 100. * correct_val / total_val
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)
            log_str += f' - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%'
        print(log_str)
        
    history = {
        'train_loss': train_losses, 'train_acc': train_accuracies,
        'val_loss': val_losses, 'val_acc': val_accuracies
    }
    return history


# --- 3. Train CNN on Original MNIST Data (Baseline) ---
print("\n--- Training CNN on REAL MNIST data (Baseline - PyTorch) ---")
cnn_baseline = CNNClassifier().to(device)
print(cnn_baseline)
optimizer_baseline = optim.Adam(cnn_baseline.parameters(), lr=0.001)
criterion_cnn = nn.CrossEntropyLoss()

history_baseline = train_cnn_model(cnn_baseline, train_loader_real, optimizer_baseline, criterion_cnn, EPOCHS_CNN, test_loader_real, "BaselineCNN")

# Evaluate baseline
cnn_baseline.eval()
correct_baseline, total_baseline = 0, 0
with torch.no_grad():
    for images, labels in test_loader_real:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_baseline(images)
        _, predicted = torch.max(outputs.data, 1)
        total_baseline += labels.size(0)
        correct_baseline += (predicted == labels).sum().item()
baseline_accuracy = 100 * correct_baseline / total_baseline
print(f"\nBaseline CNN Test Accuracy (trained on real data): {baseline_accuracy:.2f}%")
torch.save(cnn_baseline.state_dict(), os.path.join(CNN_SAVE_DIR, 'cnn_baseline_pytorch.pth'))
print(f"Baseline CNN model saved.")

# Plot baseline CNN training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_baseline['train_acc'], label='Train Accuracy')
plt.plot(history_baseline['val_acc'], label='Val Accuracy')
plt.title('Baseline CNN Accuracy (PyTorch)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history_baseline['train_loss'], label='Train Loss')
plt.plot(history_baseline['val_loss'], label='Val Loss')
plt.title('Baseline CNN Loss (PyTorch)')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CNN_SAVE_DIR, 'cnn_baseline_training_history_pytorch.png'))
plt.close()


# --- 4. Load Trained GAN Generator (PyTorch) ---
# (Generator class definition from gan_mnist_pytorch.py - simplified here)
class Generator(nn.Module): # Copied from gan_mnist_pytorch.py for loading
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input_noise): return self.main(input_noise)

gan_generator_loaded = Generator(GAN_LATENT_DIM, GAN_NGF, IMG_CHANNELS).to(device)
try:
    gan_generator_loaded.load_state_dict(torch.load(os.path.join(GAN_SAVE_DIR, 'gan_generator_pytorch.pth'), map_location=device))
    gan_generator_loaded.eval() # Set to evaluation mode
    print("\nGAN generator weights loaded successfully for image generation (PyTorch).")
except Exception as e:
    print(f"Error loading GAN generator weights: {e}. Make sure 'gan_mnist_pytorch.py' was run.")
    exit()

# --- 5. Generate Synthetic Image Dataset using GAN ---
print(f"\n--- Generating {NUM_GAN_IMAGES_TO_GENERATE} synthetic images using GAN (PyTorch) ---")
x_train_gan_generated_list = []
with torch.no_grad():
    for _ in range(0, NUM_GAN_IMAGES_TO_GENERATE, BATCH_SIZE_CNN):
        current_batch_size = min(BATCH_SIZE_CNN, NUM_GAN_IMAGES_TO_GENERATE - len(x_train_gan_generated_list))
        if current_batch_size <= 0: break
        noise_for_gan = torch.randn(current_batch_size, GAN_LATENT_DIM, 1, 1, device=device)
        generated_batch = gan_generator_loaded(noise_for_gan)
        # GAN output is [-1, 1], CNN expects [0, 1] (if not normalized differently)
        generated_batch_cnn = (generated_batch * 0.5 + 0.5).cpu() # Rescale and move to CPU
        x_train_gan_generated_list.append(generated_batch_cnn)

x_train_gan_generated_cnn = torch.cat(x_train_gan_generated_list, dim=0)
print(f"Shape of generated GAN images for CNN: {x_train_gan_generated_cnn.shape}")

# Display a few generated images
plt.figure(figsize=(10, 2))
for i in range(min(10, x_train_gan_generated_cnn.size(0))):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train_gan_generated_cnn[i].permute(1, 2, 0).squeeze().numpy(), cmap='gray')
    plt.axis('off')
plt.suptitle('Sample GAN-generated Images for CNN Training (PyTorch)')
plt.savefig(os.path.join(CNN_SAVE_DIR, 'sample_gan_generated_for_cnn_pytorch.png'))
plt.close()


# --- 6. Assign Pseudo-Labels to GAN-generated Images ---
print("\n--- Assigning pseudo-labels to GAN-generated images (PyTorch) ---")
y_train_gan_pseudo_labels_list = []
cnn_baseline.eval() # Ensure baseline is in eval mode
with torch.no_grad():
    # Process in batches to avoid OOM if NUM_GAN_IMAGES_TO_GENERATE is large
    temp_gan_dataset = TensorDataset(x_train_gan_generated_cnn) # No labels needed yet
    temp_gan_loader = DataLoader(temp_gan_dataset, batch_size=BATCH_SIZE_CNN, shuffle=False)
    for (gan_images_batch,) in temp_gan_loader: # Unpack tuple
        gan_images_batch = gan_images_batch.to(device)
        outputs_pseudo = cnn_baseline(gan_images_batch)
        _, predicted_pseudo = torch.max(outputs_pseudo, 1)
        y_train_gan_pseudo_labels_list.append(predicted_pseudo.cpu())

y_train_gan_pseudo_labels_tensor = torch.cat(y_train_gan_pseudo_labels_list, dim=0)
print(f"Shape of GAN pseudo-labels: {y_train_gan_pseudo_labels_tensor.shape}")

# Create DataLoader for GAN-generated data with pseudo-labels
train_dataset_gan_labeled = TensorDataset(x_train_gan_generated_cnn, y_train_gan_pseudo_labels_tensor)
train_loader_gan_labeled = DataLoader(train_dataset_gan_labeled, batch_size=BATCH_SIZE_CNN, shuffle=True)


# --- 7. Train CNN on GAN-generated Data ---
print("\n--- Training CNN on GAN-generated data (PyTorch) ---")
cnn_gan_trained = CNNClassifier().to(device) # Fresh instance
optimizer_gan_trained = optim.Adam(cnn_gan_trained.parameters(), lr=0.001)

history_gan_trained = train_cnn_model(cnn_gan_trained, train_loader_gan_labeled, optimizer_gan_trained, criterion_cnn, EPOCHS_CNN, test_loader_real, "GAN-TrainedCNN")

# Evaluate GAN-trained CNN
cnn_gan_trained.eval()
correct_gan_trained, total_gan_trained = 0, 0
with torch.no_grad():
    for images, labels in test_loader_real: # Evaluate on REAL test data
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_gan_trained(images)
        _, predicted = torch.max(outputs.data, 1)
        total_gan_trained += labels.size(0)
        correct_gan_trained += (predicted == labels).sum().item()
gan_trained_accuracy = 100 * correct_gan_trained / total_gan_trained
print(f"\nCNN Test Accuracy (trained on GAN data): {gan_trained_accuracy:.2f}%")
torch.save(cnn_gan_trained.state_dict(), os.path.join(CNN_SAVE_DIR, 'cnn_gan_trained_pytorch.pth'))
print(f"CNN model (trained on GAN data) saved.")

# Plot GAN-trained CNN training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_gan_trained['train_acc'], label='Train Accuracy (on GAN data)')
plt.plot(history_gan_trained['val_acc'], label='Val Accuracy (on Real Test Data)')
plt.title('GAN-Trained CNN Accuracy (PyTorch)')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history_gan_trained['train_loss'], label='Train Loss (on GAN data)')
plt.plot(history_gan_trained['val_loss'], label='Val Loss (on Real Test Data)')
plt.title('GAN-Trained CNN Loss (PyTorch)')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CNN_SAVE_DIR, 'cnn_gan_trained_training_history_pytorch.png'))
plt.close()

# --- 8. Comparison ---
print("\n--- Performance Comparison (PyTorch) ---")
print(f"Baseline CNN Test Accuracy (trained on REAL data): {baseline_accuracy:.2f}%")
print(f"CNN Test Accuracy (trained on GAN-generated data with pseudo-labels): {gan_trained_accuracy:.2f}%")

print("\nCNN classification script (PyTorch) finished.")
