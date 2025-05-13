import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define dataset class
class NCMAPSSDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# Define a simple MLP model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to load a sample of data
def load_sample_data(data_dir, train_units, test_units, sample_size=1000):
    # Load a small sample of training data
    train_samples = []
    train_labels = []
    for unit in train_units[:3]:  # Only use first 3 units to save memory
        npz_file = f"Unit{unit}_win50_str1_smp10.npz"
        data = np.load(os.path.join(data_dir, npz_file))
        unit_samples = data['sample'].transpose(2, 0, 1)  # reshape to (n_samples, window_size, n_features)

        # Take a random sample
        if len(unit_samples) > sample_size:
            indices = np.random.choice(len(unit_samples), sample_size, replace=False)
            unit_samples = unit_samples[indices]
            unit_labels = data['label'][indices]
        else:
            unit_labels = data['label']

        train_samples.append(unit_samples)
        train_labels.append(unit_labels)

    # Concatenate data from sampled training units
    train_samples = np.vstack(train_samples)
    train_labels = np.concatenate(train_labels)

    # Load a small sample of test data
    test_samples = []
    test_labels = []
    for unit in test_units[:2]:  # Only use first 2 units to save memory
        npz_file = f"Unit{unit}_win50_str1_smp10.npz"
        data = np.load(os.path.join(data_dir, npz_file))
        unit_samples = data['sample'].transpose(2, 0, 1)  # reshape to (n_samples, window_size, n_features)

        # Take a random sample
        if len(unit_samples) > sample_size:
            indices = np.random.choice(len(unit_samples), sample_size, replace=False)
            unit_samples = unit_samples[indices]
            unit_labels = data['label'][indices]
        else:
            unit_labels = data['label']

        test_samples.append(unit_samples)
        test_labels.append(unit_labels)

    # Concatenate data from sampled test units
    test_samples = np.vstack(test_samples)
    test_labels = np.concatenate(test_labels)

    return train_samples, train_labels, test_samples, test_labels

# Function to normalize data
def preprocess_data(train_samples, test_samples):
    # Reshape for normalization
    n_train_samples, seq_len, n_features = train_samples.shape
    n_test_samples = test_samples.shape[0]

    train_flat = train_samples.reshape(-1, n_features)
    test_flat = test_samples.reshape(-1, n_features)

    # Normalize
    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_flat)
    test_flat = scaler.transform(test_flat)

    # Reshape back
    train_normalized = train_flat.reshape(n_train_samples, seq_len, n_features)
    test_normalized = test_flat.reshape(n_test_samples, seq_len, n_features)

    return train_normalized, test_normalized

# Function to train model
def train_model(model, train_loader, valid_loader, epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()

        # Calculate average validation loss
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}')

    return model, train_losses, valid_losses

# Function to evaluate model
def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    predictions = []
    actual = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            predictions.extend(output.cpu().numpy())
            actual.extend(target.cpu().numpy())

    # Calculate RMSE
    test_loss = np.sqrt(test_loss / len(test_loader))

    # Plot predictions vs actual
    predictions = np.array(predictions)
    actual = np.array(actual)

    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predictions, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'RUL Prediction (RMSE: {test_loss:.4f})')
    plt.savefig('rul_prediction_lightweight.png')
    plt.close()

    return test_loss

def main():
    # Parameters
    data_dir = "N-CMAPSS/Samples_whole"
    train_units = [2, 5, 10, 16, 18, 20]  # Units for training as mentioned in README
    test_units = [11, 14, 15]             # Units for testing as mentioned in README
    sample_size = 1000                    # Number of samples to take from each unit
    batch_size = 64                       # Smaller batch size to save memory
    seq_len = 50                          # Window size
    n_features = 20                       # Number of features
    input_dim = seq_len * n_features      # Input dimension for MLP
    hidden_dim = 32                       # Smaller hidden layer
    output_dim = 1                        # Output is scalar RUL
    learning_rate = 0.001
    epochs = 10                           # More epochs for the simpler model

    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    train_samples, train_labels, test_samples, test_labels = load_sample_data(
        data_dir, train_units, test_units, sample_size)

    print(f"Training samples: {train_samples.shape}, Labels: {train_labels.shape}")
    print(f"Test samples: {test_samples.shape}, Labels: {test_labels.shape}")

    # Normalize data
    print("Preprocessing data...")
    train_normalized, test_normalized = preprocess_data(train_samples, test_samples)

    # Split training data into train and validation sets
    valid_split = 0.2
    split_idx = int(len(train_normalized) * (1 - valid_split))

    train_data = train_normalized[:split_idx]
    train_labels_split = train_labels[:split_idx]
    valid_data = train_normalized[split_idx:]
    valid_labels = train_labels[split_idx:]

    # Create datasets and dataloaders
    train_dataset = NCMAPSSDataset(train_data, train_labels_split)
    valid_dataset = NCMAPSSDataset(valid_data, valid_labels)
    test_dataset = NCMAPSSDataset(test_normalized, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = SimpleModel(input_dim, hidden_dim, output_dim).to(device)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Train model
    print("Training model...")
    model, train_losses, valid_losses = train_model(model, train_loader, valid_loader, epochs, learning_rate, device)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history_lightweight.png')
    plt.close()

    # Evaluate model
    print("Evaluating model...")
    test_loss = evaluate_model(model, test_loader, device)
    print(f"Test RMSE: {test_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), 'rul_model_lightweight.pth')
    print("Model saved to rul_model_lightweight.pth")

if __name__ == "__main__":
    main()
