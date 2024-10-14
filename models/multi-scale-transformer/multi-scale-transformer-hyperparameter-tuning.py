import optuna
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import logging

# Setup logging
logging.basicConfig(filename='training_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

# Constants
MAX_SEQUENCE_LENGTH = 800
BATCH_SIZE = 32
input_dim = 4  # AccX, AccY, AccZ
num_classes = 2  # Fall or non-fall
num_epochs = 10  # You can reduce for faster tuning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(message):
    logger.info(message)
    print(message)

log(f'Using device: {device}')


# Custom Dataset class
class FallDetectionDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.scaler = StandardScaler()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = pd.read_csv(self.file_paths[idx])
        acc_data = data[['AccX_filtered_i', 'AccY_filtered_i', 'Corrected_AccZ_i', 'Acc_magnitude_i']].values
        acc_data = self.scaler.fit_transform(acc_data)

        if len(acc_data) < MAX_SEQUENCE_LENGTH:
            padded = np.zeros((MAX_SEQUENCE_LENGTH, input_dim))
            padded[:len(acc_data), :] = acc_data
        else:
            padded = acc_data[:MAX_SEQUENCE_LENGTH, :]

        tensor_data = torch.tensor(padded, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_data, label


# Load file paths and labels
def load_dataset(fall_folder, non_fall_folder):
    def filter_files(folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv') or f.endswith('.txt')]

    fall_files = filter_files(fall_folder)
    non_fall_files = filter_files(non_fall_folder)

    file_paths = fall_files + non_fall_files
    labels = [1] * len(fall_files) + [0] * len(non_fall_files)

    return file_paths, labels


fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\Fall'
non_fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\ADL'

file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Split data into training and test sets
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2,
                                                                      random_state=42)

# Create PyTorch datasets and loaders
train_dataset = FallDetectionDataset(train_files, train_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Define model, loss, and optimizer
class MultiScaleTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout):
        super(MultiScaleTransformerModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQUENCE_LENGTH, hidden_dim))

        # Multi-scale attention with hierarchical structures
        self.multi_scale_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Global average pooling for multiple scales
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]

        for transformer_layer in self.multi_scale_transformer_layers:
            x = transformer_layer(x)

        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x


# Train model function
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        log(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%')

        # After every epoch, evaluate the model on the test set
        test_acc = evaluate_model(model, test_loader)
        log(f'Test Accuracy: {test_acc:.2f}%')

    return test_acc


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy * 100

valid_combinations = [
    (4, 4),(4, 8), (4, 16),(4, 32),(4, 64),(4, 128),
]

# Optuna objective function
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)

    #hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    #num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    num_heads, hidden_dim = trial.suggest_categorical('num_heads_hidden_dim', valid_combinations)

    dropout = trial.suggest_uniform('dropout', 0.01, 0.5)
    num_layers = trial.suggest_int('num_layers', 1, 2, 4)

    # Log current trial parameters
    log(f"Trial {trial.number}: LR: {learning_rate}, Dropout: {dropout}, Hidden Dim: {hidden_dim}, "
                f"Num Heads: {num_heads}, Num Layers: {num_layers}")

    # Initialize model, optimizer, and loss function with trial hyperparameters
    model = MultiScaleTransformerModel(input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    test_acc = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)

    return test_acc


# Main optuna function
if __name__ == '__main__':
    # Create a study object
    study = optuna.create_study(direction='maximize')  # We aim to maximize accuracy

    # Optimize the study using 50 trials
    study.optimize(objective, n_trials=1000)

    # Print the best trial and hyperparameters
    log('Best trial:')
    trial = study.best_trial

    log(f'Accuracy: {trial.value}')
    log('Best hyperparameters: ')
    for key, value in trial.params.items():
        log(f'{key}: {value}')
