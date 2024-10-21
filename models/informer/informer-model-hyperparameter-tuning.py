import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import optuna  # Import Optuna
import logging  # Import logging module

# Set up logging
logger = logging.getLogger('T2VBERTModel')
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('training.log')

# Set level for handlers
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Constants
MAX_SEQUENCE_LENGTH = 800  # Adjust based on your dataset
input_dim = 4  # AccX, AccY, AccZ, Acc_magnitude_i
num_classes = 2  # Fall or non-fall


# Custom Dataset class
class FallDetectionDataset(Dataset):
    def __init__(self, file_paths, labels, scaler=None):
        self.file_paths = file_paths
        self.labels = labels
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = pd.read_csv(self.file_paths[idx])
        acc_data = data[['AccX_filtered_i', 'AccY_filtered_i', 'Corrected_AccZ_i', 'Acc_magnitude_i']].values

        # Apply scaling
        acc_data = self.scaler.transform(acc_data)

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


# Replace these paths with your dataset paths
fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\Fall'
non_fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\ADL'

file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Split data into training, validation, and test sets
train_files, temp_files, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=0.3, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42)

# Fit scaler on training data
scaler = StandardScaler()
all_train_data = []
for file in train_files:
    data = pd.read_csv(file)
    acc_data = data[['AccX_filtered_i', 'AccY_filtered_i', 'Corrected_AccZ_i', 'Acc_magnitude_i']].values
    all_train_data.append(acc_data)
all_train_data = np.vstack(all_train_data)
scaler.fit(all_train_data)

# Create PyTorch datasets
train_dataset = FallDetectionDataset(train_files, train_labels, scaler=scaler)
val_dataset = FallDetectionDataset(val_files, val_labels, scaler=scaler)
test_dataset = FallDetectionDataset(test_files, test_labels, scaler=scaler)


def get_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
    return pe


class InformerClassificationModel(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=128, n_heads=4, e_layers=2, dropout=0.1):
        super(InformerClassificationModel, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model

        # Embedding layers
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = get_positional_encoding(seq_len, d_model)

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # Embed input
        x = self.value_embedding(x)  # Shape: (batch_size, seq_len, d_model)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)

        # Permute for transformer input: (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # Pass through encoder
        x = self.encoder(x)  # Shape: (seq_len, batch_size, d_model)

        # Global average pooling
        x = x.mean(dim=0)  # Shape: (batch_size, d_model)

        # Classification layer
        x = self.fc(x)  # Shape: (batch_size, num_classes)

        return x

def log_model_size(model):
    # Calculate the total number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())

    # Assume each parameter is a float32 (4 bytes)
    param_size_in_bytes = num_params * 4

    # Convert bytes to megabytes
    model_size_in_mb = param_size_in_bytes / (1024 * 1024)

    return model_size_in_mb

d_model_num_heads_map = {
    '64_1': (64, 1), '64_2': (64, 2), '64_4': (64, 4),
    '128_1': (128, 1), '128_2': (128, 2), '128_4': (128, 4),
    '256_2': (256, 2), '256_4': (256, 4), '256_8': (256, 8),
    '512_4': (512, 4), '512_8': (512, 8)
}

def objective(trial):
    # Suggest d_model and num_heads as a string key, then map it to a tuple
    d_model_num_heads_str = trial.suggest_categorical('d_model_num_heads', list(d_model_num_heads_map.keys()))
    d_model, num_heads = d_model_num_heads_map[d_model_num_heads_str]

    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Create DataLoaders with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Build the model with hyperparameters
    model = InformerClassificationModel(
        input_dim=input_dim,
        seq_len=MAX_SEQUENCE_LENGTH,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=num_heads,
        e_layers=num_layers,
        dropout=dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10  # Fewer epochs for hyperparameter optimization

    for epoch in range(num_epochs):
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

        # Evaluate on validation set
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())



        # Calculate validation accuracy
        val_acc = accuracy_score(val_labels, val_preds)

        # Log the training and validation results
        logger.info(f'Trial {trial.number}, Epoch {epoch + 1}/{num_epochs}, '
                    f'Train Loss: {running_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%')

        # Report intermediate objective value (validation accuracy) to Optuna
        trial.report(val_acc, epoch)

        # Handle pruning
        if trial.should_prune():
            logger.info(f'Trial {trial.number} pruned at epoch {epoch + 1}')
            raise optuna.exceptions.TrialPruned()

    # Log the model size after training
    model_size_in_mb = log_model_size(model)
    logger.info(f"Model size after training: {model_size_in_mb:.2f} MB")

    # If model size exceeds 5 MB, prune the trial after training
    if model_size_in_mb > 5:
        logger.info(f"Pruning trial {trial.number} because model size exceeds 5 MB after training ({model_size_in_mb:.2f} MB)")
        raise optuna.exceptions.TrialPruned()

    return val_acc  # Use validation accuracy as the objective for Optuna


# Run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1500)

# Retrieve the best hyperparameters
best_params = study.best_trial.params
logger.info(f"Best hyperparameters: {best_params}")

best_d_model_num_heads = best_params['d_model_num_heads']
best_d_model, best_num_heads = d_model_num_heads_map[best_d_model_num_heads]
best_num_layers = best_params['num_layers']
best_dropout = best_params['dropout']
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']

# Combine train and validation datasets
combined_train_files = train_files + val_files
combined_train_labels = train_labels + val_labels
combined_train_dataset = FallDetectionDataset(combined_train_files, combined_train_labels, scaler=scaler)

# Create DataLoaders with the best batch size
train_loader = DataLoader(combined_train_dataset, batch_size=best_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size)

# Build the model with the best hyperparameters
best_model = InformerClassificationModel(
    input_dim=input_dim,
    seq_len=MAX_SEQUENCE_LENGTH,
    num_classes=num_classes,
    d_model=best_d_model,
    n_heads=best_num_heads,
    e_layers=best_num_layers,
    dropout=best_dropout
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)

# Retrain the model with best hyperparameters
num_epochs = 50  # Increase epochs for final training

for epoch in range(num_epochs):
    best_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = best_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.2f}%')

# Evaluate the model on the test set
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
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    logger.info("Classification Report:")
    logger.info(classification_report(all_labels, all_preds, target_names=['Non-Fall', 'Fall']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Evaluate the best model
evaluate_model(best_model, test_loader)
