import os
import optuna
import psutil
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from linformer import Linformer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import logging


def create_logger():
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create file handler which logs to a file
    file_handler = logging.FileHandler("linformer-hyperparameter-tuning.log")
    file_handler.setLevel(logging.INFO)
    # Create console handler with the same logging level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Add both handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


# Set up logging
create_logger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # For CPU-focused tuning
print(f'Using device: {device}')

print(torch.__version__)

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

        # Padding
        if len(acc_data) < MAX_SEQUENCE_LENGTH:
            padded = np.zeros((MAX_SEQUENCE_LENGTH, 4))
            padded[:len(acc_data), :] = acc_data
        else:
            padded = acc_data[:MAX_SEQUENCE_LENGTH, :]

        tensor_data = torch.tensor(padded, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_data, label

# Load dataset
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
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

num_epochs = 10
MAX_SEQUENCE_LENGTH = 800
num_trias = 100

# Linformer-based Transformer Model
class LinformerTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim=128, dropout=0.1):
        super(LinformerTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQUENCE_LENGTH, hidden_dim))
        self.linformer = Linformer(
            dim=hidden_dim,
            seq_len=MAX_SEQUENCE_LENGTH,
            depth=num_layers,
            heads=num_heads,
            k=256,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.linformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

valid_combinations = [
    (4, 4),(4, 8), (4, 16),(4, 32),(4, 64),(4, 128),
]

# Objective function for Optuna
def objective(trial):

    #num_heads, hidden_dim = trial.suggest_categorical('num_heads_hidden_dim', valid_combinations)
    num_heads = trial.suggest_categorical('num_heads', [4])
    hidden_dim = trial.suggest_categorical('hidden_dim', [4, 8, 16, 32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 16, 64)

    logger.info(f"Trial {trial.number}: num_heads={num_heads}, num_layers={num_layers}, hidden_dim={hidden_dim}, "
                f"dropout={dropout}, learning_rate={learning_rate}, batch_size={batch_size}")

    # Create DataLoader
    train_dataset = FallDetectionDataset(train_files, train_labels)
    test_dataset = FallDetectionDataset(test_files, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = LinformerTransformerModel(input_dim=4, num_heads=num_heads, num_layers=num_layers,
                                      num_classes=2, hidden_dim=hidden_dim, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss:.4f}")

    # Save model to disk and measure model size
    model_file_path = "model.pth"
    torch.save(model.state_dict(), model_file_path)
    model_size = os.path.getsize(model_file_path) / (1024 ** 2)

    # Model evaluation
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    # Measure time taken for training
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds.")
    logger.info(f"Model size: {model_size:.2f} MB, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

    # Return multiple objectives: CPU consumption, accuracy, precision, and memory consumption
    return accuracy

# Create Optuna study
study = optuna.create_study(directions=["maximize"])  # For multi-objective
study.optimize(objective, n_trials=num_trias)

# Print best hyperparameters
best_trial = study.best_trials[0]
logger.info("Best trial:")
logger.info(f"Trial {best_trial.number}")
logger.info(f"Hyperparameters: {best_trial.params}")
logger.info(f" CPU Usage: {best_trial.values[0]}%")
logger.info(f" Accuracy: {best_trial.values[1]}")
logger.info(f" Precision: {best_trial.values[2]}")

