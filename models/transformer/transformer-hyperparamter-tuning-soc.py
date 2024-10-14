import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')
logger.info(torch.__version__)

# Constants
MAX_SEQUENCE_LENGTH = 800  # Adjust based on your dataset
num_classes = 2  # Fall or non-fall
num_epochs = 10  # Reduced to speed up the hyperparameter search
input_dim = 4
num_trials = 100

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

        acc_data = data[['AccX_filtered_i', 'AccY_filtered_i'
                         # , 'AccZ_filtered_i'
            , 'Corrected_AccZ_i'
                         # ,'AsY(°/s)', 'AsZ(°/s)', 'AsZ(°/s)',
            , 'Acc_magnitude_i'
        ]].values

        acc_data = self.scaler.fit_transform(acc_data)
        add_data_final = acc_data

        if len(add_data_final) < MAX_SEQUENCE_LENGTH:
            padded = np.zeros((MAX_SEQUENCE_LENGTH, input_dim))
            padded[:len(add_data_final), :] = add_data_final
        else:
            padded = add_data_final[:MAX_SEQUENCE_LENGTH, :]
        tensor_data = torch.tensor(padded, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_data, label

# Load file paths and labels
def load_dataset(fall_folder, non_fall_folder):
    # Function to filter only .csv and .txt files
    def filter_files(folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv') or f.endswith('.txt')]

    # Get filtered file paths
    fall_files = filter_files(fall_folder)
    non_fall_files = filter_files(non_fall_folder)

    # Combine fall and non-fall file paths
    file_paths = fall_files + non_fall_files

    # Assign labels (1 for fall, 0 for non-fall)
    labels = [1] * len(fall_files) + [0] * len(non_fall_files)

    return file_paths, labels

fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\Fall'
non_fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\ADL'

file_paths, labels = load_dataset(fall_folder, non_fall_folder)
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

train_dataset = FallDetectionDataset(train_files, train_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)

# Define the model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQUENCE_LENGTH, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Function to create the model with specific hyperparameters
def create_model(input_dim, num_heads, num_layers, hidden_dim, dropout, learning_rate):
    model = TransformerModel(input_dim=input_dim, num_heads=num_heads, num_layers=num_layers,
                             num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=num_epochs):
    model.train()
    for epoch in range(epochs):
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
        logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%')

# Evaluation function
def evaluate_model(model, test_loader, return_metrics=False):
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
    precision = precision_score(all_labels, all_preds, average='macro')

    if return_metrics:
        return accuracy, precision

    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    logger.info("Classification Report:")
    logger.info(classification_report(all_labels, all_preds, target_names=['Non-Fall', 'Fall']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Function to get the model size on disk in MB
def get_model_size(model):
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_in_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    return size_in_mb

valid_combinations = [
    (4, 4), (4, 8), (4, 16), (4, 32), (4, 64), (4, 128),
]

def objective(trial):
    # Select valid combination of num_heads and hidden_dim
    num_heads_hidden_dim = trial.suggest_categorical('num_heads_hidden_dim', valid_combinations)
    num_heads, hidden_dim = num_heads_hidden_dim

    # Tune other hyperparameters
    num_layers = trial.suggest_int('num_layers', 2, 4)  # Number of transformer layers
    dropout = trial.suggest_float('dropout', 0.01, 0.1)  # Dropout rate
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)  # Learning rate
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])  # Batch size

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model and optimizer
    model, optimizer = create_model(input_dim=input_dim, num_heads=num_heads, num_layers=num_layers,
                                    hidden_dim=hidden_dim, dropout=dropout, learning_rate=learning_rate)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Train model
    train_model(model, train_loader, criterion, optimizer, epochs=num_epochs)

    # Evaluate model
    accuracy, precision = evaluate_model(model, test_loader, return_metrics=True)

    # Save model to disk and measure model size
    model_file_path = "model.pth"
    torch.save(model.state_dict(), model_file_path)
    model_size = os.path.getsize(model_file_path) / (1024 ** 2)

    logger.info(f"Model size: {model_size:.2f} MB, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

    # Return the objectives: maximize accuracy and precision, minimize model size
    return accuracy

# Run Optuna multi-objective optimization
study = optuna.create_study(directions=['maximize'])
study.optimize(objective, n_trials=num_trials)

# Best hyperparameters found by Optuna
logger.info('Number of finished trials:', len(study.trials))

# Get the Pareto-optimal trials
pareto_trials = study.best_trials

logger.info('Pareto-optimal trials:')
for trial in pareto_trials:
    logger.info(f'Trial number: {trial.number}')
    logger.info(f'  Accuracy: {trial.values[0]:.4f}')
    logger.info('  Params:')
    for key, value in trial.params.items():
        logger.info(f'    {key}: {value}')

# Choose the best trial based on your preference
best_trial = pareto_trials[0]  # For example, select the first Pareto-optimal trial

# Evaluate the best model with the tuned hyperparameters
best_params = best_trial.params
num_heads, hidden_dim = best_params['num_heads_hidden_dim']
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

best_model, best_optimizer = create_model(
    input_dim=input_dim,
    num_heads=num_heads,
    num_layers=best_params['num_layers'],
    hidden_dim=hidden_dim,
    dropout=best_params['dropout'],
    learning_rate=best_params['learning_rate']
)
best_model = best_model.to(device)
train_model(best_model, train_loader, nn.CrossEntropyLoss(), best_optimizer, epochs=num_epochs)
evaluate_model(best_model, test_loader)
