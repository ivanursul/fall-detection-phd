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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(torch.__version__)

# Constants
MAX_SEQUENCE_LENGTH = 800  # Adjust based on your dataset
num_classes = 2  # Fall or non-fall
num_epochs = 10  # Reduced to speed up the hyperparameter search
input_dim = 4
num_trials = 500

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
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%')

# Training loop with Early Stopping
def train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs, patience=5):
    best_loss = float('inf')
    best_model_weights = None
    early_stop_count = 0

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
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%')

        # Evaluate on validation (test) data
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            correct_val = 0
            total_val = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = 100 * correct_val / total_val
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = model.state_dict().copy()
            early_stop_count = 0  # reset early stopping counter
        else:
            early_stop_count += 1

        # Stop training if patience is exceeded
        if early_stop_count >= patience:
            print("Early stopping triggered")
            break

    # Restore the best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

# Evaluation function
def evaluate_model(model, test_loader, return_acc=False):
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
    if return_acc:
        return accuracy

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Fall', 'Fall']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Expanded valid combinations of num_heads and hidden_dim
# valid_combinations = [
#     # Smaller combinations
#     (4, 32), (4, 64), (4, 128),
#     # Original combinations
#     (8, 64), (8, 128), (8, 256),
#     (16, 128), (16, 256), (16, 512),
#     (32, 256), (32, 512), (32, 1024),
#     # Larger combinations
#     (64, 512), (64, 1024), (64, 2048),
#     (128, 1024), (128, 2048)
# ]

valid_combinations = [
    (4, 4),(4, 8), (4, 16),(4, 32),(4, 64),(4, 128),
]


def objective(trial):
    # Select valid combination of num_heads and hidden_dim
    num_heads, hidden_dim = trial.suggest_categorical('num_heads_hidden_dim', valid_combinations)

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
    accuracy = evaluate_model(model, test_loader, return_acc=True)
    return accuracy

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=num_trials)

# Best hyperparameters found by Optuna
print(f'Best trial: {study.best_trial.params}')

# Evaluate the best model with the tuned hyperparameters
best_params = study.best_trial.params
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

best_model, best_optimizer = create_model(input_dim=input_dim, num_heads=best_params['num_heads'], num_layers=best_params['num_layers'],
                                          hidden_dim=best_params['hidden_dim'], dropout=best_params['dropout'], learning_rate=best_params['learning_rate'])
best_model = best_model.to(device)
train_model(best_model, train_loader, nn.CrossEntropyLoss(), best_optimizer, epochs=num_epochs)
evaluate_model(best_model, test_loader)
