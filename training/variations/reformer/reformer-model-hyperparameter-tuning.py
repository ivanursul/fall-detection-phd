import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import logging  # For logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import optuna
from reformer_pytorch import Reformer  # Import Reformer from reformer-pytorch

# Set up logging
logger = logging.getLogger('ReformerModel')
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('reformer_training.log')

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
input_dim = 4  # AccX_filtered_i, AccY_filtered_i, Corrected_AccZ_i, Acc_magnitude_i
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

# First Split: Split data into training and test sets
train_files, test_files, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42
)

# Second Split: Split training data into training and validation sets
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.125, random_state=42
)
# Since the training set is now 80% of the data, a test_size of 0.125 gives a validation set of 10% of the total data

# Verify the proportions
total_samples = len(file_paths)
train_percentage = len(train_files) / total_samples * 100
val_percentage = len(val_files) / total_samples * 100
test_percentage = len(test_files) / total_samples * 100
logger.info(f'Data Split - Train: {train_percentage:.2f}%, Validation: {val_percentage:.2f}%, Test: {test_percentage:.2f}%')

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

# Reformer Model
class ReformerModelCustom(nn.Module):
    def __init__(self, input_dim, num_classes, model_dim=128, depth=6, heads=8, bucket_size=64, lsh_dropout=0.1, causal=False):
        super(ReformerModelCustom, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.reformer = Reformer(
            dim = model_dim,
            depth = depth,
            heads = heads,
            bucket_size = bucket_size,
            lsh_dropout = lsh_dropout,
            causal = causal
        )
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.reformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

def objective(trial):
    # Hyperparameters to optimize
    model_dim = trial.suggest_categorical('model_dim', [64, 128, 256, 512])
    depth = trial.suggest_int('depth', 1, 6)
    heads = trial.suggest_categorical('heads', [1, 2, 4, 8])
    bucket_size = trial.suggest_categorical('bucket_size', [32, 64, 128, 256])
    lsh_dropout = trial.suggest_float('lsh_dropout', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Create DataLoaders with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Build the model with hyperparameters
    model = ReformerModelCustom(
        input_dim=input_dim,
        num_classes=num_classes,
        model_dim=model_dim,
        depth=depth,
        heads=heads,
        bucket_size=bucket_size,
        lsh_dropout=lsh_dropout,
        causal=False  # Set to True if sequence is causal
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
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        # Log the training and validation results
        logger.info(f'Trial {trial.number}, Epoch {epoch + 1}/{num_epochs}, '
                    f'Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        # Report intermediate objective value to Optuna
        trial.report(val_acc, epoch)

        # Handle pruning
        if trial.should_prune():
            logger.info(f'Trial {trial.number} pruned at epoch {epoch + 1}')
            raise optuna.exceptions.TrialPruned()

    return val_acc

# Run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Retrieve the best hyperparameters
best_params = study.best_trial.params
logger.info(f"Best hyperparameters: {best_params}")

best_model_dim = best_params['model_dim']
best_depth = best_params['depth']
best_heads = best_params['heads']
best_bucket_size = best_params['bucket_size']
best_lsh_dropout = best_params['lsh_dropout']
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
best_model = ReformerModelCustom(
    input_dim=input_dim,
    num_classes=num_classes,
    model_dim=best_model_dim,
    depth=best_depth,
    heads=best_heads,
    bucket_size=best_bucket_size,
    lsh_dropout=best_lsh_dropout,
    causal=False  # Adjust if needed
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
