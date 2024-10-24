import torch
import psutil
import time
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from memory_profiler import memory_usage

from training.models.lstm_transformer import LongShortTermTransformerModel
from training.models.multi_scale_transformer import MultiScaleTransformerModel
from training.models.performer import PerformerModel
from training.models.t2v_bert import T2VBERTModel
from training.models.temporal_convolutional_transformer import TemporalConvolutionalTransformerModel
from training.models.transformer import TransformerModel
from training.utils.dataset_utils import load_dataset
from training.dataset.fall_detection_dataset import FallDetectionDataset
from training.utils.constants import fall_folder, non_fall_folder, input_dim, num_classes, max_sequence_length
from training.models.informer import InformerClassificationModel
from training.models.linformer import LinformerTransformerModel

# Define a dictionary for the models you want to evaluate
model_classes = {
    'Informer': {
        'class': InformerClassificationModel,
        'params': {
            'input_dim': input_dim,
            'seq_len': max_sequence_length,
            'num_classes': num_classes,
            'd_model': 128,
            'n_heads': 4,
            'e_layers': 1,
            'dropout': 0.1
        }
    },
    'Linformer': {
        'class': LinformerTransformerModel,
        'params': {
            'input_dim': input_dim,
            'num_heads': 4,
            'num_layers': 1,
            'hidden_dim': 128,
            'num_classes': num_classes,
            'dropout': 0.5,
            'max_sequence_length': max_sequence_length
        }
    },
    'LongShortTermTransformer': {
        'class': LongShortTermTransformerModel,
        'params': {
            'input_dim': input_dim,
            'num_heads': 8,
            'num_layers': 2,
            'num_classes': num_classes,
            'max_sequence_length': max_sequence_length,
            'hidden_dim': 24,
            'dropout': 0.3
        }
    },
    'MultiScaleTransformer': {
        'class': MultiScaleTransformerModel,
        'params': {
            'input_dim': input_dim,
            'num_heads': 8,
            'num_layers': 2,
            'num_classes': num_classes,
            'hidden_dim': 48,
            'dropout': 0.2
        }
    },
    'Performer': {
        'class': PerformerModel,
        'params': {
            'input_dim': input_dim,
            'num_heads': 4,
            'num_layers': 4,
            'num_classes': num_classes,
            'hidden_dim': 128,
            'dropout': 0.2
        }
    },
    'TemporalConvolutionalTransformer': {
        'class': TemporalConvolutionalTransformerModel,
        'params': {
            'input_dim': input_dim,
            'num_heads': 16,
            'num_layers': 2,
            'num_classes': num_classes,
            'hidden_dim': 64,
            'tcn_channels': [64, 64],
            'dropout': 0.5
        }
    },
    'Transformer': {
        'class': TransformerModel,
        'params': {
            'input_dim': input_dim,
            'num_heads': 4,
            'num_layers': 2,
            'num_classes': num_classes,
            'hidden_dim': 128,
            'dropout': 0.011037393228528439
        }
    },
    'T2VBERT': {
        'class': T2VBERTModel,
        'params': {
            'input_dim': input_dim,
            'num_heads': 4,
            'num_layers': 2,
            'num_classes': num_classes,
            'hidden_dim': 128,
            'dropout': 0.1
        }
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
file_paths, labels = load_dataset(fall_folder, non_fall_folder)
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)
train_dataset = FallDetectionDataset(train_files, train_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Evaluation function
def evaluate_model_performance(model, test_loader):
    y_true, y_pred = [], []
    inference_times = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            inference_time = (end_time - start_time) * 1000
            inference_times.append(inference_time)

            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    inference_time_50th = np.percentile(inference_times, 50)
    inference_time_99th = np.percentile(inference_times, 99)

    return {
        'inference_time_50th_ms': inference_time_50th,
        'inference_time_99th_ms': inference_time_99th,
        'accuracy': accuracy,
        'precision_fall': precision[1],
        'recall_fall': recall[1],
        'f1_fall': f1[1],
        'precision_non_fall': precision[0],
        'recall_non_fall': recall[0],
        'f1_non_fall': f1[0]
    }

# Train the model and measure training time
def train_model(device, model, train_loader, criterion, optimizer, epochs=10):
    start_time = time.time()  # Start time for training

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

    end_time = time.time()  # End time for training
    training_time = end_time - start_time  # Total training time in seconds

    return training_time

# Main loop to train, measure time, and evaluate models
results = []
for model_name, model_info in model_classes.items():
    print(f"Training and evaluating {model_name}")

    model_class = model_info['class']
    model_params = model_info['params']

    # Instantiate model
    model = model_class(**model_params).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model and measure training time
    training_time = train_model(device, model, train_loader, criterion, optimizer, epochs=10)

    # Measure model size
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Convert to MB

    # Measure RAM/CPU usage using memory_profiler and psutil
    ram_before = memory_usage(-1, interval=0.1, timeout=1)
    cpu_before = psutil.cpu_percent(interval=1)

    # Perform evaluation
    performance_metrics = evaluate_model_performance(model, test_loader)

    ram_after = memory_usage(-1, interval=0.1, timeout=1)
    cpu_after = psutil.cpu_percent(interval=1)

    ram_usage = max(ram_after) - min(ram_before)
    cpu_usage = cpu_after - cpu_before

    # Store the results
    results.append({
        'model': model_name,
        'training_time_seconds': training_time,
        'model_size_mb': model_size_mb,
        'cpu_usage_percent': cpu_usage,
        'ram_usage_mb': ram_usage,
        **performance_metrics
    })

# Convert results to a structured format and print the comparison
import pandas as pd
pd.set_option('display.max_columns', None)

df_results = pd.DataFrame(results)
df_results.to_csv('model_evaluation_results.csv', index=False)

print("Results have been exported to model_evaluation_results.csv")
