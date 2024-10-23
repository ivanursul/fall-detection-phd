from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from training.utils.constants import max_sequence_length, csv_columns, input_dim


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
        acc_data = data[csv_columns].values
        acc_data = self.scaler.fit_transform(acc_data)

        if len(acc_data) < max_sequence_length:
            padded = np.zeros((max_sequence_length, input_dim))
            padded[:len(acc_data), :] = acc_data
        else:
            padded = acc_data[:max_sequence_length, :]

        tensor_data = torch.tensor(padded, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_data, label
