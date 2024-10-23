import torch.nn as nn
import torch
from training.utils.constants import max_sequence_length


# Define Chomp1d to adjust sequence lengths
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


# Define TCN components with adjusted padding and Chomp1d
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        # num_channels: list of channel sizes for each layer
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # exponential dilation
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Define the Temporal Convolutional Transformer Model
class TemporalConvolutionalTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim=128, tcn_channels=[64, 64],
                 kernel_size=2, dropout=0.2):
        super(TemporalConvolutionalTransformerModel, self).__init__()

        # TCN module
        self.tcn = TemporalConvNet(input_dim, tcn_channels, kernel_size=kernel_size, dropout=dropout)

        # Linear layer to adjust dimension to hidden_dim
        self.linear = nn.Linear(tcn_channels[-1], hidden_dim)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_sequence_length, hidden_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: batch_size x seq_len x input_dim

        # TCN expects input of shape (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)

        # Pass through TCN
        x = self.tcn(x)  # Output shape: (batch_size, channels, seq_len)

        # Permute back to (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)

        # Adjust dimensions
        x = self.linear(x)  # Output shape: (batch_size, seq_len, hidden_dim)

        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Pass through Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.fc(x)

        return x