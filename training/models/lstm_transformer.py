import torch.nn as nn
import torch

class LongShortTermTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, max_sequence_length, hidden_dim=128, conv_hidden_dim=64, dropout=0.1):
        super(LongShortTermTransformerModel, self).__init__()

        # Short-term component: Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=conv_hidden_dim, out_channels=conv_hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Linear layer to match the transformer d_model
        self.linear = nn.Linear(conv_hidden_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_sequence_length, hidden_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification layers
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: batch_size x seq_len x input_dim

        # Convolutional layers expect input of shape (batch_size, in_channels, seq_len)
        x = x.permute(0, 2, 1)  # reshape to (batch_size, input_dim, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # reshape back to (batch_size, seq_len, conv_hidden_dim)

        # Linear layer to match transformer input size
        x = self.linear(x)  # x shape: batch_size x seq_len x hidden_dim

        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Transformer layers
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification layer
        x = self.fc(x)

        return x

