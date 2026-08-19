"""
Temporal Convolutional Network for RUL Prediction.
Reference: Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling," arXiv:1803.01271, 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Remove padding elements for causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation,
                 padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation, padding=padding,
                dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        return out.transpose(1, 2)


class TCNRULModel(nn.Module):
    """TCN for RUL prediction. Takes (B, L, D) and outputs (B, 1)."""
    def __init__(self, input_dim=10, seq_len=10, num_channels=[32, 64, 128],
                 kernel_size=3, dropout=0.2, hidden_dim=64):
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(num_channels[-1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, L, D)
        out = self.tcn(x)  # (B, L, C)
        out = out.transpose(1, 2)  # (B, C, L)
        out = self.global_pool(out).squeeze(-1)  # (B, C)
        return self.regressor(out).squeeze(-1)  # (B,)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
