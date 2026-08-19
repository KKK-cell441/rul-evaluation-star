"""
PatchTST-style model for RUL Prediction.
Simplified implementation: patch embedding + Transformer encoder + regression head.
Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," ICLR 2023.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Patchify input sequence into overlapping/non-overlapping patches."""
    def __init__(self, seq_len, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, x):
        # x: (B, L, D) where D=features
        B, L, D = x.shape
        # Patch per feature channel
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, num_patches, D, patch_len)
        patches = patches.permute(0, 2, 1, 3)  # (B, D, num_patches, patch_len)
        # Process each feature independently
        B, D, P, PL = patches.shape
        patches = patches.reshape(B * D, P, PL)
        x = self.proj(patches) + self.pos_embed  # (B*D, P, d_model)
        return x, B, D


class PatchTSTRULModel(nn.Module):
    """PatchTST for RUL prediction."""
    def __init__(self, input_dim=10, seq_len=10, patch_len=4, stride=2,
                 d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        self.patch_embed = PatchEmbedding(seq_len, patch_len, stride, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model * self.num_patches * input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (B, L, D)
        patches, B, D = self.patch_embed(x)  # (B*D, P, d_model)
        out = self.transformer(patches)  # (B*D, P, d_model)
        out = self.layer_norm(out)  # (B*D, P, d_model)
        # Reshape back
        P = self.num_patches
        d = out.shape[-1]
        out = out.reshape(B, D * P * d)  # (B, D*P*d_model)
        return self.head(out).squeeze(-1)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
