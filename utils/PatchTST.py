import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# -----------------------------
# Data loading and preparation
# -----------------------------
def load_and_preprocess_data(path, seq_len, num_samples, test_samples, target_idx=9):
    df = pd.read_csv(path)
    df = df.drop(columns=["date", "Subset"], errors='ignore')
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    data = df.to_numpy()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(data[:num_samples])
    test_scaled = scaler.transform(data[num_samples:num_samples + test_samples])

    def embed(series):
        x, y = [], []
        for i in range(len(series) - 2 * seq_len):
            x.append(series[i:i + seq_len])
            y.append(series[i + seq_len:i + 2 * seq_len, target_idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    x_train, y_train = embed(train_scaled)
    x_test, y_test = embed(test_scaled)
    return x_train, y_train, x_test, y_test, scaler, data[:, target_idx]

# -----------------------------
# Dataset
# -----------------------------
class PatchTSTDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# -----------------------------
# PatchTST Model
# -----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(in_channels * patch_size, embed_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, num_variables)
        batch_size, seq_len, num_vars = x.shape
        x = x.permute(0, 2, 1)  # (batch, num_vars, seq_len)
        # reshape into patches along time dimension
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)  # (batch, num_vars, num_patches, patch_size)
        x = x.contiguous().view(batch_size, num_vars, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # (batch, num_patches, num_vars, patch_size)
        x = x.reshape(batch_size, self.num_patches, num_vars * self.patch_size)  # flatten patches
        x = self.proj(x)  # (batch, num_patches, embed_dim)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class PatchTST(nn.Module):
    def __init__(self, num_variables, seq_len, patch_size=16, embed_dim=128,
                 num_layers=3, num_heads=4, output_steps=24, dropout=0.1):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.patch_embed = PatchEmbedding(num_variables, patch_size, embed_dim, seq_len)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len // patch_size, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim * (seq_len // patch_size), output_steps)

    def forward(self, x):
        # x shape: (batch, seq_len, num_vars)
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.norm(x)
        x = torch.flatten(x, start_dim=1)
        out = self.head(x)
        return out

# -----------------------------
# Training and Evaluation
# -----------------------------
def train(model, train_loader, device, epochs=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

def predict(model, data_loader, device):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            all_preds.append(y_pred.cpu())
            all_trues.append(y.cpu())
    return torch.cat(all_preds, dim=0), torch.cat(all_trues, dim=0)

# -----------------------------
# Plotting functions (reuse yours)
# -----------------------------
import matplotlib.pyplot as plt

def plot_forecast(pred, true, title="Forecast vs Actual"):
    steps = range(len(pred))
    plt.figure(figsize=(8, 4))
    plt.plot(steps, true, label="Actual", marker='o')
    plt.plot(steps, pred, label="Predicted", marker='x')
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Value (e.g. EUR/MWh)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_on_series(full_series, y_preds, y_trues, input_len, output_len, sample_rate=10):
    num_samples = len(y_preds)
    plt.figure(figsize=(15, 6))
    plt.plot(full_series, label="Full time series (Germany)", color='black')
    for i in range(0, num_samples, max(1, num_samples // sample_rate)):
        input_range = range(i, i + input_len)
        forecast_range = range(i + input_len, i + input_len + output_len)
        plt.plot(input_range, full_series[i:i + input_len], color='blue', alpha=0.3)
        plt.plot(forecast_range, y_preds[i].numpy(), color='red', marker='x', linestyle='--', alpha=0.8)
        plt.plot(forecast_range, y_trues[i].numpy(), color='green', marker='o', linestyle='--', alpha=0.8)
    plt.xlabel("Time Steps")
    plt.ylabel("Electricity Value (EUR/MWh)")
    plt.title("Full Time Series and Forecasts for Germany (Target variable index 9)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sample(full_series, y_preds, y_trues, sample_idx, input_len, output_len):
    input_start = sample_idx
    input_end = sample_idx + input_len
    forecast_start = input_end
    forecast_end = forecast_start + output_len

    plt.figure(figsize=(12, 5))
    plt.plot(full_series, label="Full time series", color='black')
    plt.plot(range(input_start, input_end), full_series[input_start:input_end],
             label="Input window", marker='o', color='blue')
    plt.plot(range(forecast_start, forecast_end), y_preds[sample_idx].numpy(),
             label="Predictions", marker='x', color='red')
    plt.plot(range(forecast_start, forecast_end), y_trues[sample_idx].numpy(),
             label="True future", marker='s', color='green')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Sample {sample_idx}: Inputs, Predictions, and True Future aligned")
    plt.legend()
    plt.grid(True)
    plt.show()