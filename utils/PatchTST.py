import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# Data loading and preparation
# -----------------------------
def load_patchtst_data(path, seq_len, num_samples, test_samples, target_idx=9):
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
        for i in range(len(series) - seq_len):
            x.append(series[i:i + seq_len])
            y.append(series[i + seq_len, target_idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    x_train, y_train = embed(train_scaled)
    x_test, y_test = embed(test_scaled)
    return x_train, y_train, x_test, y_test, scaler, data[:, target_idx]

# -----------------------------
# Full PatchTST Model
# -----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_len, input_dim):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.proj = nn.Linear(patch_len * input_dim, 128)

    def forward(self, x):
        B, T, C = x.shape
        x = x[:, :self.n_patches * self.patch_len, :]
        x = x.view(B, self.n_patches, self.patch_len * C)
        return self.proj(x)

class PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, patch_len=16, n_heads=4, n_layers=3):
        super().__init__()
        self.embedding = PatchEmbedding(seq_len, patch_len, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=n_heads, dim_feedforward=256, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)  # [B, P, 128]
        x = self.encoder(x)    # [B, P, 128]
        x = x.mean(dim=1)      # global average pooling over patches
        return self.head(x).squeeze()

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
# Training
# -----------------------------
def train_patchtst(model, train_loader, device, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, test_loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            preds.append(y_pred.cpu())
            trues.append(y.cpu())
    return torch.cat(preds), torch.cat(trues)