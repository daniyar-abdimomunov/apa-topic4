# PatchTST.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn


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
                 num_layers=3, num_heads=4, output_steps=24, dropout=0.1, return_embeddings=False):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.return_embeddings = return_embeddings

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

        if self.return_embeddings:
            return x
        else:
         return self.head(x)

    # -----------------------------
    # Training and Evaluation
    # -----------------------------

    def fit(self, train_loader, device, epochs=10, lr=1e-3):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(self(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    def predict(self, data_loader, device):
        self.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                y_pred = self(x)
                all_preds.append(y_pred.cpu())
                all_trues.append(y.cpu())
        return torch.cat(all_preds, dim=0), torch.cat(all_trues, dim=0)
