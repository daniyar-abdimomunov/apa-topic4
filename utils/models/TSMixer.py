# MTSMixer: A PyTorch implementation of the MTSMixer model for multivariate time series forecasting.

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# -----------------------------
# Dataset
# -----------------------------
class MTSMixerDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# -----------------------------
# MTSMixer Model
# -----------------------------
class MTSMixerBlock(nn.Module):
    def __init__(self, num_variables, time_steps, hidden_dim=64, activation='gelu'):
        super().__init__()
        act_fn = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'silu': nn.SiLU()}[activation]
        self.channel_norm = nn.LayerNorm(num_variables)
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_variables, hidden_dim), act_fn, nn.Linear(hidden_dim, num_variables))
        self.time_norm = nn.LayerNorm(time_steps)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_steps, hidden_dim), act_fn, nn.Linear(hidden_dim, time_steps))

    def forward(self, x):
        x = x + self.channel_mlp(self.channel_norm(x))
        x_t = self.time_norm(x.permute(0, 2, 1))
        x = x + self.time_mlp(x_t).permute(0, 2, 1)
        return x

class MTSMixer(nn.Module):
    def __init__(self, num_variables, time_steps, output_steps, num_blocks=3, hidden_dim=64, activation='gelu'):
        super().__init__()
        self.blocks = nn.ModuleList([
            MTSMixerBlock(num_variables, time_steps, hidden_dim, activation) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(num_variables * time_steps, output_steps))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    # -----------------------------
    # Training and Evaluation
    # -----------------------------
    def fit(self, train_loader, device, epochs=2, lr=1e-3):
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
