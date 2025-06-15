import pandas as pd
import torch
from torch.utils.data import Dataset
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
def train(model, train_loader, device, epochs=2):
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
# Plotting
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
