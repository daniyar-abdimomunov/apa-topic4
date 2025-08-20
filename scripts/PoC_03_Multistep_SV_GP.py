# %%
# %load_ext autoreload
# %autoreload 2


import os
import pickle
import sys
from torch.utils.data import TensorDataset

PROJECT_ROOT = os.path.dirname(os.getcwd())
sys.path.insert(0, PROJECT_ROOT)
from __init__ import *

# %% [markdown]
# ## 1. Import Data for Traning and Testing

# %%
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# %%
# import the data:
with open(os.path.join(DATA_DIR, 'EU_Electricity_TimeSeries.pkl'),'rb') as f:
    eu_df=pickle.load(f)

print(f'Dataset shape: {eu_df.shape}\n')
eu_df

# %%
# define train-test split and prediction parameters
LOOKBACK = 192 # lookback (L)
HORIZON = 96 # horizon (H)
NUM_TRAIN_SAMPLES = 3000 + LOOKBACK + HORIZON - 1 # num_samples (N) + lookback (L) + horizon (H)
NUM_TEST_SAMPLES = 1000 + LOOKBACK + HORIZON - 1

# %%
# split data into train and test datasets
train  = eu_df[:NUM_TRAIN_SAMPLES][['Germany (EUR/MWhe)', 'Austria (EUR/MWhe)']].to_numpy()
test = eu_df[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES][['Germany (EUR/MWhe)', 'Austria (EUR/MWhe)']].to_numpy()
print(f'Train dataset shape: {train.shape}\n'
      f'Test dataset shape: {test.shape}\n')

# %%
# Scale the data (i.e. electricity prices)
num_input_vars = train.shape[-1] # num_input_vars (V0)
num_output_vars = 1 # num_output_vars (V1)
scaler = StandardScaler()
scaler.fit(train.reshape(-1, 1))
train_scaled = scaler.transform(train.reshape(-1, 1)).reshape(train.shape)
test_scaled = scaler.transform(test.reshape(-1, 1)).reshape(test.shape)

print(f'Original Train dataset  mean: {round(train.mean(), 2)}; \tstd: {round(train.std(), 2)}\n'
      f'Scaled Train dataset    mean: {round(train_scaled.mean(), 2)}; \t\tstd: {round(train_scaled.std(), 2)}\n'
      f'Scaled Test dataset     mean: {round(test_scaled.mean(), 2)}; \tstd: {round(test_scaled.std(), 2)}\n')

# %%
# re-shape model into set of sequences,
train_input, train_true = sequentialize(train_scaled, LOOKBACK, HORIZON)
test_input, test_true = sequentialize(test_scaled, LOOKBACK, HORIZON)

train_input = torch.Tensor(train_input) # shape [N, L, V0]
train_true = torch.Tensor(train_true[:, : , :1]) # shape [N, H, V1]; [:, : , :1] - only select the prices of the first country
test_input = torch.Tensor(test_input)  # shape [N1, L, V0]
test_true = torch.Tensor(test_true[:, : , :1]) # # shape [N1, H, V1]; [:, : , :1] - only select the prices of the first country

print('Shape of predictor inputs: ',train_input.shape)
print('Shape of outputs: ',train_true.shape)
print('Shape of test predictor inputs: ',test_input.shape)
print('Shape of test outputs: ',test_true.shape)

# %%
train_dataset = TensorDataset(train_input, train_true)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # batch_size (M), batch shape = ([M, L, V0], [M, H, V1])

# %%
test_input_os = scaler.inverse_transform(test_input.reshape(test_input.shape[0], -1).numpy()).reshape(test_input.shape)
test_true_os = scaler.inverse_transform(test_true.reshape(test_true.shape[0], -1).numpy()).reshape(test_true.shape)

# %% [markdown]
# ## 2. Training Multistep Variational GP

# %%
NUM_INDUCING_POINTS = 200 # num_inducing_points (P)
NUM_LATENTS_SVGP = 8 # num_latents_svgp (lf0)

# %%
inducing_points = train_input[:NUM_INDUCING_POINTS]
inducing_points.shape

# %%
model = TSGPModel(inducing_points, horizon=HORIZON, num_latents_svgp=NUM_LATENTS_SVGP)

# %%
model.train_model(train_loader, num_data=train_input.size(0), epochs=100)

# %%
preds, lowers, uppers = model.infer(test_input, test_true.shape)

# %%
# Rescale target to original scale (os)
preds_os = scaler.inverse_transform(preds.mean(dim=-1))
lowers_os = scaler.inverse_transform(lowers.quantile(q=0.05, dim=-1))
uppers_os = scaler.inverse_transform(uppers.quantile(q=0.95, dim=-1))

# %%
test_input.reshape(-1, LOOKBACK).shape

# %%
# plot predictions
TEST_CASE = 789

plot_predictions(
    input=test_input_os[TEST_CASE],
    true=test_true_os[TEST_CASE],
    pred=preds_os[TEST_CASE],
    lower=lowers_os[TEST_CASE],
    upper=uppers_os[TEST_CASE],
    title=f'Time-series GP Model Predictions\nTest Case: #{TEST_CASE}'
)

# %% [markdown]
# ## 3. Neural Extension of MultistepSVGP

# %%
NUM_LATENTS_LFE = 6

# %%
model = NeuralTSGPModel(inducing_points, horizon=HORIZON, num_latents_svgp=NUM_LATENTS_SVGP, num_latents_lfe=NUM_LATENTS_LFE)

# %%
model.train_model(train_loader, num_data=train_input.size(0))

# %%
preds, lowers, uppers = model.infer(test_input, test_true.shape)

# %%
# Rescale target to original scale (os)
preds_os = scaler.inverse_transform(preds.mean(dim=-1))
lowers_os = scaler.inverse_transform(lowers.quantile(q=0.05, dim=-1))
uppers_os = scaler.inverse_transform(uppers.quantile(q=0.95, dim=-1))

# %%
# plot predictions
TEST_CASE = 789

plot_predictions(
    input=test_input_os[TEST_CASE],
    true=test_true_os[TEST_CASE],
    pred=preds_os[TEST_CASE],
    lower=lowers_os[TEST_CASE],
    upper=uppers_os[TEST_CASE],
    title=f'Time-series GP Model Predictions\nTest Case: #{TEST_CASE}'
)
