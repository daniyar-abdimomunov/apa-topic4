# %% [markdown]
# # Benchmark Model Training

# %%
import numpy as np
# %load_ext autoreload
# %autoreload 2

import os
import pickle
from sklearn.preprocessing import StandardScaler
import sys
import torch
from torch.utils.data import TensorDataset
from transformers import AutoModelForCausalLM

PROJECT_ROOT = os.path.dirname(os.getcwd())
sys.path.insert(0, PROJECT_ROOT)
from __init__ import *

# %% [markdown]
# ## 1. Import Data for Training and Testing

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
LOOKBACK = 192
HORIZON = 96
NUM_TRAIN_SAMPLES = 3000 + LOOKBACK + HORIZON - 1
NUM_TEST_SAMPLES = 1000 + LOOKBACK + HORIZON - 1

# %%
# split data into train and test datasets
train  = eu_df[:NUM_TRAIN_SAMPLES][['Germany (EUR/MWhe)', 'Austria (EUR/MWhe)']].to_numpy()
test = eu_df[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES][['Germany (EUR/MWhe)', 'Austria (EUR/MWhe)']].to_numpy()
print(f'Train dataset shape: {train.shape}\n'
      f'Test dataset shape: {test.shape}\n')

# %%
# Scale the data (i.e. electricity prices)
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

train_input = torch.Tensor(train_input)
train_true = torch.Tensor(train_true[:, : , 0]) # we only want to make predictions for one of the countries, so we select the first one (i.e. Germany) as our true values
test_input = torch.Tensor(test_input)
test_true = torch.Tensor(test_true[:, : , 0]) # we only want to make predictions for one of the countries, so we select the first one (i.e. Germany) as our true values

print('Shape of predictor inputs: ',train_input.shape)
print('Shape of outputs: ',train_true.shape)
print('Shape of test predictor inputs: ',test_input.shape)
print('Shape of test outputs: ',test_true.shape)

# %%
test_input_os = scaler.inverse_transform(test_input.reshape(test_input.shape[0], -1).numpy()).reshape(test_input.shape)
test_true_os = scaler.inverse_transform(test_true.numpy())

# %%
# save test data
np.save('../data/inputs.npy', test_input_os)
np.save('../data/trues.npy', test_true_os)

# %% [markdown]
# ## 2. Model Training

# %%
...

# %% [markdown]
# ### 2.1 TSMixer

# %%
train_loader = DataLoader(MTSMixerDataset(train_input, train_true), batch_size=32, shuffle=True)
test_loader = DataLoader(MTSMixerDataset(test_input, test_true), batch_size=32)

# %%
# TO-DO: train TSMixer model and export predictions
ts_mixer = MTSMixer(
    num_variables=train_input.size(-1),
    time_steps=LOOKBACK,
    output_steps=HORIZON,
    num_blocks=3,
    hidden_dim=64,
    activation='gelu'
)

# %%
ts_mixer.fit(train_loader, device='cpu', epochs=10, lr=1e-3)

# %%
ts_mixer_preds, _ = ts_mixer.predict(test_loader, device='cpu')
ts_mixer_preds_os = scaler.inverse_transform(ts_mixer_preds)

# %%
np.save(os.path.join(DATA_DIR, 'ts_mixer_pred.npy'), ts_mixer_preds_os)

# %%
# visualize predictions and confidence interval
TEST_CASE = 567

plot_predictions(
    test_input_os[TEST_CASE],
    test_true_os[TEST_CASE],
    ts_mixer_preds_os[TEST_CASE],
    title=f'TS Mixer Predictions\nTest Case: #{TEST_CASE}')

# %% [markdown]
# ### 2.2 PatchTST

# %%
train_loader_patch = DataLoader(PatchTSTDataset(train_input, train_true), batch_size=32, shuffle=True)
test_loader_patch = DataLoader(PatchTSTDataset(test_input, test_true), batch_size=32)

# %%
# TO-DO: train PatchTST model and export predictions
patch_TST = PatchTST(
    num_variables=train_input.size(-1),  # number of variables in the time series
    seq_len=LOOKBACK,
    patch_size=16, # must be a divisor of LOOKBACK
    embed_dim=128,  # embedding dimension
    num_layers=3,
    num_heads=4,
    output_steps=HORIZON,
    dropout=0.1
)

patch_TST.fit(train_loader_patch, device='cpu', epochs=15, lr=1e-3)

# %%
patch_TST_preds, trues = patch_TST.predict(test_loader_patch, device='cpu')
patch_TST_preds_os = scaler.inverse_transform(patch_TST_preds)

# %%
np.save(os.path.join(DATA_DIR, 'patch_TST_pred.npy'), patch_TST_preds_os)

# %%
# visualize predictions and confidence interval
TEST_CASE = 789

plot_predictions(
    test_input_os[TEST_CASE],
    test_true_os[TEST_CASE],
    patch_TST_preds_os[TEST_CASE],
    title=f'Patch TST Predictions\nTest Case: #{TEST_CASE}')

# %% [markdown]
# ### 2.3 Distributional Neural Network (DNN)

# %%
# TO-DO: train DNN model and export predictions
dnn = ...
dnn.train(train_input, train_true)

# %%
dnn.evaluate(test_input, test_true)

# %%
dnn_pred = dnn.pred(test_input)
dnn_lowers = ...
dnn_uppers = ...
dnn_pred # NOTE: for probabilistic models we should get predictions in dimensions  (n, m), where n is the number of predicted time-steps, and m is the number of samples

# %%
np.savetxt(os.path.join(DATA_DIR, 'dnn_preds.csv'), dnn_pred, delimiter=",")
np.savetxt(os.path.join(DATA_DIR, 'dnn_lowers.csv'), dnn_lowers, delimiter=",")
np.savetxt(os.path.join(DATA_DIR, 'dnn_uppers.csv'), dnn_uppers, delimiter=",")

# %% [markdown]
# ### 2.4 LLMTime

# %%
# TO-DO: train LLMTime model and export predictions
...

# %% [markdown]
# ### 2.5 Sundial Model
# Sundial is a pre-trained model for Time-Series Forecasting.
# This section is adapted from the [quickstart_zero_shot.ipynb](https://github.com/thuml/Sundial/blob/main/examples/quickstart_zero_shot.ipynb) provided by the developers of the Sundial Model.

# %%
# load model and dataset
model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True)

# %%
test_input_univariate = test_input[:, :, 0]
test_input_univariate_os = scaler.inverse_transform(test_input_univariate)
test_input_univariate.shape

# %%
# forecasting configurations
NUM_SAMPLES = 20           # generate 20 samples
forecast = model.generate(
    test_input_univariate, # take only samples from Germany to predict prices for Germany
    max_new_tokens=HORIZON,
    num_samples=NUM_SAMPLES) # generate 20 probable predictions
print(f'Predictions shape: {forecast.shape}\n'
      f'{forecast.shape[0]} - test cases\n'
      f'{forecast.shape[1]} - prediction samples\n'
      f'{forecast.shape[2]} - predicted time-steps (i.e. prediction horizon)\n')

# %%
# separate predictions and confidence interval
sundial_preds = scaler.inverse_transform(forecast.mean(dim=1))
sundial_lowers = scaler.inverse_transform(forecast.quantile(q=0.025, dim=1))
sundial_uppers = scaler.inverse_transform(forecast.quantile(q=0.975, dim=1))

# %%
# export forecast outputs for benchmarking
np.save('../data/sundial_preds.npy', sundial_preds)
np.save('../data/sundial_lowers.npy', sundial_lowers)
np.save('../data/sundial_uppers.npy', sundial_uppers)

# %%
forecast[TEST_CASE].transpose(1, 0).shape

# %%
# visualize raw predictions
TEST_CASE = 666

plt.figure(figsize=(15, 5))
plt.xlim(0, LOOKBACK + HORIZON)
plt.plot(np.arange(LOOKBACK), test_input_univariate[TEST_CASE], color='black')
plt.plot(np.arange(LOOKBACK, LOOKBACK + HORIZON), forecast[TEST_CASE].transpose(1, 0))
plt.grid()
plt.show()

# %%
# visualize predictions and confidence interval
plot_predictions(
    test_input_univariate_os[TEST_CASE],
    test_true_os[TEST_CASE],
    sundial_preds[TEST_CASE],
    sundial_lowers[TEST_CASE],
    sundial_uppers[TEST_CASE],
    title=f'Sundial Model Predictions\nTest Case: #{TEST_CASE}')

# %% [markdown]
# ## 2.6 GP Extensions

# %%
train_dataset = TensorDataset(train_input, train_true)
train_loader_svgp = DataLoader(train_dataset, batch_size=16, shuffle=True) # batch_size (M), batch shape = ([M, L, V0], [M, H, V1])

# %%
NUM_INDUCING_POINTS = 200 # num_inducing_points (P)
NUM_LATENTS_SVGP = 8 # num_latents_svgp (lf0)
inducing_points = train_input[:NUM_INDUCING_POINTS]

# %% [markdown]
# ### 2.6.1 Base Multistep SVGP

 # %%
 # subset of training inputs
tsgp_model = TSGPModel(inducing_points, HORIZON, num_latents_svgp=NUM_LATENTS_SVGP)
tsgp_model.train_model(train_loader_svgp, num_data=train_input.size(0), epochs=100)

# %%
# Inference with TSGP
tsgp_preds, tsgp_lowers, tsgp_uppers = tsgp_model.infer(test_input, true_shape=test_true.shape)

# %%
# separate predictions and confidence interval
tsgp_preds = scaler.inverse_transform(tsgp_preds.detach().numpy())
tsgp_lowers = scaler.inverse_transform(tsgp_lowers.detach().numpy())
tsgp_uppers = scaler.inverse_transform(tsgp_uppers.detach().numpy())

# %%
# export forecast outputs for benchmarking
np.save(os.path.join(DATA_DIR, 'tsgp_preds.npy'), tsgp_preds)
np.save(os.path.join(DATA_DIR, 'tsgp_lowers.npy'),tsgp_lowers)
np.save(os.path.join(DATA_DIR, 'tsgp_uppers.npy'),tsgp_uppers)

# %%
# visualize predictions and confidence interval
TEST_CASE = 0
plot_predictions(
    test_input_os[TEST_CASE],
    test_true_os[TEST_CASE],
    tsgp_preds[TEST_CASE],
    tsgp_lowers[TEST_CASE],
    tsgp_uppers[TEST_CASE],
    title=f'TSGP Model Predictions\nTest Case: #{TEST_CASE}'
)

# %% [markdown]
# ## 2.6.2 Neural SVGP

# %%
# training
neural_tsgp = NeuralTSGPModel(inducing_points, HORIZON, num_latents_svgp=10, num_latents_lfe=16)
neural_tsgp.train_model(train_loader_svgp, num_data=train_input.size(0), epochs=10)

# %%
# Inference with Neural TSGP
neural_tsgp_preds, neural_tsgp_lowers, neural_tsgp_uppers = neural_tsgp.infer(test_input, true_shape=test_true.shape)

# %%
# separate predictions and confidence interval
neural_tsgp_preds = scaler.inverse_transform(neural_tsgp_preds.detach().numpy())
neural_tsgp_lowers = scaler.inverse_transform(neural_tsgp_lowers.detach().numpy())
neural_tsgp_uppers = scaler.inverse_transform(neural_tsgp_uppers.detach().numpy())

# %%
# export
np.save(os.path.join(DATA_DIR, 'neural_tsgp_preds.npy'), neural_tsgp_preds)
np.save(os.path.join(DATA_DIR, 'neural_tsgp_lowers.npy'), neural_tsgp_lowers)
np.save(os.path.join(DATA_DIR, 'neural_tsgp_uppers.npy'), neural_tsgp_uppers)

# %%
# visualization
TEST_CASE = 666
plot_predictions(
    test_input_os[TEST_CASE],
    test_true_os[TEST_CASE],
    neural_tsgp_preds[TEST_CASE],
    neural_tsgp_lowers[TEST_CASE],
    neural_tsgp_uppers[TEST_CASE],
    title=f'Neural TSGP Model Predictions\nTest Case: #{TEST_CASE}'
)

# %%
