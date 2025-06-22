# %% [markdown]
# # Benchmark Model Training

# %%
# %load_ext autoreload
# %autoreload 2

import os
import pickle
from sklearn.preprocessing import StandardScaler
import sys
import torch

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
LOOKBACK=192
HORIZON=96
NUM_TRAIN_SAMPLES=3000 # Must be ≥ (LOOKBACK + HORIZON)
NUM_TEST_SAMPLES=1000 # Must be ≥ (LOOKBACK + HORIZON)

# %%
# split data into train and test datasets
train  = eu_df[:NUM_TRAIN_SAMPLES]['Germany (EUR/MWhe)'].to_numpy()
test = eu_df[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES]['Germany (EUR/MWhe)'].to_numpy()
print(f'Train dataset shape: {train.shape}\n'
      f'Test dataset shape: {test.shape}\n')

# %%
# Scale the data (i.e. electricity prices)
scaler = StandardScaler()
scaler.fit(train.reshape(-1, 1))
train_scaled = scaler.transform(train.reshape(-1, 1)).reshape(-1)
test_scaled = scaler.transform(test.reshape(-1, 1)).reshape(-1)

print(f'Original Train dataset  mean: {round(train.mean(), 2)}; \tstd: {round(train.std(), 2)}\n'
      f'Scaled Train dataset    mean: {round(train_scaled.mean(), 2)}; \t\tstd: {round(train_scaled.std(), 2)}\n'
      f'Scaled Test dataset     mean: {round(test_scaled.mean(), 2)}; \tstd: {round(test_scaled.std(), 2)}\n')


# %%
def sequentialize(data: np.ndarray, lookback:int, horizon:int) -> (np.ndarray, np.ndarray):
    input = list()
    true = list()
    for index, value in enumerate(data[:-(lookback+horizon)]):
        input.append(data[index:index+lookback])
        true.append(data[index+lookback:index+lookback+horizon])
    return np.array(input), np.array(true)


# %%
# re-shape model into set of sequences,
train_input, train_true = sequentialize(train_scaled, LOOKBACK, HORIZON)
test_input, test_true = sequentialize(test_scaled, LOOKBACK, HORIZON)

train_input = torch.Tensor(train_input)
train_true = torch.Tensor(train_true)
test_input = torch.Tensor(test_input)
test_true = torch.Tensor(test_true)

print('Shape of predictor inputs: ',train_input.shape)
print('Shape of outputs: ',train_true.shape)
print('Shape of test predictor inputs: ',test_input.shape)
print('Shape of test outputs: ',test_true.shape)

# %%
test_input_os = scaler.inverse_transform(test_input.numpy())
test_true_os = scaler.inverse_transform(test_true.numpy())

# %%
# save test data
np.savetxt('../data/inputs.csv', test_input_os, delimiter=",")
np.savetxt('../data/trues.csv', test_true_os,delimiter=",")

# %% [markdown]
# ## 2. Model Training

# %%
...

# %% [markdown]
# ### 2.1 TSMixer

# %%
# TO-DO: train TSMixer model and export predictions
ts_mixer = ...
ts_mixer.train(train_input, train_true)

# %%
ts_mixer.evaluate(test_input, test_true)

# %%
ts_mixer_preds = ts_mixer.pred(test_input)
ts_mixer_preds # NOTE: make sure predictions are type np.ndarray and have shape (n, ), where n is the number of predicted time-steps

# %%
np.savetxt(os.path.join(DATA_DIR, 'ts_mixer_pred.csv'), ts_mixer_preds, delimiter=",")

# %% [markdown]
# ### 2.2 PatchTST

# %%
# TO-DO: train PatchTST model and export predictions
...

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
