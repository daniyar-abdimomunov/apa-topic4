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
LOOKBACK = 192 #number of past time steps used as input
HORIZON = 96   #number of future time steps to predict 
NUM_TRAIN_SAMPLES = 3000 + LOOKBACK + HORIZON - 1 #train samples
NUM_TEST_SAMPLES = 1000 + LOOKBACK + HORIZON - 1  #test sapmles
# %%
# split the dataset into train and test sets then show the shapes of the resulting arrays
train  = eu_df[:NUM_TRAIN_SAMPLES]['Germany (EUR/MWhe)'].to_numpy()
test = eu_df[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES]['Germany (EUR/MWhe)'].to_numpy()
print(f'Train dataset shape: {train.shape}\n'
      f'Test dataset shape: {test.shape}\n')

# %%
# Scale the data (i.e. electricity prices)
# Standardize the training and test datasets
scaler = StandardScaler()
# Fit the scaler on the training data 
scaler.fit(train.reshape(-1, 1))
# Apply the transformation and reshape back to 1D
train_scaled = scaler.transform(train.reshape(-1, 1)).reshape(-1)
test_scaled = scaler.transform(test.reshape(-1, 1)).reshape(-1)

# Print statistics before and after scaling to verify standadization 
print(f'Original Train dataset  mean: {round(train.mean(), 2)}; \tstd: {round(train.std(), 2)}\n'
      f'Scaled Train dataset    mean: {round(train_scaled.mean(), 2)}; \t\tstd: {round(train_scaled.std(), 2)}\n'
      f'Scaled Test dataset     mean: {round(test_scaled.mean(), 2)}; \tstd: {round(test_scaled.std(), 2)}\n')

# %%
# re-shape model into set of sequences,
# Prepare sequential datasets for training and testing 
train_input, train_true = sequentialize(train_scaled, LOOKBACK, HORIZON)
test_input, test_true = sequentialize(test_scaled, LOOKBACK, HORIZON)
# Convert arrays to Pytorch tensors
train_input = torch.Tensor(train_input)
train_true = torch.Tensor(train_true)
test_input = torch.Tensor(test_input)
test_true = torch.Tensor(test_true)
# Display the shapes of the prepared datasets
print('Shape of predictor inputs: ',train_input.shape)
print('Shape of outputs: ',train_true.shape)
print('Shape of test predictor inputs: ',test_input.shape)
print('Shape of test outputs: ',test_true.shape)

# %%
# Convert scaled test data back to the original feature space ( Undo standardization)
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

#Prepare DataLoaders for training and testing.
train_loader = DataLoader(MTSMixerDataset(train_input.unsqueeze(-1), train_true), batch_size=32, shuffle=True)
test_loader = DataLoader(MTSMixerDataset(test_input.unsqueeze(-1), test_true), batch_size=32)

# %%
# Initialize the MTSMixer model 
ts_mixer = MTSMixer(
    num_variables= 1,    # Number of input variables
    time_steps=LOOKBACK, # Length of the input sequence 
    output_steps=HORIZON,# Number of steps to predict into the future
    num_blocks=3,        # Number of mixer blocks in the architecture 
    hidden_dim=64,       # Dimension of hidden layers inside the model 
    activation='gelu'    # non-linear activation function used in the network
)

# %%
#Train the MTSMixer model 
ts_mixer.fit(train_loader, device='cpu', epochs=2, lr=1e-3)

# %%
# Generate prediction with the trained MTSMixer model 
ts_mixer_preds, trues = ts_mixer.predict(test_loader, device='cpu')
ts_mixer_preds_os = scaler.inverse_transform(ts_mixer_preds)

# %%
np.savetxt(os.path.join(DATA_DIR, 'ts_mixer_pred.csv'), ts_mixer_preds_os, delimiter=",")

# %%
# visualize predictions and confidence interval
TEST_CASE = 0

plot_predictions(
    scaler.inverse_transform(test_input.numpy())[TEST_CASE],
    scaler.inverse_transform(test_true.numpy())[TEST_CASE],
    ts_mixer_preds_os[TEST_CASE],
    title=f'TS Mixer Predictions\nTest Case: #{TEST_CASE}')

# %% [markdown]
# ### 2.2 PatchTST

# %%
# Prepare DataLoaders for the PatchTST model 

train_loader_patch = DataLoader(PatchTSTDataset(train_input.unsqueeze(-1), train_true),
                                batch_size=32, shuffle=True)
test_loader_patch = DataLoader(PatchTSTDataset(test_input.unsqueeze(-1), test_true),
                               batch_size=32)

# %%
# Initialize the PatchTST model
patch_TST = PatchTST(
    num_variables=1,     # Number of input features in the time series
    seq_len=LOOKBACK,    # Length of the input sequence 
    patch_size=16,       # Size of each patch (Must be a divisor of LOOKBACK )
    embed_dim=128,       # Dimension of the embedding space
    num_layers=3,        # Number of transformer encoder layers 
    num_heads=4,         # Number of attention heads per layer
    output_steps=HORIZON,# Number of future steps to predict  
    dropout=0.1          # Dropout rate for regularization
)
# Train the patchTST model 
patch_TST.fit(train_loader_patch, device='cpu', epochs=2, lr=1e-3)

# %%
# Generate predictions with the trained PatchTST model

patch_TST_preds, trues = patch_TST.predict(test_loader_patch, device='cpu')
patch_TST_preds_os = scaler.inverse_transform(patch_TST_preds)

# %%
np.savetxt(os.path.join(DATA_DIR, 'patch_TST_pred.csv'), patch_TST_preds_os, delimiter=",")

# %%
# visualize predictions and confidence interval
TEST_CASE = 0

plot_predictions(
    scaler.inverse_transform(test_input.numpy())[TEST_CASE],
    scaler.inverse_transform(test_true.numpy())[TEST_CASE],
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
df = pd.read_csv("https://raw.githubusercontent.com/WenWeiTHU/TimeSeriesDatasets/refs/heads/main/ETT-small/ETTh2.csv")

# %%
# forecasting configurations
NUM_SAMPLES = 20           # generate 20 samples
forecast = model.generate(test_input, max_new_tokens=HORIZON, num_samples=NUM_SAMPLES) # generate 20 probable predictions
print(f'Predictions shape: {forecast.shape}\n'
      f'{forecast.shape[0]} - test cases\n'
      f'{forecast.shape[1]} - prediction samples\n'
      f'{forecast.shape[2]} - predicted time-steps (i.e. prediction horizon)\n')

# %%
# separate predictions and confidence interval
sundial_preds = scaler.inverse_transform(forecast.mean(dim=1))
sundial_lowers = scaler.inverse_transform(forecast.quantile(q=0.05, dim=1))
sundial_uppers = scaler.inverse_transform(forecast.quantile(q=0.95, dim=1))

# %%
# export forecast outputs for benchmarking
np.savetxt('../data/sundial_preds.csv',sundial_preds,delimiter=",")
np.savetxt('../data/sundial_lowers.csv',sundial_lowers,delimiter=",")
np.savetxt('../data/sundial_uppers.csv',sundial_uppers,delimiter=",")

# %%
# visualize raw predictions
TEST_CASE = 666

plt.figure(figsize=(15, 5))
plt.xlim(0, LOOKBACK + HORIZON)
plt.plot(np.arange(LOOKBACK), test_input[TEST_CASE], color='black')
plt.plot(np.arange(LOOKBACK, LOOKBACK + HORIZON), forecast[TEST_CASE].transpose(1, 0))
plt.grid()
plt.show()

# %%
# visualize predictions and confidence interval
plot_predictions(
    scaler.inverse_transform(test_input.numpy())[TEST_CASE],
    scaler.inverse_transform(test_true.numpy())[TEST_CASE],
    sundial_preds[TEST_CASE],
    sundial_lowers[TEST_CASE],
    sundial_uppers[TEST_CASE],
    title=f'Sundial Model Predictions\nTest Case: #{TEST_CASE}')
