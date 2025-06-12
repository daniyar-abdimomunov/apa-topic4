# %% [markdown]
# # Benchmark Model Training

# %%
# %load_ext autoreload
# %autoreload 2

import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.getcwd())
sys.path.insert(0, PROJECT_ROOT)
from __init__ import *

# %% [markdown]
# ## 1. Import Data for Training and Testing

# %%
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# %%
#Import the data:
with open(os.path.join(DATA_DIR, 'EU_Electricity_TimeSeries.pkl'),'rb') as f:
    eu_df=pickle.load(f)

print(f'Dataset shape: {eu_df.shape}\n')
eu_df

# %%
# Define variables for splitting train and test datasets
# TO-DO: replace placeholder values for SEQ_LEN, NUM_SAMPLES, TEST_SAMPLES

SEQ_LEN=96
NUM_SAMPLES=600
TEST_SAMPLES=200

# %%
# Split Train and Test datasets
train_ts, test_ts = eu_df[:NUM_SAMPLES][['Germany (EUR/MWhe)']], eu_df[NUM_SAMPLES:NUM_SAMPLES + TEST_SAMPLES][['Germany (EUR/MWhe)']]

# %%
# Scale data
# TO-DO: scale train and test datasets
...

# %%
# Do time-series embedding for
# TO-DO: implement time-series embedding
...
x_train = np.array()
y_train = np.array()
x_test = np.array()
y_test = np.array()

# %% [markdown]
# ## 2. Model Training

# %%
...

# %% [markdown]
# ### 2.1 TSMixer

# %%
# TO-DO: train TSMixer model and export predictions
ts_mixer = ...
ts_mixer.train(x_train, y_train)

# %%
ts_mixer.evaluate(x_test, y_test)

# %%
y_pred_ts_mixer = ts_mixer.pred(x_test)
y_pred_ts_mixer # NOTE: make sure predictions are type np.ndarray and have shape (n, ), where n is the number of predicted time-steps

# %%
np.savetxt(os.path.join(DATA_DIR, 'ts_mixer_pred.csv'), y_pred_ts_mixer, delimiter=",")

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
dnn.train(x_train, y_train)

# %%
dnn.evaluate(x_test, y_test)

# %%
y_pred_dnn = dnn.pred(x_test)
y_lower_dnn = ...
y_upper_dnn = ...
y_pred_dnn # NOTE: for probabilistic models we should get predictions in dimensions  (n, m), where n is the number of predicted time-steps, and m is the number of samples

# %%
np.savetxt(os.path.join(DATA_DIR, 'dnn_pred.csv'), y_pred_dnn, delimiter=",")
np.savetxt(os.path.join(DATA_DIR, 'dnn_lower.csv'), y_lower_dnn, delimiter=",")
np.savetxt(os.path.join(DATA_DIR, 'dnn_upper.csv'), y_upper_dnn, delimiter=",")

# %% [markdown]
# ### 2.4 LLMTime

# %%
# TO-DO: train LLMTime model and export predictions
...
