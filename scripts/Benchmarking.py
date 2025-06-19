# %% [markdown]
# # Benchmarking

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import pandas as pd
import sys

PROJECT_ROOT = os.path.dirname(os.getcwd())
sys.path.insert(0, PROJECT_ROOT)
from __init__ import *

# %% [markdown]
# ## 1. Import Input and Prediction Data

# %%
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# %%
inputs = np.genfromtxt(os.path.join(DATA_DIR, 'inputs.csv'), delimiter=',')
trues = np.genfromtxt(os.path.join(DATA_DIR, 'trues.csv') , delimiter=',')
print(f'shape of inputs: {inputs.shape}\n'
      f'shape of trues: {trues.shape}\n')

# %%
ts_mixer_preds = np.genfromtxt(os.path.join(DATA_DIR, 'ts_mixer_pred.csv') , delimiter=',')
print(f'shape of trues: {trues.shape}\n'
      f'shape of preds: {ts_mixer_preds.shape}\n')

# %%
patch_tst_preds = np.genfromtxt(os.path.join(DATA_DIR, 'patch_TST_pred.csv') , delimiter=',')
print(f'shape of trues: {trues.shape}\n'
      f'shape of preds: {ts_mixer_preds.shape}\n')

# %%
sundial_preds = np.genfromtxt(os.path.join(DATA_DIR, 'sundial_preds.csv') , delimiter=',')
sundial_lowers = np.genfromtxt(os.path.join(DATA_DIR, 'sundial_lowers.csv') , delimiter=',')
sundial_uppers = np.genfromtxt(os.path.join(DATA_DIR, 'sundial_uppers.csv') , delimiter=',')
print(f'shape of trues: {trues.shape}\n'
      f'shape of preds: {sundial_preds.shape}\n'
      f'shape of lowers: {sundial_lowers.shape}\n'
      f'shape of uppers: {sundial_uppers.shape}')

# %%
gp_preds = np.genfromtxt(os.path.join(DATA_DIR, 'mt_batch_ts_gp_preds.csv') , delimiter=',')
gp_lowers = np.genfromtxt(os.path.join(DATA_DIR, 'mt_batch_ts_gp_lowers.csv') , delimiter=',')
gp_uppers = np.genfromtxt(os.path.join(DATA_DIR, 'mt_batch_ts_gp_uppers.csv') , delimiter=',')
print(f'shape of true: {trues.shape}\n'
      f'shape of preds: {gp_preds.shape}\n'
      f'shape of lowers: {gp_lowers.shape}\n'
      f'shape of uppers: {gp_uppers.shape}')

# %%
benchmarking = {
      'ts_mixer': {
            'name': 'TS Mixer',
            'pred': ts_mixer_preds,
            'lower': None,
            'upper': None,
            'metrics': dict()
      },
      'patch_tst': {
            'name': 'Patch TST',
            'pred': patch_tst_preds,
            'lower': None,
            'upper': None,
            'metrics': dict()
      },
      'sundial': {
            'name': 'Sundial',
            'pred': sundial_preds,
            'lower': sundial_lowers,
            'upper': sundial_uppers,
            'metrics': dict(),
      },
      'timeseries_gp': {
            'name': 'Time-series GP',
            'pred': gp_preds,
            'lower': gp_lowers,
            'upper': gp_uppers,
            'metrics': dict(),
      },
}

# %% [markdown]
# ## 2. Calculate Benchmark Metrics

# %%
for model in benchmarking.keys():
      benchmarking[model]['metrics']['rmse'] = calculate_rmse(trues, benchmarking[model]['pred'])
      print(f"{benchmarking[model]['name']} RMSE: {benchmarking[model]['metrics']['rmse']}")

# %%
for model in benchmarking.keys():
      benchmarking[model]['metrics']['mape'] = calculate_mape(trues, benchmarking[model]['pred'])
      print(f"{benchmarking[model]['name']} MAPE: {benchmarking[model]['metrics']['mape']}")

# %%
for model in benchmarking.keys():
      benchmarking[model]['metrics']['picp'] = calculate_picp(trues, benchmarking[model]['lower'], benchmarking[model]['upper'])
      print(f"{benchmarking[model]['name']} PICP: {benchmarking[model]['metrics']['picp']}")

# %%
for model in benchmarking.keys():
      benchmarking[model]['metrics']['crps'] = calculate_crps(trues, benchmarking[model]['pred'])
      print(f"{benchmarking[model]['name']} CRPS: {benchmarking[model]['metrics']['crps']}")

# %%
benchmarking_scores = dict()
for model_name, results in benchmarking.items():
      benchmarking_scores[model_name] = dict()
      benchmarking_scores[model_name] = results['metrics']
pd.DataFrame.from_dict(benchmarking_scores, orient='index')

# %% [markdown]
# ## 3. Benchmark Scores and Predictions Evaluation

# %%
figures = compare_scores(benchmarking)
plt.show()

# %%
fig, ax = compare_single_prediction(inputs, trues, benchmarking, test_case = 488)
plt.show()

# %%
test_cases = [0, 178, 367, 711]

fig, axs = compare_multi_predictions(inputs, trues, benchmarking, test_cases = test_cases)
plt.show()
