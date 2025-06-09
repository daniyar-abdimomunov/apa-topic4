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
x_input = np.array(range(176))
y_input = true = np.genfromtxt(os.path.join(DATA_DIR, 'example_input.csv'), delimiter=',')
x = 176 + np.array(range(24))
true = np.genfromtxt(os.path.join(DATA_DIR, 'example_true.csv') , delimiter=',')
print(f'shape of x_input: {x_input.shape}\n'
      f'shape of y_input: {y_input.shape}\n'
      f'shape of x: {x.shape}\n'
      f'shape of true: {true.shape}\n')

# %%
pred_dist = np.genfromtxt(os.path.join(DATA_DIR, 'example_pred_dist.csv') , delimiter=',')
pred_lower_dist = np.genfromtxt(os.path.join(DATA_DIR, 'example_pred_lower_dist.csv') , delimiter=',')
pred_upper_dist = np.genfromtxt(os.path.join(DATA_DIR, 'example_pred_upper_dist.csv') , delimiter=',')
print(f'shape of true: {true.shape}\n'
      f'shape of pred_dist: {pred_dist.shape}\n'
      f'shape of pred_lower_dist: {pred_lower_dist.shape}\n'
      f'shape of pred_upper_dist: {pred_upper_dist.shape}')

# %%
deterministic_pred = pred_dist.mean(axis=1) + np.random.normal(0,1,len(pred_dist))
print(f'shape of true: {true.shape}\n'
      f'shape of pred_dist: {deterministic_pred.shape}\n')

# %%
benchmarking = {
      'deterministic_model': {
            'name': 'DeterministicModel',
            'pred': deterministic_pred,
            'metrics': dict(),
      },
      'timeseries_gp': {
            'name': 'TimeSeriesGP',
            'pred': pred_dist,
            'lower': pred_lower_dist,
            'upper': pred_upper_dist,
            'metrics': dict(),
      },
}

# %% [markdown]
# ## 2. Calculate Benchmark Metrics

# %%
benchmarking['deterministic_model']['metrics']['rmse'] = calculate_rmse(true, deterministic_pred)
benchmarking['timeseries_gp']['metrics']['rmse'] = calculate_rmse(true, pred_dist)

print(f"{benchmarking['deterministic_model']['name']} RMSE: {benchmarking['deterministic_model']['metrics']['rmse']}\n"
      f"{benchmarking['timeseries_gp']['name']} RMSE: {benchmarking['timeseries_gp']['metrics']['rmse']}")

# %%
benchmarking['deterministic_model']['metrics']['mape'] = calculate_mape(true, deterministic_pred)
benchmarking['timeseries_gp']['metrics']['mape'] = calculate_mape(true, pred_dist)

print(f"{benchmarking['deterministic_model']['name']} MAPE: {benchmarking['deterministic_model']['metrics']['mape']}\n"
      f"{benchmarking['timeseries_gp']['name']} MAPE: {benchmarking['timeseries_gp']['metrics']['mape']}")

# %%
benchmarking['deterministic_model']['metrics']['picp'] = calculate_picp(true, deterministic_pred, deterministic_pred)
benchmarking['timeseries_gp']['metrics']['picp'] = calculate_picp(true, pred_lower_dist, pred_upper_dist)

print(f"{benchmarking['deterministic_model']['name']} PICP: {benchmarking['deterministic_model']['metrics']['picp']}\n"
      f"{benchmarking['timeseries_gp']['name']} PICP: {benchmarking['timeseries_gp']['metrics']['picp']}")

# %%
benchmarking['deterministic_model']['metrics']['crps'] = calculate_crps(true, deterministic_pred)
benchmarking['timeseries_gp']['metrics']['crps'] = calculate_crps(true, pred_dist)
print(f"{benchmarking['deterministic_model']['name']} CRPS: {round(benchmarking['deterministic_model']['metrics']['crps'], 2)}\n"
      f"{benchmarking['timeseries_gp']['name']} CRPS: {round(benchmarking['timeseries_gp']['metrics']['crps'], 2)}")

# %%
benchmarking_scores = dict()
for model_name, results in benchmarking.items():
      benchmarking_scores[model_name] = dict()
      benchmarking_scores[model_name] = results['metrics']
pd.DataFrame.from_dict(benchmarking_scores, orient='index')

# %% [markdown]
# ## 3. Benchmark Scores and Predictions Evaluation

# %%
fig, ax = compare_scores(benchmarking)
plt.show()

# %%
fig, ax = compare_predictions(x_input, y_input, x, true, benchmarking)
plt.show()
