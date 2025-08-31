# %% [markdown]
# # Benchmarking
#
# Evaluate multiple forecasting models on a shared dataset by computing RMSE
# MAPE, PICP and CRPS, then visualize results for aggregate and case level
# insights.
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
inputs = np.load(os.path.join(DATA_DIR, 'inputs.npy'))
inputs_univ = np.load(os.path.join(DATA_DIR, 'inputs_univariate.npy'))
trues = np.load(os.path.join(DATA_DIR, 'trues.npy'))
print(f'shape of inputs: {inputs.shape}\n'
      f'shape of inputs_univariate: {inputs_univ.shape}\n'
      f'shape of trues: {trues.shape}\n')

# %%
ts_mixer_preds = np.load(os.path.join(DATA_DIR, 'ts_mixer_pred.npy'))
print(f'shape of trues: {trues.shape}\n'
      f'shape of preds: {ts_mixer_preds.shape}\n')

# %%
patch_tst_preds = np.load(os.path.join(DATA_DIR, 'patch_TST_pred.npy'))
print(f'shape of trues: {trues.shape}\n'
      f'shape of preds: {ts_mixer_preds.shape}\n')

# %%
sundial_preds = np.load(os.path.join(DATA_DIR, 'sundial_preds.npy'))
sundial_lowers = np.load(os.path.join(DATA_DIR, 'sundial_lowers.npy'))
sundial_uppers = np.load(os.path.join(DATA_DIR, 'sundial_uppers.npy'))
print(f'shape of trues: {trues.shape}\n'
      f'shape of preds: {sundial_preds.shape}\n'
      f'shape of lowers: {sundial_lowers.shape}\n'
      f'shape of uppers: {sundial_uppers.shape}')

# %%
tsgp_univ_preds = np.load(os.path.join(DATA_DIR, 'tsgp_uni_preds.npy'))
tsgp_univ_lowers = np.load(os.path.join(DATA_DIR, 'tsgp_uni_lowers.npy'))
tsgp_univ_uppers = np.load(os.path.join(DATA_DIR, 'tsgp_uni_uppers.npy'))
print(f'shape of true: {trues.shape}\n'
      f'shape of preds: {tsgp_univ_preds.shape}\n'
      f'shape of lowers: {tsgp_univ_lowers.shape}\n'
      f'shape of uppers: {tsgp_univ_uppers.shape}')
# %%
neural_tsgp_univ_preds = np.load(os.path.join(DATA_DIR, 'neural_tsgp_uni_preds.npy'))
neural_tsgp_univ_lowers = np.load(os.path.join(DATA_DIR, 'neural_tsgp_uni_lowers.npy'))
neural_tsgp_univ_uppers = np.load(os.path.join(DATA_DIR, 'neural_tsgp_uni_uppers.npy'))
print(f'shape of true: {trues.shape}\n'
      f'shape of preds: {neural_tsgp_univ_preds.shape}\n'
      f'shape of lowers: {neural_tsgp_univ_lowers.shape}\n'
      f'shape of uppers: {neural_tsgp_univ_uppers.shape}')

# %%
patch_tst_gp_univ_preds = np.load(os.path.join(DATA_DIR, 'patch_tst_gp_uni_preds.npy'))
patch_tst_gp_univ_lowers = np.load(os.path.join(DATA_DIR, 'patch_tst_gp_uni_lowers.npy'))
patch_tst_gp_univ_uppers = np.load(os.path.join(DATA_DIR, 'patch_tst_gp_uni_uppers.npy'))
print(f'shape of true: {trues.shape}\n'
      f'shape of preds: {patch_tst_gp_univ_preds.shape}\n'
      f'shape of lowers: {patch_tst_gp_univ_lowers.shape}\n'
      f'shape of uppers: {patch_tst_gp_univ_uppers.shape}')

# %%
ts_mixer_gp_univ_preds = np.load(os.path.join(DATA_DIR, 'ts_mixer_gp_uni_preds.npy'))
ts_mixer_gp_univ_lowers = np.load(os.path.join(DATA_DIR, 'ts_mixer_gp_uni_lowers.npy'))
ts_mixer_gp_univ_uppers = np.load(os.path.join(DATA_DIR, 'ts_mixer_gp_uni_uppers.npy'))
print(f'shape of true: {trues.shape}\n'
      f'shape of preds: {ts_mixer_gp_univ_preds.shape}\n'
      f'shape of lowers: {ts_mixer_gp_univ_lowers.shape}\n'
      f'shape of uppers: {ts_mixer_gp_univ_uppers.shape}')


# %%
tsgp_preds = np.load(os.path.join(DATA_DIR, 'tsgp_preds.npy'))
tsgp_lowers = np.load(os.path.join(DATA_DIR, 'tsgp_lowers.npy'))
tsgp_uppers = np.load(os.path.join(DATA_DIR, 'tsgp_uppers.npy'))
print(f'shape of true: {trues.shape}\n'
      f'shape of preds: {tsgp_preds.shape}\n'
      f'shape of lowers: {tsgp_lowers.shape}\n'
      f'shape of uppers: {tsgp_uppers.shape}')

# %%
neural_tsgp_preds = np.load(os.path.join(DATA_DIR, 'neural_tsgp_preds.npy'))
neural_tsgp_lowers = np.load(os.path.join(DATA_DIR, 'neural_tsgp_lowers.npy'))
neural_tsgp_uppers = np.load(os.path.join(DATA_DIR, 'neural_tsgp_uppers.npy'))
print(f'shape of true: {trues.shape}\n'
      f'shape of preds: {neural_tsgp_preds.shape}\n'
      f'shape of lowers: {neural_tsgp_lowers.shape}\n'
      f'shape of uppers: {neural_tsgp_uppers.shape}')

# %%
patch_tst_gp_preds = np.load(os.path.join(DATA_DIR, 'patch_tst_gp_preds.npy'))
patch_tst_gp_lowers = np.load(os.path.join(DATA_DIR, 'patch_tst_gp_lowers.npy'))
patch_tst_gp_uppers = np.load(os.path.join(DATA_DIR, 'patch_tst_gp_uppers.npy'))
print(f'shape of true: {trues.shape}\n'
      f'shape of preds: {patch_tst_gp_preds.shape}\n'
      f'shape of lowers: {patch_tst_gp_lowers.shape}\n'
      f'shape of uppers: {patch_tst_gp_uppers.shape}')

# %%
ts_mixer_gp_preds = np.load(os.path.join(DATA_DIR, 'ts_mixer_gp_preds.npy'))
ts_mixer_gp_lowers = np.load(os.path.join(DATA_DIR, 'ts_mixer_gp_lowers.npy'))
ts_mixer_gp_uppers = np.load(os.path.join(DATA_DIR, 'ts_mixer_gp_uppers.npy'))
print(f'shape of true: {trues.shape}\n'
      f'shape of preds: {ts_mixer_gp_preds.shape}\n'
      f'shape of lowers: {ts_mixer_gp_lowers.shape}\n'
      f'shape of uppers: {ts_mixer_gp_uppers.shape}')

# %%
# benchmarking structure:
# {
#   'model_id': {
#       'name': str,                   # display label
#       'pred': Sequence[float],       # point predictions
#       'lower': Sequence[float] | None, # lower prediction interval (None if unavailable)
#       'upper': Sequence[float] | None, # upper prediction interval (None if unavailable)
#       'metrics': dict[str, float],   # evaluation metrics (e.g. RMSE, MAPE)
#   },
#   ...
# }
benchmarking = {
      'ts_mixer': {
            'name': 'TS Mixer',
            'pred': ts_mixer_preds,
            'metrics': dict()
      },
      'patch_tst': {
            'name': 'Patch TST',
            'pred': patch_tst_preds,
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
            'pred': tsgp_preds,
            'lower': tsgp_lowers,
            'upper': tsgp_uppers,
            'metrics': dict(),
      },
      'tsgp_univariate': {
            'name': 'Time-series GP Univariate',
            'pred': ts_mixer_gp_univ_preds,
            'lower': ts_mixer_gp_univ_lowers,
            'upper': ts_mixer_gp_univ_uppers,
            'metrics': dict(),
      },
      'neural_timeseries_gp': {
            'name': 'Neural Time-series GP',
            'pred': neural_tsgp_preds,
            'lower': neural_tsgp_lowers,
            'upper': neural_tsgp_uppers,
            'metrics': dict(),
      },
      'neural_tsgp_univariate': {
            'name': 'Neural Time-series GP Univariate',
            'pred': neural_tsgp_univ_preds,
            'lower': neural_tsgp_univ_lowers,
            'upper': neural_tsgp_univ_uppers,
            'metrics': dict(),
      },
      'patcht_tst_timeseries_gp': {
            'name': 'PatchTST Time-series GP',
            'pred': patch_tst_gp_preds,
            'lower': patch_tst_gp_lowers,
            'upper': patch_tst_gp_uppers,
            'metrics': dict(),
      },
      'patch_tst_gp_univariate': {
            'name': 'PatchTST Time-series GP Univariate',
            'pred': patch_tst_gp_univ_preds,
            'lower': patch_tst_gp_univ_lowers,
            'upper': patch_tst_gp_univ_uppers,
            'metrics': dict(),
      },
      'ts_mixer_timeseries_gp': {
            'name': 'TS Mixer Time-series GP',
            'pred': ts_mixer_gp_preds,
            'lower': ts_mixer_gp_lowers,
            'upper': ts_mixer_gp_uppers,
            'metrics': dict(),
      },
      'ts_mixer_gp_univariate': {
            'name': 'TS Mixer Time-series GP Univariate',
            'pred': ts_mixer_gp_univ_preds,
            'lower': ts_mixer_gp_univ_lowers,
            'upper': ts_mixer_gp_univ_uppers,
            'metrics': dict(),
      }
}

# %% [markdown]
# ## 2. Calculate Benchmark Metrics

# %%
# Compute RMSE for each model
for model in benchmarking.keys():
      benchmarking[model]['metrics']['rmse'] = calculate_rmse(trues, benchmarking[model]['pred'])
      print(f"{benchmarking[model]['name']} RMSE: {benchmarking[model]['metrics']['rmse']}")

# %%
# Compute MAPE for each model 
for model in benchmarking.keys():
      benchmarking[model]['metrics']['mape'] = calculate_mape(trues, benchmarking[model]['pred'])
      print(f"{benchmarking[model]['name']} MAPE: {benchmarking[model]['metrics']['mape']}")

# %%
# Compute PICP 
#only valid for models providing lower and upper prediction bounds
for model in benchmarking.keys():
      if 'lower' in benchmarking[model].keys() and 'upper' in benchmarking[model].keys():
            picp = calculate_picp(trues, benchmarking[model]['lower'], benchmarking[model]['upper'])
      else:
            picp = None
      benchmarking[model]['metrics']['picp'] = picp
      print(f"{benchmarking[model]['name']} PICP: {benchmarking[model]['metrics']['picp']}")

# %%
#Compute CRPS using the true values and model predictions and display results
for model in benchmarking.keys():
      benchmarking[model]['metrics']['crps'] = calculate_crps(trues, benchmarking[model]['pred'])
      print(f"{benchmarking[model]['name']} CRPS: {benchmarking[model]['metrics']['crps']}")

# %%
#Collect all evaluation metrics from the benchmarking dict
#into a new dictionary (benchmarking_scores) , then convert it
#into a pandas dataframe for tabular display
benchmarking_scores = dict()
for model_name, results in benchmarking.items():
      benchmarking_scores[model_name] = dict()
      benchmarking_scores[model_name] = results['metrics']
#create a data frame with model names as rows and metrics as columns
pd.DataFrame.from_dict(benchmarking_scores, orient='index')

# %% [markdown]
# ## 3. Benchmark Scores and Predictions Evaluation

# %%
#Compare overall models and display the results
figures = compare_scores(benchmarking)
plt.show()

# %%
#visualize predictions of all models for a single test case
fig, ax = compare_single_prediction(inputs, trues, benchmarking, test_case = 488)
plt.show()

# %%
#Visualize predictions of multivariate trained models for multiple selected test cases
test_cases = [0, 178, 367, 711]
fig, axs = compare_multi_predictions(inputs, trues, benchmarking, test_cases = test_cases)
plt.show()

# %% [markdown]
# ### Evaluation
#
# After training and some hyperparameter tuning, a lookback window of 168 hours (one week) and a prediction horizon of 24 hours (one day) were selected as the most suitable configuration for reliably capturing temporal dependencies and correlations in the data. Larger lookback windows led models to collapse toward predicting the average electricity price at a given time, with correspondingly wider prediction intervals. However, lookback windows shorter than 96 hours (four days) caused models to revert to simple sinusoidal predictions that failed to capture more complex fluctuations.
#
# Model performance was assessed using multiple key performance indicators, including RMSE, MAPE, PICP, and CRPS. Analysis of RMSE and CRPS revealed a consistent trend: the Sundial and TS-Mixer GP models achieved lower scores, indicating higher point-prediction accuracy. In contrast, the Time-series GP and PatchTST-based models exhibited higher RMSE values. However, the PICP metric, which measures the reliability of uncertainty quantification, highlighted a different perspective. Several Gaussian Process variants—specifically the Time-series GP, Neural Time-series GP, and PatchTST Time-series GP achieved a close to perfect coverage (PICP = 1.0), mostly due to having a very wide confidence interval. By comparison, the TS-Mixer GP and Sundial models produced narrower intervals with lower PICP values, reflecting sharper but less conservative uncertainty estimates.
#
# Across the selected metrics, Sundial outperforms the other models in most cases. On the other hand, the various GP models, do not perform well in the context of multi-step time-series predictions. While, the PatchTST and TSMixer extensions did help improve the prediction performance of GPs in multivariates scenarios, they were still behind the base PatchTST and TSMixer models. Also, the Neural extension (inclusion of a latent feature extractor) did help the GP make better predictions under a univariate case as highlighted by a lower RMSE, MAPE without a significant decrease in the PICP despite having a narrower confidence interval. Nevertheless, in most metrics, the various GP models were outperformed by the benchmark models.

# %% [markdown]
# ## Limitations and Future Work
# Several limitations were encountered during this analysis, which suggest avenues for future research. Due to capacity constraints, we were unable to implement advanced techniques such as Kernel Selection. It would also be beneficial to implement further hyperparameter optimization techniques for all models to ensure their peak performance. Furthermore, our experiments with a multivariate approach for Gaussian Process models revealed that using multiple highly correlated time series as inputs did not improve performance. This is because the high correlation increases dimensionality without adding new information, effectively introducing more noise. For this reason, the Base and Neural TSGP models performed better with a univariate approach, particularly for longer lookback and shorter horizon windows.
#
# The following tasks could be done to explore whether the extended GPs could be further improved:
# - introduce kernel selection and initilaization
# - introduce hyperparameter selection for each GP extension
# - tune the lookback and horizon windows
# - combine multiple GP extensions (e.g. patching and latent feature extraction)

# %%
