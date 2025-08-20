# %% [markdown]
# # Proof-of-Concept 02: Multistep (i.e. Batch Independent Multioutput) GP

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import gpytorch
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import torch

from __init__ import plot_predictions, sequentialize

# %% [markdown]
# ## 1. Data Import and Processing

# %%
# import the data:
with open('../data/EU_Electricity_TimeSeries.pkl','rb') as f:
    eu_df=pickle.load(f)

print(f'Dataset shape: {eu_df.shape}\n')
eu_df

# %%
# define train-test split and prediction parameters
LOOKBACK = 192
HORIZON = 96
NUM_TRAIN_SAMPLES = 3000 # Must be ≥ (LOOKBACK + HORIZON)
NUM_TEST_SAMPLES = 1000 # Must be ≥ (LOOKBACK + HORIZON)

# %%
# split data into train and test datasets
train = eu_df[:NUM_TRAIN_SAMPLES]['Germany (EUR/MWhe)'].to_numpy()
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


# %% [markdown]
# ## 2. Train Multistep (ie Batch Independent Multioutput) GP

# %%
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = likelihood.num_tasks
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.num_tasks])),
            batch_shape=torch.Size([self.num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


# %%
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=likelihood.num_tasks.item()
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=likelihood.num_tasks.item(), rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# %%
# instantiate likelihood and model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=HORIZON)
model = BatchIndependentMultitaskGPModel(train_input, train_true, likelihood)
#model = MultitaskGPModel(train_input, train_true, likelihood)

# %%
# train model
NUM_ITERATIONS = 70

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(NUM_ITERATIONS):
    optimizer.zero_grad()
    output = model(train_input)
    loss = -mll(output, train_true)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, NUM_ITERATIONS, loss.item()))
    optimizer.step()

# %%
# save model
with open(f'../models/multitask_batch_l{LOOKBACK}_h{HORIZON}_t{NUM_ITERATIONS}_model.pickle', 'wb') as f:
    pickle.dump(model, f)

with open(f'../models/multitask_batch_l{LOOKBACK}_h{HORIZON}_t{NUM_ITERATIONS}_likelihood.pickle', 'wb') as f:
    pickle.dump(likelihood, f)


# %%
# load trained model, if necessary
with open('../models/multitask_batch_l192_h96_t40_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('../models/multitask_batch_l192_h96_t40_likelihood.pickle', 'rb') as f:
    likelihood = pickle.load(f)

# %%
# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_input[:]))
    preds = predictions.mean
    lowers, uppers = predictions.confidence_region()

# %%
# Rescale target to original scale (os)
preds_os = scaler.inverse_transform(preds.numpy())
lowers_os = scaler.inverse_transform(lowers.numpy())
uppers_os = scaler.inverse_transform(uppers.numpy())

# %%
# plot predictions
TEST_CASE = 0

plot_predictions(
    input=scaler.inverse_transform(test_input.numpy())[TEST_CASE],
    true=scaler.inverse_transform(test_true.numpy())[TEST_CASE],
    pred=preds_os[TEST_CASE],
    lower=lowers_os[TEST_CASE],
    upper=uppers_os[TEST_CASE],
    title=f'Time-series GP Model Predictions\nTest Case: #{TEST_CASE}'
)

# %%
# save predictions
np.savetxt('../data/mt_batch_ts_gp_preds.csv', preds_os, delimiter=",")
np.savetxt('../data/mt_batch_ts_gp_lowers.csv',lowers_os,delimiter=",")
np.savetxt('../data/mt_batch_ts_gp_uppers.csv',uppers_os,delimiter=",")
