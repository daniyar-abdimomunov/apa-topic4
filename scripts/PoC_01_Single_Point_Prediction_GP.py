# %% [markdown]
# # Proof-of-Concept 01: Single-Point Prediction GP

# %%
import gpytorch
from gpytorch.means import LinearMean
from gpytorch.kernels import ScaleKernel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor

# %% [markdown]
# ## 1. Data Import and Processing

# %%
#Import the data:
with open('../data/EU_Electricity_TimeSeries.pkl','rb') as f:
    eu_df=pickle.load(f)

#subsets=list(eu_df['Subset'])
#eu_df['Subset'].unique().tolist()
print(f'Dataset shape: {eu_df.shape}\n')
eu_df

# %%
# Split data into train and test sets
train, test = eu_df['2024-02-26':'2024-03-01'][['Germany (EUR/MWhe)']], eu_df['2024-03-02':][['Germany (EUR/MWhe)']]
print(f'Train dataset shape: {train.shape}\n'
      f'Test dataset shape: {test.shape}\n')

# %%
# Scale the data (i.e. electricity prices)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)[:, 0]
test_scaled = scaler.transform(test)[:, 0]

print(f'Original Train dataset  mean: {round(train.iloc[0].mean(), 2)}; \tstd: {round(train.std().iloc[0], 2)}\n'
      f'Scaled Train dataset    mean: {round(train_scaled.mean(), 2)}; \t\tstd: {round(train_scaled.std(), 2)}\n'
      f'Scaled Test dataset     mean: {round(test_scaled.mean(), 2)}; \tstd: {round(test_scaled.std(), 2)}\n')

# %%
x_train, y_train = Tensor(np.array(range(len(train_scaled))) / 24.0).view(-1,1), Tensor(train_scaled)
x_test, y_test = Tensor(np.array(range(len(test_scaled))) / 24.0 + 5).view(-1,1), Tensor(test_scaled)

plt.plot(x_train, y_train)

# %% [markdown]
# ## 2. Simple Regression

# %%
import torch
import matplotlib.pyplot as plt

# Generate a synthetic time-series signal
t = x_train[:, 0]
signal = y_train
#t = torch.linspace(0, 1, 500)  # Time range (1 second, 500 samples)
#signal = torch.sin(2 * torch.pi * 50 * t) + 0.5 * torch.sin(2 * torch.pi * 120 * t) + 0.1 * torch.randn_like(t)

# Compute the Fourier Transform
fft_values = torch.fft.fft(signal)  # Apply FFT
frequencies = torch.fft.fftfreq(signal.shape[0], d=(t[1] - t[0])) # Compute frequency bins

# Plot the frequency spectrum
plt.figure(figsize=(8, 5))
plt.plot(frequencies[:len(frequencies)//2], torch.abs(fft_values[:len(frequencies)//2]).numpy())  # Only positive frequencies
plt.title("Fourier Transform: Frequency Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.show()


# %%
class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    num_mixtures = 3

    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=self.num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def plot_kernels(self):
        plt.figure(figsize=(8, 5))
        weights = self.covar_module.mixture_weights.detach().view(-1)
        means = self.covar_module.mixture_means.detach().view(-1)
        scales = self.covar_module.mixture_scales.detach().view(-1)
        x_vals = x_train.view(-1, 1)

        for i in range(self.num_mixtures):
            component = weights[i] * torch.exp(-0.5 * ((x_vals - means[i]) ** 2) / scales[i] ** 2)
            plt.plot(x_vals.numpy(), component.numpy(), label=f'Mixture {i+1}')
        #plt.xlim(0,25)
        #plt.xticks(np.arange(0, 25, 1))
        plt.show()

    def visualize_smk_components(self, x):
        kernel=self.covar_module
        x = x.squeeze() if x.dim() > 1 else x
        N = x.size(0)

        fig, axes = plt.subplots(1, kernel.num_mixtures,  figsize=(10 * kernel.num_mixtures, 10))

        if kernel.num_mixtures == 1:
            axes = [axes]

        for i in range(kernel.num_mixtures):
            # Extract i-th component parameters
            mixture_weight = kernel.mixture_weights[i]
            mixture_mean = kernel.mixture_means[i]
            mixture_scale = kernel.mixture_scales[i]

            # Create a new single-component SM kernel
            component_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=1)
            component_kernel.mixture_weights = mixture_weight.unsqueeze(0)
            component_kernel.mixture_means = mixture_mean.unsqueeze(0)
            component_kernel.mixture_scales = mixture_scale.unsqueeze(0)

            # Evaluate covariance matrix
            with torch.no_grad():
                cov_matrix = component_kernel(x.unsqueeze(-1), x.unsqueeze(-1)).evaluate()

            # Plot heatmap
            sns.heatmap(cov_matrix.numpy(), ax=axes[i], cmap='viridis')
            axes[i].set_title(f'Component {i + 1}')

        plt.tight_layout()
        plt.show()

    def visualize_smk(self, x):
        kernel = self.covar_module
        with torch.no_grad():
            cov_matrix = kernel(x).evaluate()

        # Convert to numpy for plotting
        cov_matrix_np = cov_matrix.numpy()

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cov_matrix_np, cmap="viridis")
        plt.title("Spectral Mixture Kernel Covariance Heatmap")
        plt.xlabel("Time Index")
        plt.ylabel("Time Index")
        plt.tight_layout()
        plt.show()


# %%
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGPModel(x_train, y_train, likelihood)

# %%
model.plot_kernels()

# %%
model.visualize_smk_components(x_train)

# %%
model.visualize_smk(x_train)

# %%
training_iter = 50

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(x_train)
    loss = -mll(output, y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()

# %%
model.visualize_smk(x_train)

# %%
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions
    observed_pred = likelihood(model(x_test))

# %%
x_test

# %%
observed_pred.mean.shape

# %%
# Reformat train and test data
_x_input = x_train.numpy()[:, 0]
_y_input = train.to_numpy()[:,0]
_x_test = x_test.numpy()[:, 0]
_y_test = test.to_numpy()[:,0]

# Get prediction, upper and lower confidence bounds
lower, upper = observed_pred.confidence_region()
mean = observed_pred.mean

# Rescale target
observed_pred_os = scaler.inverse_transform([mean.numpy()]).T[:, 0]
lower_os = scaler.inverse_transform([lower.numpy()]).T[:, 0]
upper_os = scaler.inverse_transform([upper.numpy()]).T[:, 0]

# %%
# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(14, 5))

# Plot training data as black stars
ax.plot(_x_input, _y_input, 'g')
# Plot predictive means as blue line
ax.plot(_x_test, _y_test, 'r.', alpha=0.5)
# Plot predictive means as blue line
ax.plot(_x_test, observed_pred_os, 'b')
# Shade between the lower and upper confidence bounds
ax.fill_between(x=_x_test, y1=lower_os, y2=upper_os, alpha=0.5)
ax.set_xlabel('Days')
ax.set_ylabel('Price (EUR/MWhe)')
ax.set_xlim(-10, 20)
##ax.set_ylim([-20, 160])
ax.legend(['Input Data', 'Observed Data', 'Prediction', 'Confidence Interval'])
