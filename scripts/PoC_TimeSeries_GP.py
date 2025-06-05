# %% [markdown]
# # Proof-of-Concept: Time-Series GP

# %%
import gpytorch
from gpytorch.means import LinearMean
from gpytorch.kernels import ScaleKernel
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
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
eu_df.shape

# %%
eu_df

# %%
eu_df['Germany (EUR/MWhe) norm.'] = (eu_df['Germany (EUR/MWhe)'] - eu_df['Germany (EUR/MWhe)'].mean(axis=0)) / eu_df['Germany (EUR/MWhe)'].std(axis=0)

# %%
train, test = eu_df['2024-02-28':'2024-03-01']['Germany (EUR/MWhe) norm.'], eu_df['2024-03-02':]['Germany (EUR/MWhe) norm.']
train_x, train_y = Tensor(train.index.astype(np.int64)  // 10**9 / 86400 - 19779).view(-1,1), Tensor(train)
test_x, test_y = Tensor(test.index.astype(np.int64)  // 10**9 / 86400 - 19779), Tensor(test)

plt.plot(train_x, train_y)

# %% [markdown]
# ## 2. Simple Regression

# %%
import torch
import matplotlib.pyplot as plt

# Generate a synthetic time-series signal
t = Tensor(train.index.astype(np.int64)  // 10**9 / 86400 - 19605)
signal = Tensor(train)
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
        x_vals = train_x.view(-1, 1)

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
model = SpectralMixtureGPModel(train_x, train_y, likelihood)

# %%
model.plot_kernels()

# %%
model.visualize_smk_components(train_x)

# %%
model.visualize_smk(train_x)

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
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()

# %%
model.visualize_smk(train_x)

# %%
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions
    observed_pred = likelihood(model(test_x))

    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(14, 5))


    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Rescale target
    _train_y = train_y * eu_df['Germany (EUR/MWhe)'].std(axis=0) + eu_df['Germany (EUR/MWhe)'].mean(axis=0)
    _observed_pred = observed_pred * eu_df['Germany (EUR/MWhe)'].std(axis=0) + eu_df['Germany (EUR/MWhe)'].mean(axis=0)
    _lower = lower * eu_df['Germany (EUR/MWhe)'].std(axis=0) + eu_df['Germany (EUR/MWhe)'].mean(axis=0)
    _upper = upper * eu_df['Germany (EUR/MWhe)'].std(axis=0) + eu_df['Germany (EUR/MWhe)'].mean(axis=0)
    _test_y = test_y * eu_df['Germany (EUR/MWhe)'].std(axis=0) + eu_df['Germany (EUR/MWhe)'].mean(axis=0)

    # Plot training data as black stars
    ax.plot(train_x.numpy(), _train_y.numpy())
    # Plot predictive means as blue line
    ##ax.plot(test_x.numpy(), _test_y.numpy(), 'r.', alpha=0.5)
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), _observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x=test_x.numpy(), y1=_lower.numpy(), y2=_upper.numpy(), alpha=0.5)
    ax.set_xlabel('Days')
    ax.set_ylabel('Price (EUR/MWhe)')
    ax.set_xlim(-10, 20)
    ax.set_ylim([-20, 160])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

# %%
eu_df['Germany (EUR/MWhe)'].std(axis=0)

# %% [markdown]
# ## 3. Time-Series embedding

# %%
...
