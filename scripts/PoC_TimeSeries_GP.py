# %% [markdown]
# # Proof-of-Concept: Time-Series GP

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

# %% [markdown]
# ## 3. Time-Series embedding

# %%
# Embedding
"""
# Sample Code
number_samples=600
seq_len=24
number_predictors=3

X=torch.randn(number_samples,seq_len,number_predictors)
y=torch.nn.ReLU()(X@torch.rand(number_predictors))+torch.sin(X@torch.rand(number_predictors))

X=(X-X.mean(axis=0))/X.std(axis=0)
y=(y-y.mean(axis=0))/y.std(axis=0)

print('Shape if predictor inputs: ',X.shape)
print('Shape of outputs: ',y.shape)
"""

# %%
seq_len=24
num_samples=600
test_samples=200
train_ts, test_ts = eu_df[:num_samples][['Germany (EUR/MWhe)']], eu_df[num_samples:num_samples + test_samples][['Germany (EUR/MWhe)']]
train_ts = train_ts.reset_index(drop=True)
test_ts = test_ts.reset_index(drop=True)
print(f'Train dataset shape: {train_ts.shape}\n'
      f'Test dataset shape: {test_ts.shape}\n')

# %%
# Scale the data (i.e. electricity prices)
scaler_ts = StandardScaler()
train_scaled_ts = scaler_ts.fit_transform(train_ts)
test_scaled_ts = scaler_ts.transform(test_ts)

print(f'Original Train dataset  mean: {round(train_ts.iloc[0].mean(), 2)}; \tstd: {round(train_ts.std().iloc[0], 2)}\n'
      f'Scaled Train dataset    mean: {round(train_scaled_ts.mean(), 2)}; \t\tstd: {round(train_scaled_ts.std(), 2)}\n'
      f'Scaled Test dataset     mean: {round(test_scaled_ts.mean(), 2)}; \tstd: {round(test_scaled_ts.std(), 2)}\n')

# %%
import pandas as pd
_x = list()
_y = list()
for i in range(train_scaled_ts.shape[1]):
    for row, value in enumerate(train_scaled_ts[:, i][:num_samples - seq_len * 2]):
        _x.append(train_scaled_ts[row:row + seq_len])
        _y.append(train_scaled_ts[int(row + seq_len):int(row + seq_len * 2), i])
pd.DataFrame(_y)

_x_test = list()
_y_test = list()
for i in range(test_scaled_ts.shape[1]):
    for row, value in enumerate(test_scaled_ts[:, i][:test_samples - seq_len * 2]):
        test_row = num_samples + row
        _x_test.append(test_scaled_ts[row:row + seq_len])
        _y_test.append(test_scaled_ts[int(row + seq_len):int(row + seq_len * 2), i])
pd.DataFrame(_y_test)


X_ts = Tensor(_x)
y_ts = Tensor(_y)
X_test_ts = Tensor(_x_test)
y_test_ts = Tensor(_y_test)
print('Shape of predictor inputs: ',X_ts.shape)
print('Shape of outputs: ',y_ts.shape)
print('Shape of test predictor inputs: ',X_test_ts.shape)
print('Shape of test outputs: ',y_test_ts.shape)

# %%
covar_module = gpytorch.kernels.SpectralMixtureKernel(ard_num_dims=X_ts.shape[-1],num_mixtures=1)
covar = covar_module(X_ts)
mean_module = LinearMean(input_size=X_ts.shape[-1])
mean=mean_module(X_ts)

#Let's examine the output shapes:
print('Shape of covariance matrix: ',covar.shape)
print('Shape of means: ',mean.shape)


# %%
class LatentFeatureExtractor(torch.nn.Module):
    def __init__(self,
                 input_dim:int,
                 latent_dimension:int,
                 num_mixtures:int,
                 num_layers:int,
                 dropout_rate:float,
                 skip_connections:bool,
                 normalize_latent_outputs:bool):

        '''
        Extracts (nonlinear) latent representation from input variables fed to a deep GP model. The LatentFeatureExtractor
        represents the deterministic part of the whole network.


        Parameters:
        -----------
        input_dim: int
            The number of input features.

        latent_dimension:int
            The number of hidden units each mixture component of the neural GP receives.

        num_mixtures:int
            Number of mixture components.

        num_layers:int
            Number of hidden layers.

        dropout_rate:float
            The percentage of hidden units to drop out during the forward pass.

        skip_connections:bool
            Whether to apply skip connections after each hidden layer.

        normalize_latent_outputs:bool
            Whether to normalize the latent representation produced by each hidden layer.


        Attributes:
        (in addition to the Parameters)
        -----------
        latent_output_dim: int
            An attribute, which is computed based on the product of latent_dimension with num_mixtures. It determines
            the amount of hidden units, which act as inputs to every mixture component of the neural GP.

        latent_layers: torch.nn.ModuleList
            The torch module list containing the nonlinear layers.

        norm_latent_layers: torch.nn.ModuleList
            The torch module list containing the normalization layers, applied in the forward pass after each hidden layer.


        Methods:
        ---------
        forward:
            Performs the forward pass with the latent feature extractor.

        '''


        super(LatentFeatureExtractor, self).__init__()
        self.input_dim=input_dim
        self.num_layers=num_layers
        self.dropout_rate=dropout_rate
        self.skip_connections=skip_connections
        self.normalize_latent_outputs=normalize_latent_outputs

        self.latent_dimension=latent_dimension
        self.num_mixtures=num_mixtures
        self.latent_output_dim=num_mixtures*latent_dimension

        #Put all trainable variables in a ModuleList, otherwise pytroch will not update the weights
        #in the latent feature extractor:
        self.latent_layers=torch.nn.ModuleList([torch.nn.Linear(input_dim, self.latent_output_dim) if nr_layer==0 else torch.nn.Linear(self.latent_output_dim, self.latent_output_dim) for nr_layer in range(0,self.num_layers+1)])
        #Latent normalization layers:
        if self.normalize_latent_outputs==True:
            self.norm_latent_layers=torch.nn.ModuleList([torch.nn.LayerNorm(self.latent_output_dim) for nr_layer in range(0,self.num_layers+1)])



    def forward(self,x):
        for nr_layer in range(0,self.num_layers+1):
            #Apply linear projection to a latent dimension:
            latent_output=(self.latent_layers[nr_layer](x) if nr_layer==0 else self.latent_layers[nr_layer](previous_latent_output))

            #Apply nonlinear transformations followed with dorpouts only to intermediate layers (excluding last linear projection layer):
            if nr_layer!=self.num_layers:
                #Apply nonlinear transformation: ReLU in this case.
                latent_output=torch.nn.ReLU()(latent_output)

                if self.dropout_rate!=0.0:
                    latent_output=torch.nn.Dropout(self.dropout_rate)(latent_output)

                #Apply skip connection if current layer in [1,self.num_layers-1]
                if nr_layer!=0:
                    if self.skip_connections==True:
                        latent_output=latent_output+previous_latent_output


            #Check if the latent feature extractor makes use of normalization layers:
            if self.normalize_latent_outputs==True:
                latent_output=self.norm_latent_layers[nr_layer](latent_output)

            previous_latent_output=latent_output

        #TO-DO:
        #When you set the self.num_mixtures parameter to anything higher than 1,
        #then here you would have to split the torch Tensor along the last dimension to obtain
        #the latent features for each GP mixture component, and encode an if-else condition what the output values
        #of the forward method should be based on self.num_mixtures
        return latent_output

# %%
from gpytorch.kernels import GridInterpolationKernel
class Exact_GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x,
                 likelihood,kernel_type,
                 latent_dimension,num_mixtures,num_layers,
                 dropout_rate,skip_connections,normalize_latent_outputs):

        '''
        Estimates an exact neural GP model.


        Parameters:
        -----------
        train_x: torch.Tensor
            The input sequences.

        likelihood:gpytorch.likelihoods.Likelihood
            The likelihood function, which is required as one of the input parameters to the superclass gpytorch.models.ExactGP.

        kernel_type:str
            The type of the kernel to use during the modeling process.

        The remaining input parameters:
        latent_dimension, num_mixtures, num_layers,
        dropout_rate,skip_connections,normalize_latent_outputs: the same as for the latent feature extractor.


        Attributes:
        -----------
        num_mixtures: int
            Number of Gp mixture components.

        feature_extractor: torch.nn.ModuleList
            The torch module list containing the nonlinear layers for extracting latent representation from the inputs.

        mean_module: gpytorch.means.Mean
            The module computing the vector of means for the multivariate Gaussian distribution.

        covar_module: gpytorch.kernels.Kernel
            The kernel module computing the covariance matrix for the multivariate Gaussian distribution.


        Methods:
        ---------
        forward(x):
            Performs the forward pass for input x with the mean and covar module.

        get_kernel(kernel_type,latent_dimension):
            Selects a kernel function based on the specified values in string format, i.e., kernel_type.
        '''

        super(Exact_GPRegressionModel, self).__init__(None,None,likelihood)
        self.num_mixtures=num_mixtures

        #Initialize latent feature extractor:
        self.feature_extractor = LatentFeatureExtractor(input_dim=train_x.shape[-1],latent_dimension=latent_dimension,
                                                        num_mixtures=1,
                                                        dropout_rate=dropout_rate,
                                                        num_layers=num_layers,skip_connections=skip_connections,
                                                        normalize_latent_outputs=normalize_latent_outputs)

        self.mean_module = gpytorch.means.LinearMean(input_size=latent_dimension)
        self.covar_module=self.get_kernel(kernel_type=kernel_type,latent_dimension=latent_dimension)


    def get_kernel(self,kernel_type,latent_dimension):

        if type(kernel_type)==str:
                covar_module = (GridInterpolationKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([latent_dimension])), grid_size=500, num_dims=1) if kernel_type=='RBF' else
                                (GridInterpolationKernel(gpytorch.kernels.MaternKernel(batch_shape=torch.Size([latent_dimension])), grid_size=500, num_dims=1) if kernel_type=='Matern' else # kernel_type=='Spectral'
                                GridInterpolationKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=self.num_mixtures,batch_shape=torch.Size([latent_dimension])), grid_size=500, num_dims=1)))

        else:#list of kernels:
            for idx in range(0,len(kernel_type)):
                if idx==0:
                    covar_module = (GridInterpolationKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([latent_dimension])), grid_size=500, num_dims=1)  if kernel_type[idx]=='RBF' else
                                    (GridInterpolationKernel(gpytorch.kernels.MaternKernel(batch_shape=torch.Size([latent_dimension])), grid_size=500, num_dims=1) if kernel_type[idx]=='Matern' else #kernel_type[idx]=='Spectral'
                                     GridInterpolationKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=self.num_mixtures,batch_shape=torch.Size([latent_dimension])), grid_size=500, num_dims=1)))
                else:
                    covar_module = covar_module+(GridInterpolationKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([latent_dimension])), grid_size=500, num_dims=1)  if kernel_type[idx]=='RBF' else
                                    (GridInterpolationKernel(gpytorch.kernels.MaternKernel(batch_shape=torch.Size([latent_dimension])), grid_size=500, num_dims=1) if kernel_type[idx]=='Matern' else #kernel_type[idx]=='Spectral'
                                     GridInterpolationKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=self.num_mixtures,batch_shape=torch.Size([latent_dimension])), grid_size=500, num_dims=1)))

        return ScaleKernel(covar_module)

    def forward(self, x):
        projected_x = self.feature_extractor(x)

        mean_x = self.mean_module(projected_x)
        univariate_covars = self.covar_module(projected_x.mT.unsqueeze(-1))
        covar_x = univariate_covars.sum(dim=-3)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# %%
#initialize the model and the device to train the neural GP on:
likelihood_ts = gpytorch.likelihoods.GaussianLikelihood()
num_mixtures=4
model_ts = Exact_GPRegressionModel(train_x=X_ts,
                                likelihood=likelihood_ts,
                                #Hyperparameters related to deep latent feature extractor:
                                latent_dimension=200,num_layers=3,dropout_rate=0.0,
                                skip_connections=False,normalize_latent_outputs=True,
                                #hyperparameters related to probabilistic forecasts:
                                num_mixtures=num_mixtures,kernel_type='RBF_Matern_Spectral'.split('_'))

if torch.cuda.is_available():
    model_ts = model_ts.cuda()
    likelihood = likelihood_ts.cuda()
    X_ts=X_ts.to(device='cuda')
    y_ts=y_ts.to(device='cuda')

# %%
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_ts, y_ts)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# %% jupyter={"is_executing": true}
#Train the deep GP time series model:
model_ts.train()
likelihood_ts.train()
optimizer = torch.optim.Adam(model_ts.parameters(), lr=0.005)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_ts, model_ts)


training_iterations = 15

for i in range(0,training_iterations):
    epoch_loss=[]
    print('Iteration: ',i+1)
    for batch_idx,(x_train, y_train) in enumerate(train_loader):

        #necessary in the case of exact GPs:
        model_ts.set_train_data(inputs=x_train, targets=y_train, strict=False)

        print(f'set_train_data called: {i+1}, {batch_idx+1}')
        with gpytorch.settings.fast_pred_samples():
            optimizer.zero_grad()
            print(f'optimizer called: {i+1}, {batch_idx+1}')
            #Perform forward pass:
            output = model_ts(x_train)
            print(f'performed forward pass: output: {i+1}, {batch_idx+1}')
            current_loss = -mll(output, y_train)
            print(f'performed forward pass: current_loss: {i+1}, {batch_idx+1}')
            loss=current_loss.mean()
            print(f'performed forward pass: {i+1}, {batch_idx+1}')

            #Perform the backward pass:
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            print(f'performed backward pass: {i+1}, {batch_idx+1}')

    print('Training Iteration/Epoch : ',i+1,', Loss Value: ',np.mean(epoch_loss))


# %%
model_ts.eval()
likelihood_ts.eval()

# %%
with open('../models/model_ts.pickle', 'wb') as f:
    pickle.dump(model_ts, f)

with open('../models/likelihood_ts.pickle', 'wb') as f:
    pickle.dump(likelihood_ts, f)

with open('../models/output_ts.pickle', 'wb') as f:
    pickle.dump(output, f)

# %%
with open('../models/model_ts.pickle', 'rb') as f:
    model_ts = pickle.load(f)

with open('../models/likelihood_ts.pickle', 'rb') as f:
    likelihood_ts = pickle.load(f)

with open('../models/output_ts.pickle', 'rb') as f:
    output = pickle.load(f)

# %%
test_dataset = TensorDataset(X_test_ts, y_test_ts)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
preds = list()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions
    for test in X_test_ts:
        observed_pred = likelihood_ts(model_ts(test))
        preds.append(observed_pred)

preds[-1:]

# %%
# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
#with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions
    #observed_pred = likelihood_ts(model_ts(test_x2))

# Reformat train and test data
_x_input = list(range(test_samples-seq_len))
_y_input = test_ts['Germany (EUR/MWhe)'][:test_samples-seq_len]
_x_test = list(range(test_samples-seq_len,test_samples))
_y_test = test_ts['Germany (EUR/MWhe)'][test_samples-seq_len:]

# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(14, 5))

# Get upper and lower confidence bounds
lower, upper = observed_pred.confidence_region()
mean = observed_pred.mean

# Rescale target
observed_pred_os = scaler_ts.inverse_transform([mean.numpy().mean(axis=0)]).T[:, 0]
lower_os = scaler_ts.inverse_transform([lower.numpy().mean(axis=0)]).T[:, 0]
upper_os = scaler_ts.inverse_transform([upper.numpy().mean(axis=0)]).T[:, 0]

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
#ax.set_xlim(-10, 20)
#ax.set_ylim([-20, 160])
ax.legend(['Input Data', 'Observed Data', 'Prediction', 'Confidence Interval'])

# %%
import matplotlib.pyplot as plt
fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
ax[0].hist(y_ts[0,:].cpu(),label='True Values')
ax[0].hist(output.sample()[0,:].cpu(),label='1. Sample')
ax[0].hist(output.sample()[0,:].cpu(),label='2. Sample')
ax[0].hist(output.sample()[0,:].cpu(),label='3. Sample')
ax[0].hist(output.mean.cpu().detach().numpy()[0,:],label='Mean of GP',alpha=0.75)
ax[0].set_title('True Values vs. estimated GP Distribution\n1.Sequence')
ax[0].set_xlabel('Binned Sequence Values')
ax[0].set_ylabel('Values sof Y')
ax[0].legend(ncols=3,mode='expand',loc=[0.0,-0.30])

ax[1].hist(y_ts[1,:].cpu(),label='True Values')
ax[1].hist(output.sample()[1,:].cpu(),label='1. Sample')
ax[1].hist(output.sample()[1,:].cpu(),label='2. Sample')
ax[1].hist(output.sample()[1,:].cpu(),label='3. Sample')
ax[1].hist(output.mean.cpu().detach().numpy()[1,:],label='Mean of GP',alpha=0.75)
ax[1].set_title('True Values vs. estimated GP Distribution\n2.Sequence')
ax[1].set_xlabel('Binned Sequence Values')
ax[1].set_ylabel('Values sof Y')

ax[2].hist(y_ts[2,:].cpu(),label='True Values')
ax[2].hist(output.sample()[2,:].cpu(),label='1. Sample')
ax[2].hist(output.sample()[2,:].cpu(),label='2. Sample')
ax[2].hist(output.sample()[2,:].cpu(),label='3. Sample')
ax[2].hist(output.mean.cpu().detach().numpy()[2,:],label='Mean of GP',alpha=0.75)
ax[2].set_title('True Values vs. estimated GP Distribution\n3.Sequence')
ax[2].set_xlabel('Binned Sequence Values')
ax[2].set_ylabel('Values sof Y')

fig.tight_layout()
plt.show()
plt.close()

# %% [markdown]
# ## 4. Export Example Data and Predictions

# %%
pred_dist = scaler.inverse_transform(mean.numpy()).T
pred_lower_dist = scaler.inverse_transform(upper.numpy()).T
pred_upper_dist = scaler.inverse_transform(upper.numpy()).T


np.savetxt('../data/example_input.csv',_y_input,delimiter=",")
np.savetxt('../data/example_true.csv',_y_test,delimiter=",")
np.savetxt('../data/example_pred_dist.csv',pred_dist,delimiter=",")
np.savetxt('../data/example_pred_lower_dist.csv',pred_lower_dist,delimiter=",")
np.savetxt('../data/example_pred_upper_dist.csv',pred_upper_dist,delimiter=",")

# %%
np.genfromtxt("../data/example_pred_upper_dist.csv", delimiter=",")

# %%
np.genfromtxt("../data/example_pred_dist.csv", delimiter=",")
