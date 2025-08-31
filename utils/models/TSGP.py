from gpytorch import Module
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.settings import fast_pred_var
from gpytorch.variational import CholeskyVariationalDistribution, LMCVariationalStrategy, VariationalStrategy
from torch import no_grad, Size, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

class MultistepSVGP(ApproximateGP):
    """
    Extends GPyTorch's ApproximateGP for multi-step time-series
    forecasting using a sparse variational approach. It combines a base
    variational strategy with an LMC (Linear Model of Coregionalization)
    strategy to handle multiple forecast horizons simultaneously.

    Args:
        inducing_points (Tensor): Initial inducing point locations,
            shape (M, D) where M = number of inducing points and D = input dimension
        horizon (int): Number of future steps to predict (multi-task outputs)
        num_latents_svgp (int): Number of latent GPs to use in the LMC variational strategy

    Attributes:
        mean_module (gpytorch.means.Mean): Constant mean function
        covar_module (gpytorch.kernels.Kernel): Scaled RBF kernel with ARD
        variational_strategy (gpytorch.variational.VariationalStrategy):
            Wraps the variational distribution and defines the approximate inference
    """
    def __init__(self, inducing_points, horizon, num_latents_svgp):
        # Number of inducing point (M)
        num_inducing_points = inducing_points.size(0) # num_inducing_points (P) ≤ total number of samples (N)

        # Variational distribution for inducing points , batched across Latent GPs
        variational_dist = CholeskyVariationalDistribution(
            num_inducing_points,
            batch_shape=Size([num_latents_svgp])
        )

        # Base variational strategy approximates the GP using a small set of inducing points
        base_var_strategy = VariationalStrategy(
            self,  # model reference
            inducing_points,
            variational_dist,
            learn_inducing_locations=True
        )

        # LMC variational strategy for multistep outputs
        variational_strategy = LMCVariationalStrategy(
            base_var_strategy,
            num_tasks=horizon,
            num_latents=num_latents_svgp
        )

        super().__init__(variational_strategy)

        # Mean and covariance modules
        input_dim = inducing_points.size(1)  # input_dim (d0) = lookback (L) * num_features (V); i.e. flattened input
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dim))

    def forward(self, x):
        """
        Forward pass of the GP model :

        Args :
        x (Tensor) : Input data of shape (N= number of data points , D=input dimension)

        Returns:
        - mean_x (Tensor): mean vector from the mean module
        - covar_x(Tensor): covariance matrix from the kernel
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class TSGPModel(Module):
    """
    Time-Series GP (TSGP) wrapper:
    - Flattens sequences
    - Trains with variational ELBO
    - Infers mean + confidence region per horizon step
    """
    def __init__(self, inducing_points, horizon, num_latents_svgp):
        super().__init__()
        # Flatten inducing points to (M, D)
        inducing_points = inducing_points.reshape(inducing_points.size(0), -1)
        self.gp_layer = MultistepSVGP(inducing_points, horizon, num_latents_svgp)
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=horizon)

    def forward(self, x):
        # input data needs to be reshaped into 2D-array before passing to GP
        x = x.reshape(x.size(0), -1)
        return self.gp_layer(x)

    def train_model(
            self,
            train_loader: DataLoader,
            num_data: int,
            epochs: int = 25,
            add_optimizer_params: list = []):
        # Variational training with ELBO over (gp_layer + likelihood)
        self.gp_layer.train()
        self.likelihood.train()

        mll = VariationalELBO(
            self.likelihood,
            self.gp_layer,
            num_data=num_data  # num_samples (N)
        )
        # Optimizer
        params = [
            {'params': self.gp_layer.parameters()},
            {'params': self.likelihood.parameters()},
        ] + add_optimizer_params
        optimizer = Adam(params, lr=0.01)

        for epoch in range(epochs):
            running_loss = 0.0

            # tqdm progress bar for batches
            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                optimizer.zero_grad()

                y_batch = y_batch.reshape(y_batch.size(0), -1)
                output = self.__call__(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            tqdm.write(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

        return

    def infer(
            self,
            input: Tensor,
            true_shape = None
    ):
        """
        Predict mean and confidence

        Args:
          x_test (Tensor) : Test inputs of shape
          true_shape : If provided , the outputs will be reshaped to this size

        Returns:
          preds : Predicted mean values
          lowers: Lower bound of the confidence interval
          uppers: Upper bound of the confidence interval

        """
        self.gp_layer.eval()
        self.likelihood.eval()

        with no_grad(), fast_pred_var():
            predictions = self.likelihood(self.__call__(input))
            preds = predictions.mean
            lowers, uppers = predictions.confidence_region()

        # reshape output back to original shape of expected output
        if true_shape:
            preds = Tensor(preds).reshape(true_shape)
            lowers = Tensor(lowers).reshape(true_shape)
            uppers = Tensor(uppers).reshape(true_shape)

        return preds, lowers, uppers



