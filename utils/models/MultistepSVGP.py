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
    def __init__(self, inducing_points, horizon, num_latents_svgp):
        # Variational distribution over inducing points (full covariance)
        num_inducing_points = inducing_points.size(0) # num_inducing_points (P) ≤ total number of samples (N)
        variational_dist = CholeskyVariationalDistribution(
            num_inducing_points,
            batch_shape=Size([num_latents_svgp])
        )

        # Base variational strategy (shared inducing points)
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
        # x should have shape [batch_size (M), input_dim (d0)]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class TSGPModel(Module):
    def __init__(self, inducing_points, horizon, num_latents_svgp):
        super().__init__()
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
            epochs: int = 25):
        self.gp_layer.train()
        self.likelihood.train()

        mll = VariationalELBO(
            self.likelihood,
            self.gp_layer,
            num_data=num_data  # num_samples (N)
        )
        # Optimizer
        optimizer = Adam([
            {'params': self.gp_layer.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)

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



