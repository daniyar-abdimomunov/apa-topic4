from torch.nn import Module
from gpytorch.utils.grid import ScaleToBounds
from utils.models.MultistepSVGP import TSGPModel
from utils.models.PatchTST import PatchTST

class PatchTSTFeatureExtractor(Module):
    def __init__(self, num_variables, time_steps, latent_dimension=50, patch_size=16, embed_dim=128,
                 num_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_tst = PatchTST(num_variables, time_steps, patch_size, embed_dim,
                                  num_layers, num_heads, output_steps=latent_dimension,
                                  dropout=dropout, return_embeddings=True)

    def forward(self, x):
        return self.patch_tst(x)  # (B, latent_dim)

class PatchTSTGPModel(TSGPModel):
    def __init__(self, inducing_points, horizon, lookback, num_latents_svgp, grid_bounds=(-1., 1.), latent_dimension=50, patch_size=16, embed_dim=128,
                 num_layers=3, num_heads=4, dropout=0.1):

        num_input_vars = int(inducing_points.size(-1))
        patch_tst_layer = PatchTSTFeatureExtractor(
            num_variables=num_input_vars,
            time_steps=lookback,
            latent_dimension=latent_dimension,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        scale_to_bounds = ScaleToBounds(grid_bounds[0], grid_bounds[1])

        inducing_points = patch_tst_layer(inducing_points)
        inducing_points = scale_to_bounds(inducing_points)
        super().__init__(inducing_points, horizon, num_latents_svgp)

        self.patch_tst_layer = patch_tst_layer
        self.scale_to_bounds = scale_to_bounds

    def forward(self, x):
        patched_x = self.patch_tst_layer(x)
        patched_x = self.scale_to_bounds(patched_x)

        return super().forward(patched_x)

    def train_model(self, *args, **kwargs):
        self.patch_tst_layer.train()
        super().train_model(add_optimizer_params = [{'params': self.patch_tst_layer.parameters()}], *args,  **kwargs)

    def infer(self, *args, **kwargs):
        self.patch_tst_layer.eval()
        return super().infer(*args, **kwargs)