from torch.nn import Sequential, Linear, ReLU, Module, ModuleList, Flatten
from gpytorch.utils.grid import ScaleToBounds
from utils.models.MultistepSVGP import TSGPModel
from utils.models.TSMixer import MTSMixerBlock
from utils.models.PatchTST import PatchTST

import torch

class LargeFeatureExtractor(Sequential):
    def __init__(self, input_dim, latent_dimension):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', Linear(input_dim, 1000))
        self.add_module('relu1', ReLU())
        self.add_module('linear2', Linear(1000, 500))
        self.add_module('relu2', ReLU())
        self.add_module('linear3', Linear(500, 50))
        self.add_module('relu3', ReLU())
        self.add_module('linear4', Linear(50, latent_dimension))

class TSMixerFeatureExtractor(Module):
    def __init__(self, num_variables, time_steps, latent_dimension, num_blocks=3, hidden_dim=64, activation='gelu'):
        super().__init__()
        self.blocks = ModuleList([
            MTSMixerBlock(num_variables, time_steps, hidden_dim, activation)
            for _ in range(num_blocks)
        ])

        self.flatten = Flatten(start_dim=1)
        self.proj = Linear(num_variables * time_steps, latent_dimension)


    def forward(self, x):
        for block in self.blocks:
            x = block(x)                #(B, T, D)
        x = self.proj(self.flatten(x))  #(B,T*D) -> (B, latent_dim)
        return x  # embedding instead of prediction
    

class PatchTSTFeatureExtractor(Module):
    def __init__(self, num_variables, time_steps, latent_dimension=50, patch_size=16, embed_dim=128,
                 num_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_tst = PatchTST(num_variables, time_steps, patch_size, embed_dim,
                                  num_layers, num_heads, output_steps=latent_dimension,
                                  dropout=dropout, return_embeddings=True)

    def forward(self, x):
        return self.patch_tst(x)  # (B, latent_dim)


class NeuralTSGPModel(TSGPModel):
    def __init__(self, inducing_points, horizon, num_latents_svgp, num_latents_lfe, grid_bounds=(-1., 1.)):
        num_input_vars = inducing_points.size(-1)
        latent_feature_extractor = LargeFeatureExtractor(num_input_vars, latent_dimension=num_latents_lfe)
        scale_to_bounds = ScaleToBounds(grid_bounds[0], grid_bounds[1])

        inducing_points = latent_feature_extractor(inducing_points)
        inducing_points = scale_to_bounds(inducing_points)
        super().__init__(inducing_points, horizon, num_latents_svgp)

        self.latent_feature_extractor = latent_feature_extractor
        self.scale_to_bounds = scale_to_bounds


    def forward(self, x):
        projected_x = self.latent_feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)

        return super().forward(projected_x)

    def train_model(self, *args, **kwargs):
        self.latent_feature_extractor.train()
        super().train_model(add_optimizer_params = [{'params': self.latent_feature_extractor.parameters()}], *args, **kwargs)

    def infer(self, *args, **kwargs):
        self.latent_feature_extractor.eval()
        return super().infer(*args, **kwargs)


class TSMixerGPModel(TSGPModel):
    def __init__(self, inducing_points, horizon, num_latents_svgp, num_latents_lfe,
                 grid_bounds=(-1., 1.), num_variables=1, time_steps=192):

        # initialize GP with dummy inducing points in latent space
        super().__init__(
            inducing_points=torch.randn(inducing_points.shape[0], num_latents_lfe),
            horizon=horizon,
            num_latents_svgp=num_latents_svgp
        )


        # trainable feature extractor
        self.ts_mixer_layer = TSMixerFeatureExtractor(
            num_variables=num_variables,
            time_steps=time_steps,
            latent_dimension=num_latents_lfe
        )

        self.scale_to_bounds = ScaleToBounds(*grid_bounds)

    def forward(self, x):
        # project inputs dynamically
        mixed_x = self.ts_mixer_layer(x)
        mixed_x = self.scale_to_bounds(mixed_x)
        return super().forward(mixed_x)

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
