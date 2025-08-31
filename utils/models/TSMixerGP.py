from torch.nn import Linear, Module, ModuleList, Flatten
from gpytorch.utils.grid import ScaleToBounds
from utils.models.TSGP import TSGPModel
from utils.models.TSMixer import MTSMixerBlock

import torch

class TSMixerFeatureExtractor(Module):
    """
    Simple TSMixer-based feature extractor to project inputs to a latent space for the GP
    
    """
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

class TSMixerGPModel(TSGPModel):
    """

    TSGP variant with a learnable TSMixer feature extractor :
    - Projects inputs with TSMixerFeatureExtractor
    - Scales features to fixed bounds for stable GP behavior

    """
    def __init__(self, inducing_points, horizon, lookback, num_latents_svgp, num_latents_lfe,
                 grid_bounds=(-1., 1.), num_blocks=3, hidden_dim=64, activation='gelu'):
        
        num_input_vars = int(inducing_points.size(-1))

        # initialize GP with dummy inducing points in latent space
        super().__init__(
            inducing_points=torch.randn(inducing_points.shape[0], num_latents_lfe),
            horizon=horizon,
            num_latents_svgp=num_latents_svgp
        )


        # trainable feature extractor
        self.ts_mixer_layer = TSMixerFeatureExtractor(
            num_variables=num_input_vars,
            time_steps=lookback,
            latent_dimension=num_latents_lfe,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            activation=activation
        )

        self.scale_to_bounds = ScaleToBounds(*grid_bounds)

    def forward(self, x):
        # project inputs dynamically
        mixed_x = self.ts_mixer_layer(x)
        mixed_x = self.scale_to_bounds(mixed_x)
        return super().forward(mixed_x)
    
    def fit(self, *args, **kwargs):
        # ensure extractor is in train mode
        self.ts_mixer_layer.train()
        # add extractor params to optimizer
        super().fit(
            add_optimizer_params=[{'params': self.ts_mixer_layer.parameters()}],
            *args, **kwargs
        )

    def predict(self, *args, **kwargs):
        # eval mode for extractor
        self.ts_mixer_layer.eval()
        return super().predict(*args, **kwargs)