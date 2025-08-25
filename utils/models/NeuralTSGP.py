from torch.nn import Sequential, Linear, ReLU, Module, ModuleList, Flatten
from gpytorch.utils.grid import ScaleToBounds
from utils.models.MultistepSVGP import TSGPModel
from utils.models.TSMixer import MTSMixerBlock
from utils.models.PatchTST import PatchTST

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
    def __init__(self, inducing_points, horizon, num_latents_svgp, num_latents_lfe, grid_bounds=(-1., 1.), feature_extractor_type='large',
                 num_variables=None, time_steps=None):

        num_input_vars = inducing_points.size(-1)

        match feature_extractor_type:
            case 'tsmixer':
                if time_steps is None or num_variables is None:
                    raise ValueError("num_variables and time_steps must be provided for TSMixer feature extractor.")
                latent_feature_extractor = TSMixerFeatureExtractor(num_variables, time_steps, latent_dimension=num_latents_lfe)
            case 'large':
                latent_feature_extractor = LargeFeatureExtractor(num_input_vars, latent_dimension=num_latents_lfe)
            case 'patch_tst':
                latent_feature_extractor = PatchTST(num_variables, time_steps, output_steps=num_latents_lfe, return_embeddings=True)
            case _:
                raise ValueError(f"Unknown feature_extractor_type: {feature_extractor_type}. Choose 'large' or 'tsmixer'.")


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
