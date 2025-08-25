from torch.nn import Sequential, Linear, ReLU
from gpytorch.utils.grid import ScaleToBounds
from utils.models.MultistepSVGP import TSGPModel

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