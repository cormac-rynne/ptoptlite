import torch
import torch.nn as nn


class LogLogModel(nn.Module):
    """
    A vectorized log-log model with learnable scaling and exponent parameters.

    Args:
        num_curves (int): Number of curves to model.
        scale_min (float, optional): Minimum scaling factor. Defaults to -4.
        scale_max (float, optional): Maximum scaling factor. Defaults to 4.
        exp_min (float, optional): Minimum exponent value. Defaults to 0.1.
        exp_max (float, optional): Maximum exponent value. Defaults to 1.0.
    """

    def __init__(self, num_curves, scale_min=-4, scale_max=4, exp_min=0.1, exp_max=1.0):
        super(LogLogModel, self).__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.exp_min = exp_min
        self.exp_max = exp_max
        self.scaling = nn.Parameter(torch.ones((num_curves, 1)))
        self.exponent = nn.Parameter(torch.ones((num_curves, 1)))

    def forward(self, x):
        clamped_scale = torch.clamp(self.scaling, min=self.scale_min, max=self.scale_max)
        clamped_exp = torch.clamp(self.exponent, min=self.exp_min, max=self.exp_max)
        return clamped_scale * x.pow(clamped_exp)
