from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def multistart(
    num_curves: int,
    num_inits: int,
    model: nn.Model,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> nn.Model:
    """
    Perform multi-start initialization to find the best parameters for each curve.

    Args:
        num_curves (int): Number of curves.
        num_inits (int): Number of initializations.
        model (nn.Model): The model instance.
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Target tensor.
        device (torch.device, optional): Device to perform computations on. Defaults to CPU.

    Returns:
        nn.Model: Model with optimised parameters.
    """
    for name, param in model.named_parameters():
        new_param = nn.Parameter(torch.rand((num_inits, num_curves, 1)) * 4 - 3).to(device)
        setattr(model, name, new_param)

    model.to(device)
    model.eval()
    with torch.no_grad():
        y_ = model(x.to(device))  # Shape: (num_curves, 1)
        mse = ((y.unsqueeze(0) - y_) ** 2).mean(dim=2)  # Shape: (num_inits, num_curves)
        idx = mse.argmin(dim=0)  # Shape: (num_curves,)

    for name, params in model.named_parameters():
        selected_params = params[idx, torch.arange(num_curves), 0].unsqueeze(-1)
        setattr(model, name, nn.Parameter(selected_params).to(device))

    return model
