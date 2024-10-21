from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def optimise_input(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    n_opt_epochs: int = 1000,
    learning_rate: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimise the input tensor to maximize the model's output while keeping x within bounds.

    Args:
        x (torch.Tensor): Current input tensor.
        y (torch.Tensor): Current output tensor.
        model (nn.Module): The trained model.
        lower_bound (torch.Tensor): Lower bound for x.
        upper_bound (torch.Tensor): Upper bound for x.
        n_opt_epochs (int, optional): Number of optimisation epochs. Defaults to 1000.
        learning_rate (float, optional): Learning rate for the optimiser. Defaults to 0.1.
        device (torch.device, optional): Device to perform optimisation on. Defaults to CPU.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Optimised x and corresponding y.
    """
    model.to(device)
    x_opt = x.clone().detach().requires_grad_(True).to(device)
    optimiser_opt = optim.Adam([x_opt], lr=learning_rate)

    for epoch in range(1, n_opt_epochs + 1):
        optimiser_opt.zero_grad()
        y_opt = model(x_opt)
        loss = -y_opt.sum() + torch.abs(x_opt.sum() - x.sum().to(device))
        loss.backward()
        optimiser_opt.step()

        with torch.no_grad():
            x_opt.clamp_(lower_bound.to(device), upper_bound.to(device))

        if epoch % 100 == 0 or epoch == 1:
            total_y = y.sum().item()
            print(f"Optimisation Epoch {epoch}/{n_opt_epochs}, Total y: {total_y:.2f}")

    optimised_x = x_opt.detach()
    optimised_y = model(optimised_x).detach()

    return optimised_x, optimised_y
