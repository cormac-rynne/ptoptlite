from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    epochs: int = 1000,
    learning_rate: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Train the LogLogModel using AdamW optimizer and Smooth L1 Loss.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Target tensor.
        model (nn.Module): The LogLogModel instance.
        epochs (int, optional): Number of training epochs. Defaults to 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        device (torch.device, optional): Device to train on. Defaults to CPU.

    Returns:
        nn.Module: The trained model.
    """
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(x.to(device))
        loss = nn.functional.smooth_l1_loss(y_pred, y.to(device))
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == 1:
            with torch.no_grad():
                per_curve_loss = torch.mean((y_pred - y.to(device)) ** 2, dim=1)
                mean_per_curve_loss = per_curve_loss.mean().item()
                print(
                    f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}, Mean Per-Curve Loss: {mean_per_curve_loss:.6f}"
                )

    return model
