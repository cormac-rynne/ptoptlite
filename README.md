# ptoptlite

![PyPI](https://img.shields.io/pypi/v/ptoptlite)
![License](https://img.shields.io/github/license/yourusername/ptoptlite)
![Python Version](https://img.shields.io/pypi/pyversions/ptoptlite)
![Build Status](https://github.com/yourusername/ptoptlite/workflows/CI/badge.svg)

**ptoptlite** is a lightweight and flexible PyTorch-based package designed for parameter optimization and model training across a variety of models. Whether you're working with linear models, neural networks, or custom architectures, ptoptlite provides a streamlined interface to initialize, train, and optimize your models efficiently.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Example Usage](#example-usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Model Agnostic**: Supports a wide range of PyTorch models, not limited to log-log models.
- **Multi-Start Initialization**: Enhance model performance by initializing parameters multiple times and selecting the best.
- **Flexible Training Loop**: Customize training parameters such as epochs, learning rates, loss functions, and optimizers.
- **Input Optimization**: Optimize input tensors to maximize or minimize model outputs within specified bounds.
- **Device Management**: Seamlessly switch between CPU and GPU for computations.
- **Extensible**: Easily extend the package to accommodate new models and optimization strategies.
- **Comprehensive Logging**: Integrated logging for monitoring training and optimization progress.
- **Unit Tested**: Reliable performance ensured through extensive unit tests.

## Installation

<!-- You can install **ptoptlite** via [PyPI](https://pypi.org/) using `pip`: -->

```bash
pip install https://github.com/cormac-rynne/potptlite.git
```



## Examples
### [Link to Colab](https://colab.research.google.com/drive/17PWMMrHbzrWGrqOCq1KvfZwsXHzBXsLK?usp=sharing)

### Example Usage
```python
import torch
from ptoptlite.models import LogLogModel
from ptoptlite.utils import multistart
from ptoptlite.training import train_model
from ptoptlite.optimisation import optimise_input


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example data
num_curves = 10
x = torch.linspace(1, 10, steps=100).unsqueeze(0).repeat(num_curves, 1)  # Shape: (num_curves, 100)
y = torch.randn_like(x)  # Replace with actual target data

num_inits = 100
model = LogLogModel(num_curves).to(device)
model = multistart(num_curves, num_inits, model, x, y, device=device)
model = train_model(x, y, model, epochs=15000, device=device)

current_x = x.mean(dim=1, keepdim=True).detach().to(device)
current_y = model(current_x)

lower_bound = 0.8 * current_x
upper_bound = 1.2 * current_x

optimized_x, optimized_y = optimise_input(
    current_x, current_y, model, lower_bound, upper_bound, n_opt_epochs=1000, device=device
)

print("Current x sum:", current_x.sum().item(), "Optimized x sum:", optimized_x.sum().item())
print("Current y sum:", current_y.sum().item(), "Optimized y sum:", optimized_y.sum().item())
```


