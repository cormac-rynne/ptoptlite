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

You can install **ptoptlite** via [PyPI](https://pypi.org/) using `pip`:

```bash
pip install ptoptlite
