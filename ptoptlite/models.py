# models.py
import datetime as dt

import torch
import torch.nn as nn
from pyspark.sql import functions as f


class GeneralisedModel(nn.Module):
    def __init__(self, curve_function, param_names):
        super().__init__()
        self.curve_function = curve_function
        self.param_names = param_names
        self.args_lst = None

    def forward(self, x):
        return self.curve_function(x, *self.args_lst)

    def args(self):
        self.args_lst = [getattr(self, param) for param in self.param_names]


def build_model(model_params, param_names, curve_function_pt, device):
    """
    Builds a GeneralisedModel using PySpark DataFrame operations instead of converting to pandas.

    Args:
        model_params (Pyspark DataFrame): Spark DataFrame containing a 'params' column with dictionaries.
        param_names (list): List of parameter names to extract from the 'params' column.
        curve_function_pt (callable): The curve function to be used in the model.

    Returns:
        GeneralisedModel: The constructed model with parameters set as PyTorch tensors.
    """

    # Step 2: Select the parameter columns and collect the data
    selected_cols = [f.col(param) for param in param_names]
    param_values = model_params.select(selected_cols).collect()

    # Step 4: Initialize the model
    model = GeneralisedModel(curve_function=curve_function_pt, param_names=param_names).to(device)

    # Step 3: Convert collected data directly to PyTorch tensors
    for param_name in param_names:
        # Extract the column values and handle potential None values
        values = [row[param_name] if row[param_name] is not None else float("nan") for row in param_values]
        # Convert the list to a PyTorch tensor with appropriate shape and dtype
        tensor = nn.Parameter(torch.tensor(values, dtype=torch.float32).unsqueeze(1)).to(device)  # Shape: (N, 1)
        model.register_parameter(param_name, tensor)

    model.args()

    return model


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
