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
