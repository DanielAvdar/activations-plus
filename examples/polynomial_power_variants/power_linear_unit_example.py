import matplotlib.pyplot as plt
import torch

from activations_plus.simple.polynomial_power_variants import power_linear_unit

x = torch.linspace(-3, 3, 200)
y = power_linear_unit(x)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("Power Linear Unit")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
fig.show()
