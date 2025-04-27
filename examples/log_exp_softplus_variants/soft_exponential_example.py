import matplotlib.pyplot as plt
import torch

from activations_plus.simple.log_exp_softplus_variants import soft_exponential

x = torch.linspace(-2, 2, 200)
y = soft_exponential(x)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("Soft Exponential")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
# fig.show()
