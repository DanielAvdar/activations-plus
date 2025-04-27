import matplotlib.pyplot as plt
import torch

from activations_plus.simple.sigmoid_tanh_variants import softplus

x = torch.linspace(-6, 6, 200)
y = softplus(x)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("Softplus")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
# fig.show()
