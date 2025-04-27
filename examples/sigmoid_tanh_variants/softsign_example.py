import matplotlib.pyplot as plt
import torch

from activations_plus.simple.sigmoid_tanh_variants import softsign

x = torch.linspace(-6, 6, 200)
y = softsign(x)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("Softsign")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
# fig.show()
