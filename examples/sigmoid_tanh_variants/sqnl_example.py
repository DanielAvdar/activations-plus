import matplotlib.pyplot as plt
import torch

from activations_plus.simple.sigmoid_tanh_variants import sqnl

x = torch.linspace(-3, 3, 200)
y = sqnl(x)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("SQNL")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
fig.show()
