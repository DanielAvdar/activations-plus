import matplotlib.pyplot as plt
import torch

from activations_plus.simple import loglogish

x = torch.linspace(-3, 3, 200)
y = loglogish(x)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("LogLogish")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
fig.show()
