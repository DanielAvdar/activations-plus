import matplotlib.pyplot as plt
import torch

from activations_plus.simple import silu

x = torch.linspace(-3, 3, 200)
y = silu(x)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("SiLU (Swish)")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
fig.show()  # This will be mocked in tests
