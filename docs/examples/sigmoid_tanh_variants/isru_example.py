import matplotlib.pyplot as plt
import torch

from activations_plus.simple import isru

x = torch.linspace(-3, 3, 200)
y = isru(x, alpha=1.0)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("ISRU (alpha=1.0)")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
fig.show()  # This will be mocked in tests
