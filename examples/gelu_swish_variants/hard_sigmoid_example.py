import matplotlib.pyplot as plt
import torch

from activations_plus.simple import hard_sigmoid

x = torch.linspace(-3, 3, 200)
y = hard_sigmoid(x)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("Hard Sigmoid")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
fig.show()  # This will be mocked in tests
