import matplotlib.pyplot as plt
import torch

from activations_plus.simple.relu_variants import rrelu

x = torch.linspace(-3, 3, 200)
y = rrelu(x, a=0.1)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("Randomized Leaky ReLU (a=0.1)")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
fig.show()  # This will be mocked in tests
